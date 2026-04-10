# Copyright 2024 The Brax Authors.
# Modified for PPO+CRL with Contrastive Reward Shaping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO with Contrastive Reward Shaping training.

Combines PPO with Contrastive RL (CRL) through reward shaping:
- Adds CRL's energy as intrinsic reward to PPO
- Provides dense learning signals for sparse-reward tasks
- Addresses vanishing gradient problem in sparse-reward environments

See: https://arxiv.org/pdf/1707.06347.pdf (PPO)
See: https://arxiv.org/abs/2106.01404 (CRL)

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import acting, gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.agents.crl.networks import Encoder
from jaxgcrl.agents.crl.losses import energy_fn, contrastive_loss_fn


def create_crl_encoders(obs_dim: int, action_dim: int, repr_dim: int = 64):
    """Create CRL encoder networks for state-action and goal representations."""
    sa_encoder = Encoder(repr_dim=repr_dim)
    g_encoder = Encoder(repr_dim=repr_dim)
    return sa_encoder, g_encoder


def compute_intrinsic_reward(state, action, goal, crl_params, sa_encoder, g_encoder, energy_fn_name):
    """Compute intrinsic reward from CRL encoders."""
    sa_repr = sa_encoder.apply(crl_params["sa_encoder"], jnp.concatenate([state, action], axis=-1))
    g_repr = g_encoder.apply(crl_params["g_encoder"], goal)
    return energy_fn(energy_fn_name, sa_repr, g_repr)


def update_crl_encoders(config, sa_encoder, g_encoder, transitions, crl_params, crl_optimizer_state, key):
    """Update CRL encoders using contrastive loss."""
    
    def crl_loss(crl_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            crl_params["sa_encoder"],
            crl_params["g_encoder"],
        )

        state = transitions.observation[:, :config["state_size"]]
        action = transitions.action
        goal = transitions.observation[:, config["state_size"]:]

        sa_repr = sa_encoder.apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = g_encoder.apply(g_encoder_params, goal)

        # InfoNCE loss
        logits = energy_fn(config["energy_fn_name"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        crl_loss, has_aux=True
    )(crl_params, transitions, key)
    
    # Apply gradients
    updates, new_crl_optimizer_state = config["crl_optimizer"].update(grad, crl_optimizer_state, crl_params)
    new_crl_params = optax.apply_updates(crl_params, updates)
    
    metrics = {
        "crl/categorical_accuracy": jnp.mean(correct),
        "crl/logits_pos": logits_pos,
        "crl/logits_neg": logits_neg,
        "crl/logsumexp": logsumexp.mean(),
        "crl/loss": loss,
    }

    return new_crl_params, new_crl_optimizer_state, metrics

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray
    # CRL components
    crl_optimizer_state: optax.OptState
    crl_params: dict  # Contains sa_encoder and g_encoder parameters


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


@dataclass
class PPO_CRL_RewardShaping:
    """PPO with Contrastive Reward Shaping agent.

    Combines PPO with CRL through intrinsic reward shaping:
    total_reward = sparse_environment_reward + β * intrinsic_reward
    where intrinsic_reward = energy(φ(s,a), ψ(g))
    
    φ(s,a): state-action encoder
    ψ(g): goal encoder
    energy: similarity measure (l2, dot, cosine, norm)
    Args:
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each train_env. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new train_env rollout
      num_resets_per_eval: the number of train_env resets to run between each
        eval. The train_env resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      deterministic_eval: whether to run the eval with a deterministic policy
      network_factory: function that generates networks for policy and value
        functions
      progress_fn: a user-defined callback function for reporting/plotting metrics
      normalize_advantage: whether to normalize advantage estimate
      restore_checkpoint_path: the path used to restore previous model params
    """

    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    discounting: float = 0.9
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    deterministic_eval: bool = False
    normalize_advantage: bool = True
    restore_checkpoint_path: Optional[str] = None
    train_step_multiplier = 1
    # CRL-specific hyperparameters
    contrastive_coeff: float = 1.0          # β coefficient for intrinsic reward
    repr_dim: int = 64                      # Representation dimension for encoders
    contrastive_loss_fn: str = "sym_infonce" # Contrastive loss function
    energy_fn_name: str = "l2"              # Energy function for similarity
    crl_learning_rate: float = 3e-4         # Learning rate for CRL encoders
    logsumexp_penalty_coeff: float = 0.1    # Penalty coefficient for logsumexp
    normalize_intrinsic_reward: bool = True  # Normalize intrinsic rewards per batch

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        """PPO training.

        Args:
          train_env: the train_env to train
          eval_env: an optional train_env for eval only, defaults to `train_env`
          randomization_fn: a user-defined callback function that generates randomized environments
          progress_fn: a user-defined callback function for reporting/plotting metrics

        Returns:
          Tuple of (make_policy function, network params, metrics)
        """
        assert self.batch_size * self.num_minibatches % config.num_envs == 0
        xt = time.time()
        network_factory = ppo_networks.make_ppo_networks

        process_count = jax.process_count()
        process_id = jax.process_index()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count
        if config.max_devices_per_host:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)

        logging.info(
            "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
            jax.device_count(),
            process_count,
            process_id,
            local_device_count,
            local_devices_to_use,
        )
        device_count = local_devices_to_use * process_count

        # The number of train_env steps executed for every training step.
        utd_ratio = self.batch_size * self.unroll_length * self.num_minibatches * config.action_repeat
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of training_step calls per training_epoch call.
        # equals to ceil(total_env_steps / (num_evals * utd_ratio *
        #                                 num_resets_per_eval))
        num_training_steps_per_epoch = np.ceil(
            config.total_env_steps / (num_evals_after_init * utd_ratio * max(self.num_resets_per_eval, 1))
        ).astype(int)

        key = jax.random.PRNGKey(config.seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        # key_networks should be global, so that networks are initialized the same
        # way for different processes.
        key_policy, key_value, key_crl = jax.random.split(global_key, 3)
        del global_key

        assert config.num_envs % device_count == 0

        v_randomization_fn = None
        if randomization_fn is not None:
            randomization_batch_size = config.num_envs // local_device_count
            # all devices gets the same randomization rng
            randomization_rng = jax.random.split(key_env, randomization_batch_size)
            v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

        if isinstance(train_env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        env = train_env
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        reset_fn = jax.jit(jax.vmap(env.reset))
        key_envs = jax.random.split(key_env, config.num_envs // process_count)
        key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
        env_state = reset_fn(key_envs)

        normalize = lambda x, y: x
        if self.normalize_observations:
            normalize = running_statistics.normalize
        ppo_network = network_factory(
            env_state.obs.shape[-1],
            env.action_size,
            preprocess_observations_fn=normalize,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)

        # Create CRL encoders
        obs_dim = env_state.obs.shape[-1]
        # For goal-conditioned environments, we need to determine state and goal dimensions
        # This assumes the observation is [state, goal] concatenated
        # Try to get state size from environment if available, otherwise use heuristic
        try:
            # Some environments might have state_size attribute
            state_dim = train_env.state_size
        except AttributeError:
            # Default heuristic: assume half is state, half is goal
            state_dim = obs_dim // 2
        goal_dim = obs_dim - state_dim
        
        sa_encoder, g_encoder = create_crl_encoders(
            obs_dim=state_dim + env.action_size,
            action_dim=env.action_size,
            repr_dim=self.repr_dim
        )

        optimizer = optax.adam(learning_rate=self.learning_rate)
        crl_optimizer = optax.adam(learning_rate=self.crl_learning_rate)

        # Create modified loss function with intrinsic rewards
        def ppo_crl_loss(params, normalizer_params, data, key, crl_params):
            """PPO loss with contrastive-shaped rewards."""
            # Extract state, action, goal from observations
            # Assuming observation = [state, goal]
            state = data.observation[:, :state_dim]
            action = data.action
            goal = data.observation[:, state_dim:]
            
            # Compute intrinsic reward
            intrinsic_reward = compute_intrinsic_reward(
                state, action, goal, crl_params, sa_encoder, g_encoder, self.energy_fn_name
            )
            
            # Normalize intrinsic reward if enabled
            if self.normalize_intrinsic_reward:
                intrinsic_reward = (intrinsic_reward - intrinsic_reward.mean()) / (intrinsic_reward.std() + 1e-8)
            
            # Combine with sparse environment reward
            total_reward = data.reward + self.contrastive_coeff * intrinsic_reward
            
            # Use shaped reward in PPO loss
            shaped_data = data._replace(reward=total_reward)
            
            # Track intrinsic reward metrics
            intrinsic_metrics = {
                "intrinsic_reward/mean": intrinsic_reward.mean(),
                "intrinsic_reward/std": intrinsic_reward.std(),
                "intrinsic_reward/max": intrinsic_reward.max(),
                "intrinsic_reward/min": intrinsic_reward.min(),
                "total_reward/mean": total_reward.mean(),
            }
            
            # Compute original PPO loss with shaped rewards
            loss, metrics = ppo_losses.compute_ppo_loss(
                ppo_network=ppo_network,
                params=params,
                normalizer_params=normalizer_params,
                data=shaped_data,
                key=key,
                entropy_cost=self.entropy_cost,
                discounting=self.discounting,
                reward_scaling=self.reward_scaling,
                gae_lambda=self.gae_lambda,
                clipping_epsilon=self.clipping_epsilon,
                normalize_advantage=self.normalize_advantage,
            )
            
            # Combine metrics
            metrics.update(intrinsic_metrics)
            return loss, metrics

        loss_fn = functools.partial(ppo_crl_loss)

        def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, crl_params, crl_optimizer_state, key = carry
            key, key_loss, key_crl = jax.random.split(key, 3)
            
            # Update PPO with shaped rewards
            # Create a wrapper function that includes CRL params
            def loss_fn_with_crl(params, normalizer_params, data, key):
                return loss_fn(params, normalizer_params, data, key, crl_params)
            
            # Create gradient update function with wrapped loss
            ppo_gradient_update_fn = gradients.gradient_update_fn(
                loss_fn_with_crl,
                optimizer,
                pmap_axis_name=_PMAP_AXIS_NAME,
                has_aux=True,
            )
            
            (_, metrics), params, optimizer_state = ppo_gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

            # Update CRL encoders
            crl_config = {
                "state_size": state_dim,
                "energy_fn_name": self.energy_fn_name,
                "contrastive_loss_fn": self.contrastive_loss_fn,
                "logsumexp_penalty_coeff": self.logsumexp_penalty_coeff,
                "crl_optimizer": crl_optimizer,
            }
            new_crl_params, new_crl_optimizer_state, crl_metrics = update_crl_encoders(
                crl_config, sa_encoder, g_encoder, data, crl_params, crl_optimizer_state, key_crl
            )
            
            # Combine metrics
            metrics.update(crl_metrics)

            return (optimizer_state, params, new_crl_params, new_crl_optimizer_state, key), metrics

        def update_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, crl_params, crl_optimizer_state, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
                return x

            shuffled_data = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, params, crl_params, crl_optimizer_state, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params),
                (optimizer_state, params, crl_params, crl_optimizer_state, key_grad),
                shuffled_data,
                length=self.num_minibatches,
            )
            return (optimizer_state, params, crl_params, crl_optimizer_state, key), metrics

        def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
        ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
            training_state, state, key = carry
            update_key, key_generate_unroll, new_key = jax.random.split(key, 3)

            policy = make_policy(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                )
            )

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    self.unroll_length,
                    extra_fields=("truncation",),
                )
                return (next_state, next_key), data

            (state, _), data = jax.lax.scan(
                f,
                (state, key_generate_unroll),
                (),
                length=self.batch_size * self.num_minibatches // config.num_envs,
            )
            # Have leading dimensions (batch_size * num_minibatches, unroll_length)
            data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            assert data.discount.shape[1:] == (self.unroll_length,)

            # Update normalization params and normalize observations.
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            (optimizer_state, params, crl_params, crl_optimizer_state, _), metrics = jax.lax.scan(
                functools.partial(
                    update_step,
                    data=data,
                    normalizer_params=normalizer_params,
                ),
                (
                    training_state.optimizer_state,
                    training_state.params,
                    training_state.crl_params,
                    training_state.crl_optimizer_state,
                    update_key,
                ),
                (),
                length=self.num_updates_per_batch,
            )

            new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + utd_ratio,
                crl_optimizer_state=crl_optimizer_state,
                crl_params=crl_params,
            )
            return (new_training_state, state, new_key), metrics

        def training_epoch(
            training_state: TrainingState,
            state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            nonlocal training_walltime
            t = time.time()
            training_state, env_state = _strip_weak_type((training_state, env_state))
            result = training_epoch(training_state, env_state, key)
            training_state, env_state, metrics = _strip_weak_type(result)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                num_training_steps_per_epoch * utd_ratio * max(self.num_resets_per_eval, 1)
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        # Initialize model params and training state.
        init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )
        
        # Initialize CRL encoder parameters
        key_sa, key_g = jax.random.split(key_crl)
        init_crl_params = {
            "sa_encoder": sa_encoder.init(key_sa, jnp.zeros((1, state_dim + env.action_size))),
            "g_encoder": g_encoder.init(key_g, jnp.zeros((1, goal_dim))),
        }

        training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
            params=init_params,
            normalizer_params=running_statistics.init_state(
                specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
            ),
            env_steps=0,
            crl_optimizer_state=crl_optimizer.init(init_crl_params),
            crl_params=init_crl_params,
        )

        if config.total_env_steps == 0:
            return (
                make_policy,
                (
                    training_state.normalizer_params,
                    training_state.params,
                ),
                {},
            )

        if self.restore_checkpoint_path is not None and epath.Path(self.restore_checkpoint_path).exists():
            logging.info(
                "restoring from checkpoint %s",
                self.restore_checkpoint_path,
            )
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            # Try to restore both PPO and CRL parameters
            target = training_state.normalizer_params, init_params, init_crl_params
            try:
                (normalizer_params, init_params, init_crl_params) = orbax_checkpointer.restore(
                    self.restore_checkpoint_path, item=target
                )
                training_state = training_state.replace(
                    normalizer_params=normalizer_params, 
                    params=init_params,
                    crl_params=init_crl_params
                )
            except:
                # Fallback: only restore PPO parameters (for compatibility with old checkpoints)
                logging.warning("Could not restore CRL parameters, using initialization")
                target = training_state.normalizer_params, init_params
                (normalizer_params, init_params) = orbax_checkpointer.restore(
                    self.restore_checkpoint_path, item=target
                )
                training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)

        training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use],
        )

        if not eval_env:
            eval_env = train_env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(eval_key, self.num_eval_envs),
            )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        evaluator = Evaluator(
            eval_env,
            functools.partial(
                make_policy,
                deterministic=self.deterministic_eval,
            ),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )
                ),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )
                ),
                unwrapped_env,
            )

        training_metrics = {}
        training_walltime = 0
        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info(
                "starting iteration %s %s",
                eval_epoch_num,
                time.time() - xt,
            )

            for _ in range(max(self.num_resets_per_eval, 1)):
                # optimization
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
                (
                    training_state,
                    env_state,
                    training_metrics,
                ) = training_epoch_with_timing(
                    training_state,
                    env_state,
                    epoch_keys,
                )
                current_step = int(_unpmap(training_state.env_steps))

                key_envs = jax.vmap(
                    lambda x, s: jax.random.split(x[0], s),
                    in_axes=(0, None),
                )(key_envs, key_envs.shape[1])
                # TODO: move extra reset logic to the AutoResetWrapper.
                env_state = reset_fn(key_envs) if self.num_resets_per_eval > 0 else env_state

                if process_id == 0:
                    # Run evals.
                    metrics = evaluator.run_evaluation(
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                training_state.params.policy,
                            )
                        ),
                        training_metrics,
                    )
                    do_render = (eval_epoch_num % config.visualization_interval) == 0
                    progress_fn(
                        current_step,
                        metrics,
                        make_policy,
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                training_state.params.policy,
                            )
                        ),
                        unwrapped_env,
                        do_render=do_render,
                    )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
            )
        )
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
