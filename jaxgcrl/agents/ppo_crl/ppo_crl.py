"""PPO + CRL: Proximal Policy Optimization with a CRL contrastive auxiliary loss.

CRL's core idea is that φ(s,a)·ψ(g) estimates discounted future-state
occupancy — a dense, reward-free signal for "how likely is (s,a) to reach g."
We use it two ways:

1. As an intrinsic reward added to the environment reward before GAE:
       reward_total = env_reward + intrinsic_reward_coeff * φ(s,a)·ψ(g)
   This gives PPO a dense advantage signal even when env rewards are sparse,
   which is the main reason CRL sidesteps sparse-reward exploration problems.

2. As a learned goal feature: the policy and value heads consume
   [normalized_state | g_encoder(goal)], so PPO benefits from a goal
   embedding that captures reachability structure.

The two losses are trained on separate schedules with a single shared Adam
optimizer:

    - PPO loss runs num_updates_per_batch * num_minibatches gradient steps per
      training_step (standard PPO epochs over the fresh rollout). It touches
      policy_head and value_head only.
    - Contrastive loss runs num_crl_updates_per_step gradient steps per
      training_step, each on a fresh sub-batch drawn from a pool sampled out
      of the replay buffer. It touches sa_encoder and g_encoder.

The sa_encoder and g_encoder are trained purely by the contrastive loss. PPO
consumes their outputs (both as goal features and as the intrinsic reward)
under stop_gradient, so PPO updates cannot destabilize the representation CRL
is simultaneously learning.

PPO's clipped-surrogate update stays strictly on-policy (uses only the fresh
rollout). The contrastive loss is off-policy; the replay buffer accumulates
full training history. This is sound because the contrastive loss learns
environment dynamics, not the policy — old transitions stay valid.
"""

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from brax.training import acting, types
from brax.training import distribution as brax_distribution
from brax.training import networks as brax_networks
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1

from jaxgcrl.agents.crl.losses import contrastive_loss_fn as _contrastive_loss_fn
from jaxgcrl.agents.crl.losses import energy_fn as _energy_fn
from jaxgcrl.agents.crl.networks import Encoder
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


@flax.struct.dataclass
class PPOCRLTrainingState:
    optimizer_state: optax.OptState
    params: Any
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


@flax.struct.dataclass
class BufferTransition:
    """Minimal transition stored in the contrastive replay buffer."""
    observation: jnp.ndarray
    action: jnp.ndarray
    traj_id: jnp.ndarray


def sample_contrastive_pairs(gamma, state_size, goal_indices, rollout_obs, rollout_actions, traj_ids, key):
    """For a single trajectory of length T, sample (s_t, a_t, s_future) triples."""
    seq_len = rollout_obs.shape[0]
    arrangement = jnp.arange(seq_len)

    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [traj_ids[:, jnp.newaxis].T] * seq_len, axis=0
    )
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(key, jnp.log(probs))
    future_obs = jnp.take(rollout_obs, goal_index[:-1], axis=0)
    goal = future_obs[:, jnp.asarray(goal_indices)]
    state = rollout_obs[:-1, :state_size]
    action = rollout_actions[:-1]
    return state, action, goal


@dataclass
class PPO_CRL:
    """PPO with a shared goal-embedding network trained by a contrastive auxiliary loss."""

    # PPO hyperparameters
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True
    normalize_observations: bool = True
    reward_scaling: float = 1.0
    deterministic_eval: bool = False
    policy_hidden_width: int = 256
    policy_hidden_depth: int = 4
    value_hidden_width: int = 256
    value_hidden_depth: int = 4

    # CRL auxiliary
    contrastive_coeff: float = 1.0
    repr_dim: int = 64
    contrastive_loss_fn: Literal[
        "fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"
    ] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"
    logsumexp_penalty_coeff: float = 0.1
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = False
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    contrastive_batch_size: int = 256
    num_crl_updates_per_step: int = 1024
    intrinsic_reward_coeff: float = 1.0
    crl_actor_coeff: float = 1.0

    train_step_multiplier = 1

    def train_fn(
        self,
        config,
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn=None,
        progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
    ):
        assert self.batch_size * self.num_minibatches % config.num_envs == 0
        xt = time.time()

        key = jax.random.PRNGKey(config.seed)
        key, local_key, buffer_key, eval_key = jax.random.split(key, 4)
        key_policy, key_value, key_g, key_sa = jax.random.split(local_key, 4)

        if isinstance(train_env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        unwrapped_env = train_env
        env = TrajectoryIdWrapper(train_env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        reset_fn = jax.jit(env.reset)
        key_envs = jax.random.split(key, config.num_envs)
        env_state = reset_fn(key_envs)

        # --- Dimensions ---
        state_size = unwrapped_env.state_dim
        goal_indices_static = tuple(int(i) for i in unwrapped_env.goal_indices)
        goal_size = len(goal_indices_static)
        obs_size = env_state.obs.shape[-1]
        action_size = env.action_size
        assert obs_size == state_size + goal_size, (
            f"obs_size ({obs_size}) != state_size ({state_size}) + goal_size ({goal_size})"
        )

        logging.info(
            "ppo_crl dimensions: state_size=%d goal_size=%d action_size=%d repr_dim=%d",
            state_size, goal_size, action_size, self.repr_dim,
        )

        # --- Networks ---
        parametric_action_distribution = brax_distribution.NormalTanhDistribution(
            event_size=action_size
        )

        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        policy_module = brax_networks.MLP(
            layer_sizes=[self.policy_hidden_width] * self.policy_hidden_depth
            + [parametric_action_distribution.param_size],
            activation=nn.swish,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )
        value_module = brax_networks.MLP(
            layer_sizes=[self.value_hidden_width] * self.value_hidden_depth + [1],
            activation=nn.swish,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )

        if self.normalize_observations:
            def normalize_obs(obs, params):
                return running_statistics.normalize(obs, params)
        else:
            def normalize_obs(obs, params):
                return obs

        def augmented_features(params, normalizer_params, obs):
            """Observation → [normalized_state | stop_grad(g_encoder(raw_goal))].

            The g_encoder is trained by CRL on unnormalized obs, so we feed it
            raw goals here to match the training distribution. Normalized state
            still flows to the policy / value heads since PPO benefits from obs
            normalization. stop_gradient prevents PPO updates from destabilizing
            the representation CRL is simultaneously learning.
            """
            normed = normalize_obs(obs, normalizer_params)
            state_n = normed[..., :state_size]
            goal_raw = obs[..., state_size:]
            g_feat = g_encoder.apply(params["g_encoder"], goal_raw)
            g_feat = jax.lax.stop_gradient(g_feat)
            return jnp.concatenate([state_n, g_feat], axis=-1)

        def apply_policy(params, normalizer_params, obs):
            x = augmented_features(params, normalizer_params, obs)
            return policy_module.apply(params["policy_head"], x)

        def apply_value(params, normalizer_params, obs):
            x = augmented_features(params, normalizer_params, obs)
            return jnp.squeeze(value_module.apply(params["value_head"], x), axis=-1)

        # --- Parameter initialization ---
        dummy_goal = jnp.zeros((1, goal_size))
        dummy_augmented = jnp.zeros((1, state_size + self.repr_dim))
        dummy_sa = jnp.zeros((1, state_size + action_size))

        init_params = {
            "g_encoder": g_encoder.init(key_g, dummy_goal),
            "sa_encoder": sa_encoder.init(key_sa, dummy_sa),
            "policy_head": policy_module.init(key_policy, dummy_augmented),
            "value_head": value_module.init(key_value, dummy_augmented),
        }

        normalizer_params = running_statistics.init_state(
            specs.Array((obs_size,), jnp.dtype("float32"))
        )

        optimizer = optax.adam(learning_rate=self.learning_rate)
        optimizer_state = optimizer.init(init_params)

        training_state = PPOCRLTrainingState(
            optimizer_state=optimizer_state,
            params=init_params,
            normalizer_params=normalizer_params,
            env_steps=jnp.zeros(()),
        )

        # --- Inference / rollout policy ---
        def make_policy(params_tuple, deterministic: bool = False):
            normalizer_params, ppo_params = params_tuple

            def policy(obs, key_sample):
                logits = apply_policy(ppo_params, normalizer_params, obs)
                if deterministic:
                    return parametric_action_distribution.mode(logits), {}
                raw_actions = parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
                )
                log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
                postprocessed = parametric_action_distribution.postprocess(raw_actions)
                return postprocessed, {
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                }

            return policy

        # --- Replay buffer for off-policy contrastive pairs ---
        dummy_buffer_transition = BufferTransition(
            observation=jnp.zeros(obs_size),
            action=jnp.zeros(action_size),
            traj_id=jnp.zeros(()),
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_buffer_transition,
                sample_batch_size=self.contrastive_batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # --- Losses ---
        def compute_ppo_loss(params, normalizer_params, data, key):
            # Put time dim first: (batch, unroll) → (unroll, batch)
            data_t = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

            policy_logits = apply_policy(params, normalizer_params, data_t.observation)
            baseline = apply_value(params, normalizer_params, data_t.observation)
            terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data_t.next_observation)
            bootstrap_value = apply_value(params, normalizer_params, terminal_obs)

            # Intrinsic reward: contrastive similarity between (s, a) and the
            # agent's current goal. Dense signal that converts CRL's learned
            # future-occupancy estimate into PPO's advantage computation.
            # Encoders see raw obs to match their training distribution;
            # stop_gradient prevents PPO from backpropping through them.
            obs_raw = data_t.observation
            state_raw = obs_raw[..., :state_size]
            goal_raw = obs_raw[..., state_size:]
            sa_input = jnp.concatenate([state_raw, data_t.action], axis=-1)
            sa_repr = sa_encoder.apply(params["sa_encoder"], sa_input)
            g_repr = g_encoder.apply(params["g_encoder"], goal_raw)
            intrinsic = _energy_fn(self.energy_fn, sa_repr, g_repr)
            intrinsic = jax.lax.stop_gradient(intrinsic)

            rewards = (data_t.reward + self.intrinsic_reward_coeff * intrinsic) * self.reward_scaling
            truncation = data_t.extras["state_extras"]["truncation"]
            termination = (1 - data_t.discount) * (1 - truncation)

            target_log_probs = parametric_action_distribution.log_prob(
                policy_logits, data_t.extras["policy_extras"]["raw_action"]
            )
            behaviour_log_probs = data_t.extras["policy_extras"]["log_prob"]

            vs, advantages = ppo_losses.compute_gae(
                truncation=truncation,
                termination=termination,
                rewards=rewards,
                values=baseline,
                bootstrap_value=bootstrap_value,
                lambda_=self.gae_lambda,
                discount=self.discounting,
            )
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            rho = jnp.exp(target_log_probs - behaviour_log_probs)
            surrogate1 = rho * advantages
            surrogate2 = jnp.clip(rho, 1 - self.clipping_epsilon, 1 + self.clipping_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

            v_error = vs - baseline
            v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

            entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, key))
            entropy_loss = self.entropy_cost * -entropy

            ppo_loss = policy_loss + v_loss + entropy_loss
            metrics = {
                "ppo_loss": ppo_loss,
                "policy_loss": policy_loss,
                "v_loss": v_loss,
                "entropy_loss": entropy_loss,
            }
            return ppo_loss, metrics

        def compute_crl_loss(params, normalizer_params, contrastive_batch, actor_key):
            states_mb, actions_mb, goals_mb = contrastive_batch

            # --- Critic: InfoNCE on stored (s, a, future_goal) triples ---
            sa_repr = sa_encoder.apply(
                params["sa_encoder"],
                jnp.concatenate([states_mb, actions_mb], axis=-1),
            )
            g_repr = g_encoder.apply(params["g_encoder"], goals_mb)
            logits_crl = _energy_fn(self.energy_fn, sa_repr[:, None, :], g_repr[None, :, :])
            crl_loss = _contrastive_loss_fn(self.contrastive_loss_fn, logits_crl)
            logsumexp = jax.nn.logsumexp(logits_crl + 1e-6, axis=1)
            crl_loss = crl_loss + self.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)

            eye = jnp.eye(logits_crl.shape[0])
            crl_accuracy = jnp.mean(jnp.argmax(logits_crl, axis=1) == jnp.argmax(eye, axis=1))

            critic_term = self.contrastive_coeff * crl_loss

            # --- Actor: maximize φ(s, π(s)) · ψ(g) w.r.t. policy params, with
            # encoders frozen. Runs on replay-buffer states (off-policy, diverse)
            # rather than on PPO's fresh rollout, so the actor gets signal from
            # historical experience even when the current policy is stuck.
            obs_reconstructed = jnp.concatenate([states_mb, goals_mb], axis=-1)
            policy_logits = apply_policy(params, normalizer_params, obs_reconstructed)
            a_sample_raw = parametric_action_distribution.sample_no_postprocessing(
                policy_logits, actor_key
            )
            a_sample = parametric_action_distribution.postprocess(a_sample_raw)

            frozen_sa_params = jax.lax.stop_gradient(params["sa_encoder"])
            frozen_g_params = jax.lax.stop_gradient(params["g_encoder"])
            sa_input_pi = jnp.concatenate([states_mb, a_sample], axis=-1)
            sa_repr_pi = sa_encoder.apply(frozen_sa_params, sa_input_pi)
            g_repr_frozen = g_encoder.apply(frozen_g_params, goals_mb)
            qf_pi = _energy_fn(self.energy_fn, sa_repr_pi, g_repr_frozen)
            crl_actor_loss = -jnp.mean(qf_pi)

            actor_term = self.crl_actor_coeff * crl_actor_loss

            total = critic_term + actor_term
            metrics = {
                "contrastive_loss": crl_loss,
                "contrastive_accuracy": crl_accuracy,
                "crl_actor_loss": crl_actor_loss,
            }
            return total, metrics

        ppo_loss_and_grad = jax.value_and_grad(compute_ppo_loss, has_aux=True)
        crl_loss_and_grad = jax.value_and_grad(compute_crl_loss, has_aux=True)

        def apply_update(params, optimizer_state, grads):
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            return params, optimizer_state

        env_steps_per_training_step = (
            self.batch_size * self.num_minibatches * self.unroll_length * config.action_repeat
        )
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_training_steps_per_epoch = int(np.ceil(
            config.total_env_steps / (num_evals_after_init * env_steps_per_training_step)
        ))

        # --- Rollout + buffer insert + contrastive sample + PPO updates ---
        def rollout_and_insert(training_state, buffer_state, env_state, key):
            """Generate one training-step worth of rollout, insert into buffer,
            and return PPO data + a fresh normalizer."""
            policy = make_policy((training_state.normalizer_params, training_state.params))
            num_unrolls = self.batch_size * self.num_minibatches // config.num_envs

            def f(carry, unused):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, rollout = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    self.unroll_length,
                    extra_fields=("truncation", "traj_id"),
                )
                return (next_state, next_key), rollout

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=num_unrolls
            )
            # data shape: (num_unrolls, unroll_length, num_envs, ...)

            # Reshape for PPO: merge (num_unrolls, num_envs) → batch_size * num_minibatches,
            # keep unroll_length as time dim per sample.
            ppo_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            ppo_data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), ppo_data)

            # Reshape for buffer: flatten (num_unrolls, unroll_length) → time dim,
            # keep num_envs as the per-env stream.
            buffer_obs = jnp.reshape(
                data.observation, (-1,) + data.observation.shape[2:]
            )
            buffer_action = jnp.reshape(data.action, (-1,) + data.action.shape[2:])
            buffer_traj_id = jnp.reshape(
                data.extras["state_extras"]["traj_id"],
                (-1,) + data.extras["state_extras"]["traj_id"].shape[2:],
            )
            buffer_entries = BufferTransition(
                observation=buffer_obs,
                action=buffer_action,
                traj_id=buffer_traj_id,
            )
            buffer_state = replay_buffer.insert(buffer_state, buffer_entries)

            normalizer_params = running_statistics.update(
                training_state.normalizer_params, ppo_data.observation
            )
            return ppo_data, buffer_state, env_state, normalizer_params

        def sample_contrastive_pool(buffer_state, key):
            """Sample a buffer of trajectories and flatten into (s, a, g) triples."""
            buffer_state, traj = replay_buffer.sample(buffer_state)
            # traj shape: (num_envs, episode_length, ...)
            n = traj.observation.shape[0]
            keys = jax.random.split(key, n)
            states, actions, goals = jax.vmap(
                sample_contrastive_pairs,
                in_axes=(None, None, None, 0, 0, 0, 0),
            )(
                self.discounting,
                state_size,
                goal_indices_static,
                traj.observation,
                traj.action,
                traj.traj_id,
                keys,
            )
            # (n, T-1, d) → flat (n * (T-1), d)
            states = states.reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            goals = goals.reshape(-1, goals.shape[-1])
            return buffer_state, (states, actions, goals)

        def training_step(carry, unused):
            training_state, buffer_state, env_state, key = carry
            key, rollout_key, crl_key, crl_idx_key, update_key, crl_actor_key, next_key = jax.random.split(key, 7)

            ppo_data, buffer_state, env_state, normalizer_params = rollout_and_insert(
                training_state, buffer_state, env_state, rollout_key
            )
            buffer_state, crl_pool = sample_contrastive_pool(buffer_state, crl_key)

            # Pre-slice num_crl_updates_per_step minibatches from the pool.
            # Sampling with replacement keeps shapes static regardless of pool size.
            states_pool, actions_pool, goals_pool = crl_pool
            pool_size = states_pool.shape[0]
            crl_idx = jax.random.randint(
                crl_idx_key,
                shape=(self.num_crl_updates_per_step, self.contrastive_batch_size),
                minval=0,
                maxval=pool_size,
            )
            crl_states_mb = states_pool[crl_idx]
            crl_actions_mb = actions_pool[crl_idx]
            crl_goals_mb = goals_pool[crl_idx]
            crl_actor_keys = jax.random.split(crl_actor_key, self.num_crl_updates_per_step)

            def update_over_minibatches(carry, unused):
                params, opt_state, uk = carry
                uk, perm_key, loss_key = jax.random.split(uk, 3)

                def shuffle(x):
                    x = jax.random.permutation(perm_key, x)
                    return jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])

                shuffled = jax.tree_util.tree_map(shuffle, ppo_data)

                def minibatch_step(carry, minibatch):
                    params, opt_state, mk = carry
                    mk, lk = jax.random.split(mk)
                    (_, metrics), grads = ppo_loss_and_grad(
                        params, normalizer_params, minibatch, lk
                    )
                    params, opt_state = apply_update(params, opt_state, grads)
                    return (params, opt_state, mk), metrics

                (params, opt_state, _), mb_metrics = jax.lax.scan(
                    minibatch_step,
                    (params, opt_state, loss_key),
                    shuffled,
                    length=self.num_minibatches,
                )
                return (params, opt_state, uk), mb_metrics

            (new_params, new_opt_state, _), ppo_metrics = jax.lax.scan(
                update_over_minibatches,
                (training_state.params, training_state.optimizer_state, update_key),
                (),
                length=self.num_updates_per_batch,
            )
            ppo_metrics = jax.tree_util.tree_map(jnp.mean, ppo_metrics)

            # num_crl_updates_per_step CRL gradient steps, each on a fresh
            # sub-batch drawn from the pool. Each step updates encoders (critic
            # term) AND policy_head (actor term) on replay-buffer data.
            def crl_step(carry, inputs):
                params, opt_state = carry
                states_i, actions_i, goals_i, ak = inputs
                (_, step_metrics), grads = crl_loss_and_grad(
                    params, normalizer_params, (states_i, actions_i, goals_i), ak
                )
                params, opt_state = apply_update(params, opt_state, grads)
                return (params, opt_state), step_metrics

            (new_params, new_opt_state), crl_mb_metrics = jax.lax.scan(
                crl_step,
                (new_params, new_opt_state),
                (crl_states_mb, crl_actions_mb, crl_goals_mb, crl_actor_keys),
                length=self.num_crl_updates_per_step,
            )
            crl_metrics = jax.tree_util.tree_map(jnp.mean, crl_mb_metrics)

            metrics = {**ppo_metrics, **crl_metrics}

            new_training_state = PPOCRLTrainingState(
                optimizer_state=new_opt_state,
                params=new_params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_training_step,
            )
            return (new_training_state, buffer_state, env_state, next_key), metrics

        def training_epoch(training_state, buffer_state, env_state, key):
            (training_state, buffer_state, env_state, _), metrics = jax.lax.scan(
                training_step,
                (training_state, buffer_state, env_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, buffer_state, env_state, metrics

        training_epoch = jax.jit(training_epoch)

        # --- Evaluator ---
        if eval_env is None:
            eval_env = train_env
        eval_env_wrapped = TrajectoryIdWrapper(eval_env)
        eval_env_wrapped = wrap_for_training(
            eval_env_wrapped,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        evaluator = Evaluator(
            eval_env_wrapped,
            functools.partial(make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # --- Prefill buffer so the first contrastive sample is well-formed ---
        num_prefill_training_steps = max(
            1, int(np.ceil(self.min_replay_size / (self.unroll_length * self.batch_size * self.num_minibatches // config.num_envs * config.num_envs)))
        )
        logging.info("ppo_crl prefill training-step equivalents: %d", num_prefill_training_steps)

        @jax.jit
        def prefill_step(training_state, buffer_state, env_state, key):
            key, rollout_key, next_key = jax.random.split(key, 3)
            _, buffer_state, env_state, _ = rollout_and_insert(
                training_state, buffer_state, env_state, rollout_key
            )
            return training_state, buffer_state, env_state, next_key

        prefill_key, local_key = jax.random.split(local_key)
        for _ in range(num_prefill_training_steps):
            training_state, buffer_state, env_state, prefill_key = prefill_step(
                training_state, buffer_state, env_state, prefill_key
            )

        # --- Initial evaluation ---
        metrics = {}
        if config.num_evals > 1:
            metrics = evaluator.run_evaluation(
                (training_state.normalizer_params, training_state.params),
                training_metrics={},
            )
            progress_fn(
                0,
                metrics,
                make_policy,
                (training_state.normalizer_params, training_state.params),
                unwrapped_env,
                do_render=False,
            )

        # --- Main training loop ---
        training_walltime = 0.0
        current_step = 0
        for eval_epoch in range(num_evals_after_init):
            logging.info("ppo_crl epoch %d t=%.1f", eval_epoch, time.time() - xt)
            t = time.time()
            epoch_key, local_key = jax.random.split(local_key)
            training_state, buffer_state, env_state, training_metrics = training_epoch(
                training_state, buffer_state, env_state, epoch_key
            )
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
            epoch_time = time.time() - t
            training_walltime += epoch_time

            current_step = int(training_state.env_steps)
            sps = (num_training_steps_per_epoch * env_steps_per_training_step) / epoch_time
            training_metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{k}": v for k, v in training_metrics.items()},
            }

            metrics = evaluator.run_evaluation(
                (training_state.normalizer_params, training_state.params),
                training_metrics,
            )
            do_render = eval_epoch % config.visualization_interval == 0
            progress_fn(
                current_step,
                metrics,
                make_policy,
                (training_state.normalizer_params, training_state.params),
                unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        assert total_steps >= config.total_env_steps, (
            f"total_steps ({total_steps}) < config.total_env_steps ({config.total_env_steps})"
        )
        logging.info("ppo_crl total steps: %d", total_steps)

        params = (training_state.normalizer_params, training_state.params)
        return make_policy, params, metrics
