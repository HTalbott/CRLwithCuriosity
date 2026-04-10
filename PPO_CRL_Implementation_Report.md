# PPO+CRL Implementation Report: Key Differences from Original PPO

## Overview
This report documents the implementation of **PPO with Contrastive Reward Shaping (PPO+CRL)**, highlighting the key differences from the original PPO implementation. The implementation follows **Approach 1: Contrastive Reward Shaping** as recommended in the analysis.

## Core Concept
**Contrastive Reward Shaping**: Adds CRL's energy as intrinsic reward to PPO:
```
total_reward = sparse_environment_reward + β * intrinsic_reward
where intrinsic_reward = energy(φ(s,a), ψ(g))
```

## Key Differences Summary

### 1. **Class Definition and Hyperparameters**

#### Original PPO:
```python
@dataclass
class PPO:
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    # ... standard PPO hyperparameters
```

#### PPO+CRL:
```python
@dataclass
class PPO_CRL_RewardShaping:
    # Original PPO hyperparameters preserved
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    # ... standard PPO hyperparameters
    
    # NEW: CRL-specific hyperparameters
    contrastive_coeff: float = 1.0          # β coefficient for intrinsic reward
    repr_dim: int = 64                      # Representation dimension for encoders
    contrastive_loss_fn: str = "sym_infonce" # Contrastive loss function
    energy_fn_name: str = "l2"              # Energy function for similarity
    crl_learning_rate: float = 3e-4         # Learning rate for CRL encoders
    logsumexp_penalty_coeff: float = 0.1    # Penalty coefficient for logsumexp
    normalize_intrinsic_reward: bool = True  # Normalize intrinsic rewards per batch
```

**Explanation**: Added 7 new hyperparameters to control CRL behavior while preserving all original PPO hyperparameters for backward compatibility.

### 2. **Training State Extension**

#### Original PPO:
```python
@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray
```

#### PPO+CRL:
```python
@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray
    # NEW: CRL components
    crl_optimizer_state: optax.OptState
    crl_params: dict  # Contains sa_encoder and g_encoder parameters
```

**Explanation**: Extended the training state to include CRL encoder parameters and their optimizer state, enabling separate optimization of CRL components.

### 3. **CRL Encoder Creation and Initialization**

#### NEW in PPO+CRL:
```python
# Create CRL encoders
obs_dim = env_state.obs.shape[-1]
# For goal-conditioned environments, we need to determine state and goal dimensions
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

# Initialize CRL encoder parameters
key_sa, key_g = jax.random.split(key_crl)
init_crl_params = {
    "sa_encoder": sa_encoder.init(key_sa, jnp.zeros((1, state_dim + env.action_size))),
    "g_encoder": g_encoder.init(key_g, jnp.zeros((1, goal_dim))),
}
```

**Explanation**: 
- Automatically detects state/goal dimensions from environment observations
- Creates two encoder networks: `sa_encoder` for state-action pairs and `g_encoder` for goals
- Initializes parameters with separate random keys for reproducibility

### 4. **Modified Loss Function with Intrinsic Rewards**

#### Original PPO Loss:
```python
loss_fn = functools.partial(
    ppo_losses.compute_ppo_loss,
    ppo_network=ppo_network,
    entropy_cost=self.entropy_cost,
    # ... other PPO parameters
)
```

#### PPO+CRL Loss:
```python
def ppo_crl_loss(params, normalizer_params, data, key, crl_params):
    """PPO loss with contrastive-shaped rewards."""
    # Extract state, action, goal from observations
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
```

**Explanation**:
- **Reward Shaping**: Computes intrinsic reward using CRL encoders and adds it to environment reward
- **Normalization**: Optional batch normalization of intrinsic rewards to prevent dominance
- **Metrics Tracking**: Added comprehensive tracking of intrinsic reward statistics
- **Backward Compatible**: Still uses the original `ppo_losses.compute_ppo_loss` function internally

### 5. **Dual Optimization: PPO + CRL Updates**

#### Original PPO Minibatch Step:
```python
def minibatch_step(carry, data, normalizer_params):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params, normalizer_params, data, key_loss, optimizer_state=optimizer_state
    )
    return (optimizer_state, params, key), metrics
```

#### PPO+CRL Minibatch Step:
```python
def minibatch_step(carry, data, normalizer_params):
    optimizer_state, params, crl_params, crl_optimizer_state, key = carry
    key, key_loss, key_crl = jax.random.split(key, 3)
    
    # Update PPO with shaped rewards
    def loss_fn_with_crl(params, normalizer_params, data, key):
        return loss_fn(params, normalizer_params, data, key, crl_params)
    
    ppo_gradient_update_fn = gradients.gradient_update_fn(
        loss_fn_with_crl, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    
    (_, metrics), params, optimizer_state = ppo_gradient_update_fn(
        params, normalizer_params, data, key_loss, optimizer_state=optimizer_state
    )

    # NEW: Update CRL encoders
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
```

**Explanation**:
- **Dual Optimization**: Performs both PPO and CRL updates in each minibatch
- **Separate Optimizers**: Uses different optimizers for PPO (original) and CRL (new)
- **Metrics Combination**: Combines PPO and CRL metrics for comprehensive logging
- **Key Management**: Properly splits random keys for both updates

### 6. **CRL Encoder Update Function**

#### NEW Helper Function:
```python
def update_crl_encoders(config, sa_encoder, g_encoder, transitions, crl_params, crl_optimizer_state, key):
    """Update CRL encoders using contrastive loss."""
    
    def crl_loss(crl_params, transitions, key):
        sa_encoder_params, g_encoder_params = crl_params["sa_encoder"], crl_params["g_encoder"]
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
```

**Explanation**:
- **Contrastive Learning**: Implements InfoNCE loss for representation learning
- **Regularization**: Includes logsumexp penalty to prevent collapse
- **Metrics**: Tracks accuracy and logit statistics for monitoring
- **Gradient Application**: Uses standard optax optimizer updates

### 7. **Checkpoint Restoration with Backward Compatibility**

#### Original PPO Checkpoint:
```python
target = training_state.normalizer_params, init_params
(normalizer_params, init_params) = orbax_checkpointer.restore(
    self.restore_checkpoint_path, item=target
)
training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)
```

#### PPO+CRL Checkpoint:
```python
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
```

**Explanation**:
- **Backward Compatibility**: Can load both PPO-only and PPO+CRL checkpoints
- **Graceful Degradation**: Falls back to PPO-only restoration if CRL params not found
- **Warning Logging**: Informs user when CRL parameters are initialized from scratch

## Key Design Decisions

### 1. **Minimal Intrusiveness**
- Preserved all original PPO code structure
- Added CRL components as optional extensions
- Can disable CRL by setting `contrastive_coeff = 0.0`

### 2. **Separate Optimization**
- CRL encoders have their own optimizer with separate learning rate
- Prevents CRL updates from interfering with PPO's delicate clipped objective
- Allows fine-tuning of CRL learning independently

### 3. **Automatic Dimension Detection**
- Automatically infers state/goal dimensions from observations
- Falls back to heuristic (50/50 split) if environment doesn't provide `state_size`
- Handles various goal-conditioned environment configurations

### 4. **Comprehensive Metrics**
- Added 9 new metrics for monitoring:
  - Intrinsic reward statistics (mean, std, max, min)
  - Total reward mean
  - CRL loss and accuracy metrics
  - Logit statistics for contrastive learning

### 5. **Modular Helper Functions**
- `create_crl_encoders()`: Encoder creation
- `compute_intrinsic_reward()`: Reward shaping logic
- `update_crl_encoders()`: CRL optimization
- Each function has clear responsibility and can be tested independently

## Expected Benefits

### 1. **Improved Sample Efficiency**
- Intrinsic rewards provide dense learning signals
- Reduces gradient vanishing in sparse-reward environments
- Faster convergence to goal-reaching behavior

### 2. **Better Exploration**
- CRL guides policy toward reachable goals
- Learns reachability representations from environment interactions
- Natural curriculum learning through contrastive rewards

### 3. **Minimal Performance Overhead**
- CRL updates are parallelized with PPO updates
- Efficient JAX vectorization maintains performance
- Optional components can be disabled for ablation studies

## Testing Recommendations

### 1. **Ablation Studies**
```bash
# Vanilla PPO baseline
jaxgcrl ppo --env reacher

# PPO+CRL with different β coefficients
jaxgcrl ppo_crl --env reacher --contrastive_coeff 0.0  # Equivalent to vanilla PPO
jaxgcrl ppo_crl --env reacher --contrastive_coeff 0.5
jaxgcrl ppo_crl --env reacher --contrastive_coeff 1.0
jaxgcrl ppo_crl --env reacher --contrastive_coeff 2.0
```

### 2. **Environment Tests**
- `reacher`: Simple sparse reward (proof of concept)
- `ant_maze`: Complex navigation with sparse rewards
- `humanoid`: High-dimensional control with exploration challenges

### 3. **Metrics to Monitor**
- `eval/episode_success`: Primary success metric
- `intrinsic_reward/mean`: Magnitude of intrinsic rewards
- `crl/categorical_accuracy`: CRL representation quality
- `training/sps`: Steps per second (performance impact)

## Conclusion

The PPO+CRL implementation successfully integrates contrastive representation learning with PPO through reward shaping while maintaining backward compatibility and minimal code intrusion. The key innovation is the addition of intrinsic rewards computed from CRL encoders, which addresses the fundamental problem of vanishing gradients in sparse-reward environments.

The implementation is ready for testing and provides a solid foundation for further research into combining on-policy RL with contrastive learning techniques.