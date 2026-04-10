## Basic Command Structure

```bash
# General format
jaxgcrl ppo_crl --env <environment_name> --log_wandb

# For PPO+CRL with contrastive reward shaping specifically
jaxgcrl ppo_crl --env reacher --log_wandb
```

## Complete Example Command

```bash
jaxgcrl ppo_crl \
  --env reacher \
  --total_env_steps 1000000 \
  --num_envs 512 \
  --batch_size 256 \
  --unroll_length 10 \
  --contrastive_coeff 1.0 \
  --repr_dim 64 \
  --energy_fn_name "l2" \
  --contrastive_loss_fn "sym_infonce" \
  --log_wandb \
  --wandb_project_name "jaxgcrl-ppo-crl" \
  --exp_name "ppo_crl_reacher_test"
```

## Key Parameters for PPO+CRL

### Required for WandB:
- `--log_wandb`: Enables WandB logging (without this, WandB runs in disabled mode)
- `--wandb_project_name`: Project name in WandB (defaults to something if not specified)
- `--exp_name`: Experiment name for logging

### PPO+CRL Specific Hyperparameters:
- `--contrastive_coeff`: β coefficient for intrinsic reward (default: 1.0)
- `--repr_dim`: Representation dimension for CRL encoders (default: 64)
- `--energy_fn_name`: Energy function ("l2", "dot", "cosine", "norm") (default: "l2")
- `--contrastive_loss_fn`: Contrastive loss function ("sym_infonce", "fwd_infonce", etc.) (default: "sym_infonce")
- `--crl_learning_rate`: Learning rate for CRL encoders (default: 3e-4)

### Standard PPO Parameters (with recommended values):
- `--learning_rate`: 3e-4 (PPO learning rate)
- `--entropy_cost`: 1e-4
- `--discounting`: 0.99
- `--unroll_length`: 10
- `--batch_size`: 256 (adjust based on GPU memory)
- `--num_envs`: 512 (parallel environments)
- `--total_env_steps`: 10000000 (10M for proper training)

## Example Test Commands

### Quick Test (1M steps):
```bash
jaxgcrl ppo_crl --env reacher --total_env_steps 1000000 --num_envs 128 --batch_size 128 --log_wandb --exp_name "quick_test"
```

### Full Training (10M steps):
```bash
jaxgcrl ppo_crl \
  --env reacher \
  --total_env_steps 10000000 \
  --num_envs 512 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --contrastive_coeff 1.0 \
  --repr_dim 64 \
  --log_wandb \
  --wandb_project_name "ppo-crl-experiments" \
  --exp_name "reacher_beta1.0"
```

### Ablation Study (β=0 vs β=1.0):
```bash
# Vanilla PPO baseline (β=0)
jaxgcrl ppo_crl --env reacher --contrastive_coeff 0.0 --log_wandb --exp_name "ppo_baseline"

# PPO+CRL with moderate intrinsic reward
jaxgcrl ppo_crl --env reacher --contrastive_coeff 0.5 --log_wandb --exp_name "ppo_crl_beta0.5"

# PPO+CRL with strong intrinsic reward
jaxgcrl ppo_crl --env reacher --contrastive_coeff 1.0 --log_wandb --exp_name "ppo_crl_beta1.0"
```

## Environment Options

Available environments (from `repo_overview.md`):
- **Simple arm**: `reacher`, `pusher`, `pusher2`
- **Locomotion**: `ant`, `cheetah`, `humanoid`
- **Locomotion + task**: `ant_maze`, `ant_ball`, `ant_push`, `humanoid_maze`
- **Manipulation**: `arm_reach`, `arm_grasp`, `arm_push`, `arm_binpick`

## WandB Configuration

The WandB initialization in `run.py` uses:
```python
wandb.init(
    project=config.run.wandb_project_name,
    group=config.run.wandb_group,
    name=config.run.exp_name,
    config=info,  # All hyperparameters
    mode="online" if config.run.log_wandb else "disabled",
)
```

So you can also add `--wandb_group` for grouping related experiments.

## Important Notes:

1. **First run might fail** if the agent isn't properly registered in the CLI. Check that `ppo_crl` is in the tyro configuration.

2. **Monitor GPU memory** - adjust `--num_envs` and `--batch_size` based on your GPU.

3. **Check WandB login** - make sure you're logged into WandB (`wandb login`).

4. **View metrics** - The implementation tracks:
   - Standard PPO metrics (`training/ppo_loss`, `training/value_loss`, etc.)
   - CRL metrics (`crl/loss`, `crl/categorical_accuracy`)
   - Intrinsic reward metrics (`intrinsic_reward/mean`, `intrinsic_reward/std`)
   - Evaluation metrics (`eval/episode_success`, `eval/episode_reward`)