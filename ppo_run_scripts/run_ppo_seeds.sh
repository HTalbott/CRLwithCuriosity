#!/bin/bash

# PPO Reacher Benchmark Automation Script
# Runs PPO with seeds 1 through 9 sequentially

# Base command from benchmark configuration
BASE_CMD="python run.py ppo --env reacher \
  --learning_rate 6e-4 \
  --unroll_length 62 \
  --num_updates_per_batch 5 \
  --batch_size 256 \
  --num_minibatches 16 \
  --discounting 0.97 \
  --gae_lambda 0.95 \
  --clipping_epsilon 0.2 \
  --entropy_cost 1e-4 \
  --total_env_steps 50000000 \
  --num_envs 4096 \
  --num_evals 198 \
  --log_wandb \
  --wandb_project_name \"jaxgcrl_benchmark\" \
  --episode_length 1000 \
  --action_repeat 1"

echo "Starting PPO benchmark runs with seeds 5-9"
echo "=========================================="

for SEED in {5..9}; do
    echo ""
    echo "Running PPO benchmark with seed: $SEED"
    echo "----------------------------------------"
    
    # Construct full command with seed
    FULL_CMD="$BASE_CMD --exp_name \"ppo_reacher_benchmark_seed_$SEED\" --seed $SEED"
    
    # # Execute the command
    echo "Command: $FULL_CMD"
    eval $FULL_CMD
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Seed $SEED completed successfully"
    else
        echo "✗ Seed $SEED failed with exit code $?"
        echo "Continuing with next seed..."
    fi
    
    # Small delay between runs (optional)
    sleep 2
done

echo ""
echo "=========================================="
echo "All PPO benchmark runs completed!"
echo "Seeds 1-9 have been executed."