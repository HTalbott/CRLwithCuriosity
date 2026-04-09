#!/bin/bash

# TD3 Benchmark Automation Script
# Runs TD3 without HER on multiple environments with seeds 0 through 9

# Base TD3 command from benchmark configuration
BASE_CMD="python run.py td3 \
  --learning_rate 3e-4 \
  --unroll_length 62 \
  --train_step_multiplier 1 \
  --batch_size 256 \
  --discounting 0.99 \
  --total_env_steps 50000000 \
  --num_evals 788 \
  --log_wandb \
  --wandb_project_name \"jaxgcrl_benchmark_TD3\" \
  --episode_length 1000 \
  --action_repeat 1 \
  --max_replay_size 10000 \
  --min_replay_size 1000 \
  --tau 0.005 \
  --policy_delay 2 \
  --noise_clip 0.5 \
  --smoothing_noise 0.2 \
  --exploration_noise 0.4"

# Environments to iterate over
ENVIRONMENTS=("reacher" "humanoid" "ant" "ant_u_maze" "ant_ball")

# Environment-specific parameters (num_envs from benchmark.md)
declare -A ENV_PARAMS
ENV_PARAMS["reacher"]="--num_envs 1024"
ENV_PARAMS["humanoid"]="--num_envs 512"  # Humanoid uses 512 envs per benchmark
ENV_PARAMS["ant"]="--num_envs 1024"
ENV_PARAMS["ant_u_maze"]="--num_envs 1024"
ENV_PARAMS["ant_ball"]="--num_envs 1024"

echo "Starting TD3 benchmark runs across environments with seeds 0-9"
echo "=============================================================="

TOTAL_RUNS=$(( ${#ENVIRONMENTS[@]} * 10 ))
CURRENT_RUN=1

for ENV in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "Environment: $ENV"
    echo "================================"
    
    # Get environment-specific parameters
    ENV_SPECIFIC="${ENV_PARAMS[$ENV]}"
    
    for SEED in {0..9}; do
        echo ""
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: $ENV with seed $SEED"
        echo "----------------------------------------"
        
        # Construct full command with environment and seed
        FULL_CMD="$BASE_CMD --env $ENV $ENV_SPECIFIC --exp_name \"td3_${ENV}_benchmark_seed_$SEED\" --seed $SEED"
        
        # Execute the command
        echo "Command: $FULL_CMD"
        eval $FULL_CMD
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "✓ $ENV seed $SEED completed successfully"
        else
            echo "✗ $ENV seed $SEED failed with exit code $?"
            echo "Continuing with next run..."
        fi
        
        # Small delay between runs (optional)
        sleep 2
        
        CURRENT_RUN=$((CURRENT_RUN + 1))
    done
    
    echo ""
    echo "Completed all seeds for $ENV"
    echo "================================"
    sleep 5  # Longer delay between environments
done

echo ""
echo "=============================================================="
echo "All TD3 benchmark runs completed!"
echo "Total: $TOTAL_RUNS runs across ${#ENVIRONMENTS[@]} environments with seeds 0-9"