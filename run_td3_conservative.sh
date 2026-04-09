#!/bin/bash

# TD3 Benchmark Automation Script - Conservative Memory Version
# Runs TD3 without HER with reduced memory footprint
# Uses smaller max_replay_size and runs seeds sequentially

# Base TD3 command with conservative memory settings
BASE_CMD="python run.py td3 \
  --learning_rate 3e-4 \
  --unroll_length 62 \
  --train_step_multiplier 1 \
  --batch_size 256 \
  --discounting 0.99 \
  --total_env_steps 50000000 \
  --num_evals 788 \
  --log_wandb \
  --wandb_project_name \"jaxgcrl_benchmark_TD3_conservative\" \
  --episode_length 1000 \
  --action_repeat 1 \
  --max_replay_size 5000 \
  --min_replay_size 1000 \
  --tau 0.005 \
  --policy_delay 2 \
  --noise_clip 0.5 \
  --smoothing_noise 0.2 \
  --exploration_noise 0.4"

# Environments to iterate over (start with just reacher to test)
ENVIRONMENTS=("reacher")  # Start with one environment to test

# Environment-specific parameters (num_envs from benchmark.md)
declare -A ENV_PARAMS
ENV_PARAMS["reacher"]="--num_envs 1024"
ENV_PARAMS["humanoid"]="--num_envs 512"  # Humanoid uses 512 envs per benchmark
ENV_PARAMS["ant"]="--num_envs 1024"
ENV_PARAMS["ant_u_maze"]="--num_envs 1024"
ENV_PARAMS["ant_ball"]="--num_envs 1024"

echo "Starting TD3 benchmark runs - Conservative Memory Version"
echo "Using max_replay_size=5000 to reduce memory usage"
echo "=============================================================="

TOTAL_RUNS=$(( ${#ENVIRONMENTS[@]} * 10 ))
CURRENT_RUN=1

for ENV in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "Environment: $ENV"
    echo "================================"
    
    # Get environment-specific parameters
    ENV_SPECIFIC="${ENV_PARAMS[$ENV]}"
    
    # Run only seeds 0-2 initially to test
    for SEED in {0..2}; do
        echo ""
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: $ENV with seed $SEED"
        echo "----------------------------------------"
        
        # Construct full command with environment and seed
        FULL_CMD="$BASE_CMD --env $ENV $ENV_SPECIFIC --exp_name \"td3_${ENV}_benchmark_seed_$SEED_conservative\" --seed $SEED"
        
        # Execute the command
        echo "Command: $FULL_CMD"
        eval $FULL_CMD
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "✓ $ENV seed $SEED completed successfully"
        else
            echo "✗ $ENV seed $SEED failed with exit code $?"
            echo "Stopping due to failure..."
            exit 1
        fi
        
        # Wait longer between runs to ensure memory is fully freed
        echo "Waiting 30 seconds before next run..."
        sleep 30
        
        CURRENT_RUN=$((CURRENT_RUN + 1))
    done
    
    echo ""
    echo "Completed test seeds for $ENV"
    echo "================================"
done

echo ""
echo "=============================================================="
echo "TD3 conservative memory test completed!"
echo "If successful, you can run the full benchmark with more seeds and environments."