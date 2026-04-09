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
  --num_evals 1575 \
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
  --exploration_noise 0.4 \
  --num_envs 512 \
  --env humanoid \
  --exp_name \"td3_humanoid_benchmark_seed_0\" \
  --seed 0"

eval $BASE_CMD