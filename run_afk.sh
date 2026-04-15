#!/bin/bash
# Sequential runner: alternates crl_ema_goal (various clamps) with crl_ema baseline
# All runs to 15M steps on Humanoid.
# Writes per-run stdout under /tmp/afk_runs/

set -u
mkdir -p /tmp/afk_runs

cd /home/saga/reflearn/crl/JaxGCRL

EMA_TAU=0.005
STEPS=15000000
COMMON_GOAL_ARGS="--env humanoid --total-env-steps $STEPS --ema-tau $EMA_TAU \
  --goal-critic-warmup-steps 5000000 --goal-critic-anneal-end-steps 8000000 \
  --goal-critic-coeff 1.0"
COMMON_EMA_ARGS="--env humanoid --total-env-steps $STEPS --ema-tau $EMA_TAU"

run_goal() {
    local clamp=$1
    local seed=$2
    local tag="crl_ema_goal_v7_clamp${clamp}_s${seed}"
    echo "[$(date)] Starting $tag" >> /tmp/afk_runs/queue.log
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python run.py crl_ema_goal \
        $COMMON_GOAL_ARGS \
        --goal-logit-clamp $clamp \
        --seed $seed \
        --wandb-group "crl_ema_goal_v7_afk" \
        --exp-name "$tag" \
        > /tmp/afk_runs/${tag}.log 2>&1
    echo "[$(date)] Finished $tag (exit=$?)" >> /tmp/afk_runs/queue.log
}

run_ema() {
    local seed=$1
    local tag="crl_ema_baseline_afk_s${seed}"
    echo "[$(date)] Starting $tag" >> /tmp/afk_runs/queue.log
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python run.py crl_ema \
        $COMMON_EMA_ARGS \
        --seed $seed \
        --wandb-group "crl_ema_baseline_afk" \
        --exp-name "$tag" \
        > /tmp/afk_runs/${tag}.log 2>&1
    echo "[$(date)] Finished $tag (exit=$?)" >> /tmp/afk_runs/queue.log
}

# Alternating schedule - goal variants paired with ema baselines for comparison
run_goal 2.0 2       # goal clamp [-2, 2], seed 2
run_ema 0            # baseline seed 0
run_goal 4.0 2       # goal clamp [-4, 4], seed 2 (matches prior config, new seed)
run_ema 1            # baseline seed 1
run_goal 8.0 0       # goal clamp [-8, 8], seed 0
run_ema 2            # baseline seed 2
run_goal 1.0 0       # goal clamp [-1, 1], seed 0
run_ema 3            # baseline seed 3
run_goal 2.0 3       # goal clamp [-2, 2], seed 3 (additional seed)
run_goal 4.0 3       # goal clamp [-4, 4], seed 3
run_goal 8.0 1       # goal clamp [-8, 8], seed 1
# Retries with 90s cooldown before each goal run (goal runs segfault when
# started immediately after an ema run — likely GPU state not released).
retry_goal() {
    local clamp=$1
    local seed=$2
    local tag="crl_ema_goal_v7_clamp${clamp}_s${seed}_retry"
    echo "[$(date)] Cooldown 90s then starting $tag" >> /tmp/afk_runs/queue.log
    sleep 90
    XLA_PYTHON_CLIENT_ALLOCATOR=platform python run.py crl_ema_goal \
        $COMMON_GOAL_ARGS \
        --goal-logit-clamp $clamp \
        --seed $seed \
        --wandb-group "crl_ema_goal_v7_afk" \
        --exp-name "$tag" \
        > /tmp/afk_runs/${tag}.log 2>&1
    echo "[$(date)] Finished $tag (exit=$?)" >> /tmp/afk_runs/queue.log
}

retry_goal 4.0 2     # retry for clamp 4.0 seed 2 (segfaulted at 1.6M)
retry_goal 8.0 0     # retry for clamp 8.0 seed 0 (segfaulted at 0M)
retry_goal 1.0 0     # retry for clamp 1.0 seed 0 (segfaulted ~2M)

echo "[$(date)] ALL RUNS COMPLETE" >> /tmp/afk_runs/queue.log
