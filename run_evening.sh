#!/bin/bash
# Evening runs: one ema baseline then goal with asymmetric clamp [0, 2].
set -u
mkdir -p /tmp/afk_runs
cd /home/saga/reflearn/crl/JaxGCRL

EMA_TAU=0.005
STEPS=15000000

echo "[$(date)] Starting crl_ema_baseline_evening_s4" >> /tmp/afk_runs/queue.log
XLA_PYTHON_CLIENT_ALLOCATOR=platform python run.py crl_ema \
    --env humanoid --total-env-steps $STEPS --ema-tau $EMA_TAU \
    --seed 4 \
    --wandb-group "crl_ema_baseline_afk" \
    --exp-name "crl_ema_baseline_evening_s4" \
    > /tmp/afk_runs/crl_ema_baseline_evening_s4.log 2>&1
echo "[$(date)] Finished crl_ema_baseline_evening_s4 (exit=$?)" >> /tmp/afk_runs/queue.log

echo "[$(date)] Starting crl_ema_goal_v8_clamp0toinf_s0" >> /tmp/afk_runs/queue.log
XLA_PYTHON_CLIENT_ALLOCATOR=platform python run.py crl_ema_goal \
    --env humanoid --total-env-steps $STEPS --ema-tau $EMA_TAU \
    --goal-critic-warmup-steps 5000000 --goal-critic-anneal-end-steps 8000000 \
    --goal-critic-coeff 1.0 \
    --goal-logit-clamp 1e9 --goal-logit-clamp-min 0.0 \
    --seed 0 \
    --wandb-group "crl_ema_goal_v8_afk" \
    --exp-name "crl_ema_goal_v8_clamp0toinf_s0" \
    > /tmp/afk_runs/crl_ema_goal_v8_clamp0toinf_s0.log 2>&1
echo "[$(date)] Finished crl_ema_goal_v8_clamp0toinf_s0 (exit=$?)" >> /tmp/afk_runs/queue.log

echo "[$(date)] EVENING RUNS COMPLETE" >> /tmp/afk_runs/queue.log
