#!/usr/bin/env bash
# Wait for current training PID to exit, then launch the reg=1e-4 variant.
set -u
cd /home/saga/reflearn/crl/JaxGCRL

CURRENT_PID=113126
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
NEXT_EXP=crl_ema_goal_anneal10m_gated_reg1e-4_thresh20_s0

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date -Is)] PID $CURRENT_PID exited; launching $NEXT_EXP" >> "$LOGDIR/queue_reg1e-4.log"

jaxgcrl crl_ema_goal \
  --env ant_ball \
  --total-env-steps 70092800 \
  --seed 0 \
  --exp-name "$NEXT_EXP" \
  --wandb-group ant_ball_70m_anneal10m \
  --goal-warmup-metric "eval/episode_success" \
  --goal-warmup-threshold 20.0 \
  --goal-critic-warmup-steps 5000000 \
  --goal-critic-anneal-end-steps 15000000 \
  --goal-logit-reg 0.0001 \
  > "$LOGDIR/${NEXT_EXP}.log" 2>&1

echo "[$(date -Is)] $NEXT_EXP exited rc=$?" >> "$LOGDIR/queue_reg1e-4.log"
