#!/usr/bin/env bash
# Sequential ant_ball sweep launcher (seed 0 for 4 models, then variance seeds)
# On crash (non-zero exit): wait 120s, retry up to MAX_ATTEMPTS-1 times.
# Logs per-attempt to sweep_logs/<exp>.attempt<N>.log; status appended to STATUS.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS"
GROUP=ant_ball_30m
STEPS=32000000   # bumped from 30M: 30M misaligns chunk size (200*256*62=3.17M), fails assertion in vanilla crl / crl_teammate. 32M = 10 chunks exactly.
ENV=ant_ball
MAX_ATTEMPTS=4   # initial + 3 retries
RETRY_WAIT=120   # seconds

echo "== sweep started $(date -Is) ==" >> "$STATUS"

# args: <agent> <seed>
run_one() {
  local agent=$1
  local seed=$2
  local exp="${agent}_ab_s${seed}"
  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    local log="$LOGDIR/${exp}.attempt${attempt}.log"
    echo "[$(date -Is)] START $exp attempt=$attempt" >> "$STATUS"
    jaxgcrl "$agent" \
      --env "$ENV" \
      --total-env-steps "$STEPS" \
      --seed "$seed" \
      --exp-name "$exp" \
      --wandb-group "$GROUP" \
      > "$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "[$(date -Is)] OK    $exp attempt=$attempt" >> "$STATUS"
      return 0
    fi
    echo "[$(date -Is)] FAIL  $exp attempt=$attempt rc=$rc (see $log)" >> "$STATUS"
    if (( attempt < MAX_ATTEMPTS )); then
      echo "[$(date -Is)] RETRY $exp waiting ${RETRY_WAIT}s before attempt $((attempt+1))" >> "$STATUS"
      sleep "$RETRY_WAIT"
    fi
    attempt=$(( attempt + 1 ))
  done
  echo "[$(date -Is)] GIVE_UP $exp after $MAX_ATTEMPTS attempts" >> "$STATUS"
  return 1
}

# Phase 1: seed 0 across 4 models
run_one crl                0
run_one crl_ema_goal       0
run_one crl_ema_goal_temp  0
run_one crl_teammate       0

# Phase 2: variance seeds for crl_ema_goal
run_one crl_ema_goal       1
run_one crl_ema_goal       2

# Phase 3 (if time): variance seeds for crl_ema_goal_temp
run_one crl_ema_goal_temp  1
run_one crl_ema_goal_temp  2

# Phase 4 (if time): variance seeds for vanilla crl
run_one crl                1
run_one crl                2

echo "== sweep finished $(date -Is) ==" >> "$STATUS"
