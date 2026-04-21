#!/usr/bin/env bash
# ant_ball 70M-step sweep: 4x crl_ema_goal (perf-gated) + 4x vanilla crl.
# Interleaved by seed so we get apples-to-apples pairs early.
# On crash (non-zero exit): wait RETRY_WAIT, retry up to MAX_ATTEMPTS-1 times.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log. Status: sweep_logs/STATUS_70m.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_70m"
GROUP=ant_ball_70m
# Formula: STEPS = 256_000 (prefill) + N * 3,174,400 (chunk = num_evals*num_envs*unroll = 200*256*62).
# Vanilla crl asserts total_steps >= total_env_steps, and num_training_steps_per_epoch = (total_env_steps - prefill) // chunk.
# For N=22: 256_000 + 22*3_174_400 = 70_092_800 (closest aligned value to 70M).
STEPS=70092800
ENV=ant_ball
MAX_ATTEMPTS=4   # initial + 3 retries
RETRY_WAIT=120   # seconds

# crl_ema_goal perf-gating flags (per user request)
WARMUP_METRIC="eval/episode_success"
WARMUP_THRESHOLD=50.0

echo "== 70m sweep started $(date -Is) ==" >> "$STATUS"

# args: <agent> <seed>
run_one() {
  local agent=$1
  local seed=$2
  local exp="${agent}_ab70_s${seed}"
  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    local log="$LOGDIR/${exp}.attempt${attempt}.log"
    echo "[$(date -Is)] START $exp attempt=$attempt" >> "$STATUS"

    if [[ "$agent" == "crl_ema_goal" ]]; then
      jaxgcrl "$agent" \
        --env "$ENV" \
        --total-env-steps "$STEPS" \
        --seed "$seed" \
        --exp-name "$exp" \
        --wandb-group "$GROUP" \
        --goal-warmup-metric "$WARMUP_METRIC" \
        --goal-warmup-threshold "$WARMUP_THRESHOLD" \
        --goal-logit-reg 0.0 \
        > "$log" 2>&1
    else
      jaxgcrl "$agent" \
        --env "$ENV" \
        --total-env-steps "$STEPS" \
        --seed "$seed" \
        --exp-name "$exp" \
        --wandb-group "$GROUP" \
        > "$log" 2>&1
    fi

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

# Interleaved seed pairs — seed 0 first so we get both agents' seed-0 results early.
run_one crl_ema_goal 0
run_one crl          0
run_one crl_ema_goal 1
run_one crl          1
run_one crl_ema_goal 2
run_one crl          2
run_one crl_ema_goal 3
run_one crl          3

echo "== 70m sweep finished $(date -Is) ==" >> "$STATUS"
