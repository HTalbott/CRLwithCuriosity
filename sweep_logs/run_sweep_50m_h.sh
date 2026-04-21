#!/usr/bin/env bash
# humanoid 50M sweep: vanilla crl, crl_ema, crl_ema_goal (reg=1e-7, thresh=3).
# Scheduled BY SEED — all three variants for seed 0 before moving to seed 1.
# Resumable: re-running skips any exp already marked "OK" in STATUS_50m_h.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log. Status: sweep_logs/STATUS_50m_h.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_50m_h"
GROUP=humanoid_50m_anneal10m
STEPS=51046400  # 256_000 prefill + 16 chunks * 3,174,400 — closest 50M-aligned value that passes vanilla-crl end-of-train assertion (N=15 under-shoots).
ENV=humanoid
MAX_ATTEMPTS=4
RETRY_WAIT=120

WARMUP_METRIC="eval/episode_success"
WARMUP_THRESHOLD=3.0
WARMUP_STEPS=5000000
ANNEAL_END=15000000
LOGIT_REG=0.0000001

echo "== 50m_h sweep started $(date -Is) ==" >> "$STATUS"

# args: <agent> <seed> <exp_name>
run_one() {
  local agent=$1
  local seed=$2
  local exp=$3

  # Resume: skip if already succeeded.
  if grep -q "^\[.*\] OK    $exp " "$STATUS" 2>/dev/null; then
    echo "[$(date -Is)] SKIP  $exp (already OK)" >> "$STATUS"
    return 0
  fi

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
        --goal-critic-warmup-steps "$WARMUP_STEPS" \
        --goal-critic-anneal-end-steps "$ANNEAL_END" \
        --goal-logit-reg "$LOGIT_REG" \
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
    local eval_lines
    eval_lines=$(grep -c "eval/episode_reward:" "$log" 2>/dev/null || echo 0)
    if [[ $rc -eq 0 && $eval_lines -gt 50 ]]; then
      echo "[$(date -Is)] OK    $exp attempt=$attempt evals=$eval_lines" >> "$STATUS"
      return 0
    fi
    echo "[$(date -Is)] FAIL  $exp attempt=$attempt rc=$rc evals=$eval_lines (see $log)" >> "$STATUS"
    if (( attempt < MAX_ATTEMPTS )); then
      echo "[$(date -Is)] RETRY $exp waiting ${RETRY_WAIT}s before attempt $((attempt+1))" >> "$STATUS"
      sleep "$RETRY_WAIT"
    fi
    attempt=$(( attempt + 1 ))
  done
  echo "[$(date -Is)] GIVE_UP $exp after $MAX_ATTEMPTS attempts" >> "$STATUS"
  return 1
}

# Seed-major ordering: full (crl, ema, goal) triple per seed.
for seed in 0 1 2 3; do
  run_one crl          "$seed" "crl_hu50_s${seed}"
  run_one crl_ema      "$seed" "crl_ema_hu50_s${seed}"
  run_one crl_ema_goal "$seed" "crl_ema_goal_hu50_reg1e-7_s${seed}"
done

echo "== 50m_h sweep finished $(date -Is) ==" >> "$STATUS"
