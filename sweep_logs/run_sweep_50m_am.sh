#!/usr/bin/env bash
# ant_u_maze 50M sweep — phase 1: vanilla crl + crl_ema, seeds 0-3.
# Seed-major: both variants for seed 0 before moving to seed 1.
# Resumable: re-running skips any exp already marked "OK" in STATUS_50m_am.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log. Status: sweep_logs/STATUS_50m_am.
# Phase 2 (crl_ema_goal) is a separate script launched after phase 1 finishes.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_50m_am"
GROUP=ant_u_maze_50m
STEPS=51046400  # 256_000 prefill + 16 chunks * 3,174,400 — required for vanilla-crl end-of-train assertion.
ENV=ant_u_maze
MAX_ATTEMPTS=4
RETRY_WAIT=120

echo "== 50m_am phase 1 sweep started $(date -Is) ==" >> "$STATUS"

# args: <agent> <seed> <exp_name>
run_one() {
  local agent=$1
  local seed=$2
  local exp=$3

  if grep -q "^\[.*\] OK    $exp " "$STATUS" 2>/dev/null; then
    echo "[$(date -Is)] SKIP  $exp (already OK)" >> "$STATUS"
    return 0
  fi

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

for seed in 0 1 2 3; do
  run_one crl      "$seed" "crl_am50_s${seed}"
  run_one crl_ema  "$seed" "crl_ema_am50_s${seed}"
done

echo "== 50m_am phase 1 sweep finished $(date -Is) ==" >> "$STATUS"
