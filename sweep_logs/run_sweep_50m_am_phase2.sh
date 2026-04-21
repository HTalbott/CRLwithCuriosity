#!/usr/bin/env bash
# ant_u_maze 50M sweep — phase 2: crl_ema_goal seeds 0-3 with logit_reg=1e-7.
# Threshold picked from phase 1: baselines reliably cross 20 on eval/episode_success
# by ~2-3M steps, so step-based WARMUP_STEPS=5M dominates timing (matches humanoid's
# just-above-noise gating philosophy).
# Resumable: re-running skips any exp already marked "OK" in STATUS_50m_am_p2.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log. Status: sweep_logs/STATUS_50m_am_p2.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_50m_am_p2"
GROUP=ant_u_maze_50m
STEPS=51046400
ENV=ant_u_maze
MAX_ATTEMPTS=4
RETRY_WAIT=120

WARMUP_METRIC="eval/episode_success"
WARMUP_THRESHOLD=20.0
WARMUP_STEPS=5000000
ANNEAL_END=15000000
LOGIT_REG=0.0000001

echo "== 50m_am phase 2 sweep started $(date -Is) (threshold=$WARMUP_THRESHOLD) ==" >> "$STATUS"

run_one() {
  local seed=$1
  local exp=$2

  if grep -q "^\[.*\] OK    $exp " "$STATUS" 2>/dev/null; then
    echo "[$(date -Is)] SKIP  $exp (already OK)" >> "$STATUS"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    local log="$LOGDIR/${exp}.attempt${attempt}.log"
    echo "[$(date -Is)] START $exp attempt=$attempt" >> "$STATUS"

    jaxgcrl crl_ema_goal \
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
  run_one "$seed" "crl_ema_goal_am50_reg1e-7_s${seed}"
done

echo "== 50m_am phase 2 sweep finished $(date -Is) ==" >> "$STATUS"
