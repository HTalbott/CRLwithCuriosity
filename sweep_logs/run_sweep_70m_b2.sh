#!/usr/bin/env bash
# Batch 2: crl_ema_goal with EMA OFF (ema_tau=1.0) — isolates the Goal contribution.
# Same reg/threshold/warmup params as batch 1 reg=1e-7 config. Seeds 0-3.
# Auto-launched from run_sweep_70m_b.sh on successful batch-1 completion.

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_70m_b2"
GROUP=ant_ball_70m_anneal10m
STEPS=70092800
ENV=ant_ball
MAX_ATTEMPTS=4
RETRY_WAIT=120

WARMUP_METRIC="eval/episode_success"
WARMUP_THRESHOLD=50.0
WARMUP_STEPS=5000000
ANNEAL_END=15000000
LOGIT_REG=0.0000001
EMA_TAU=1.0  # target = online (no smoothing)

echo "== 70m_b2 sweep started $(date -Is) ==" >> "$STATUS"

run_one() {
  local seed=$1
  local exp="crl_ema_goal_anneal10m_gated_reg1e-7_noema_s${seed}"
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
      --ema-tau "$EMA_TAU" \
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

run_one 0
run_one 1
run_one 2
run_one 3

echo "== 70m_b2 sweep finished $(date -Is) ==" >> "$STATUS"
