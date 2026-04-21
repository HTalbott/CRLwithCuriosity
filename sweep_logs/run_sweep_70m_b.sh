#!/usr/bin/env bash
# Batch 1: crl_ema_goal (reg=1e-7, thresh=50, warmup=5M, anneal-end=15M) seeds 1-3
#          + crl_ema (defaults) seeds 0-3. Interleaved so seed 0 ema result lands first.
# On failure (non-zero exit OR log missing eval lines): wait RETRY_WAIT, retry.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log. Status: sweep_logs/STATUS_70m_b.
# On successful completion, exec run_sweep_70m_b2.sh (noema ablation).

set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_70m_b"
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

echo "== 70m_b sweep started $(date -Is) ==" >> "$STATUS"

# args: <agent> <seed> <exp_name>
run_one() {
  local agent=$1
  local seed=$2
  local exp=$3
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

# Interleaved: ema s0 lands first, then matched (goal,ema) pairs for s1-3.
run_one crl_ema      0 "crl_ema_ab70_s0"
run_one crl_ema_goal 1 "crl_ema_goal_anneal10m_gated_reg1e-7_s1"
run_one crl_ema      1 "crl_ema_ab70_s1"
run_one crl_ema_goal 2 "crl_ema_goal_anneal10m_gated_reg1e-7_s2"
run_one crl_ema      2 "crl_ema_ab70_s2"
run_one crl_ema_goal 3 "crl_ema_goal_anneal10m_gated_reg1e-7_s3"
run_one crl_ema      3 "crl_ema_ab70_s3"

echo "== 70m_b sweep finished $(date -Is) ==" >> "$STATUS"

# Chain batch 2 (noema ablation) if batch 1 finished cleanly.
if [[ -x "$LOGDIR/run_sweep_70m_b2.sh" ]]; then
  echo "[$(date -Is)] CHAIN launching run_sweep_70m_b2.sh" >> "$STATUS"
  exec "$LOGDIR/run_sweep_70m_b2.sh"
fi
