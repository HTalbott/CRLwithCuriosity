#!/usr/bin/env bash
# reacher ~5M sweep — three variants x four seeds = 12 runs.
#   phase 1: vanilla crl, seeds 0-3
#   phase 2: crl_ema,     seeds 0-3
#   phase 3: crl_ema_goal (logit_reg=1e-7), seeds 0-3
# Gating uses eval/episode_success_any threshold 0.4 (fraction of eval
# episodes with >=1 successful step). Step gate scaled to new budget:
# 1M warmup, 3M anneal-end (~15% / ~45% of total, matching 20M's 10%/30%).
# Aligned total_env_steps = 256_000 + 2 * 3_174_400 = 6_604_800 (~6.6M,
# closest aligned value to user's 5M target). Reacher w/ spring backend
# converges to ~90% success by ~1M steps in the sanity run, so 6.6M is
# well past plateau.
# Uses --backend spring instead of reacher's default "generalized" backend:
# generalized is ~13x slower (6k vs 80k SPS) and made 20M runs take ~5h each.
# Spring is brax's simplified solver used by ant/humanoid envs.
# Resumable: re-running skips any exp already marked "OK" in STATUS_reacher_5m.
# Per-attempt logs: sweep_logs/<exp>.attempt<N>.log.
set -u
cd /home/saga/reflearn/crl/JaxGCRL
LOGDIR=/home/saga/reflearn/crl/JaxGCRL/sweep_logs
STATUS="$LOGDIR/STATUS_reacher_5m"
GROUP=reacher_5m
STEPS=6604800
ENV=reacher
BACKEND=spring
MAX_ATTEMPTS=4
RETRY_WAIT=120

WARMUP_METRIC="eval/episode_success_any"
WARMUP_THRESHOLD=0.4
WARMUP_STEPS=1000000
ANNEAL_END=3000000
LOGIT_REG=0.0000001

echo "== reacher_5m sweep started $(date -Is) (threshold=$WARMUP_THRESHOLD metric=$WARMUP_METRIC) ==" >> "$STATUS"

run_one() {
  local agent=$1
  local seed=$2
  local exp=$3
  shift 3
  local extra_args=("$@")

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
      --backend "$BACKEND" \
      --total-env-steps "$STEPS" \
      --seed "$seed" \
      --exp-name "$exp" \
      --wandb-group "$GROUP" \
      "${extra_args[@]}" \
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

# Phase 1: vanilla CRL
for seed in 0 1 2 3; do
  run_one crl "$seed" "crl_rc5_s${seed}"
done

# Phase 2: CRL + EMA
for seed in 0 1 2 3; do
  run_one crl_ema "$seed" "crl_ema_rc5_s${seed}"
done

# Phase 3: CRL + EMA + goal critic (logit_reg=1e-7, metric-gated)
for seed in 0 1 2 3; do
  run_one crl_ema_goal "$seed" "crl_ema_goal_rc5_reg1e-7_s${seed}" \
    --goal-warmup-metric "$WARMUP_METRIC" \
    --goal-warmup-threshold "$WARMUP_THRESHOLD" \
    --goal-critic-warmup-steps "$WARMUP_STEPS" \
    --goal-critic-anneal-end-steps "$ANNEAL_END" \
    --goal-logit-reg "$LOGIT_REG"
done

echo "== reacher_5m sweep finished $(date -Is) ==" >> "$STATUS"
