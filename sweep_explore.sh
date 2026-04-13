#!/usr/bin/env bash
# Exploration coefficient sweep for crl_explore on Humanoid, 10M steps each.
# Anneal fraction fixed at 0.75 (bonus hits zero at 7.5M steps).
# Coefficients: 0.0 (baseline), 0.1, 0.5, 1.0, 2.0 (insane)

set -euo pipefail

export XLA_PYTHON_CLIENT_ALLOCATOR=platform

GROUP="crl_explore_coeff_sweep_v1"
ENV="humanoid"
STEPS=10000000
SEED=0
ANNEAL_FRAC=0.75

COEFFS=(0.3 0.5 1.0 2.0)

echo "=== CRL_EXPLORE coefficient sweep ==="
echo "Group:        $GROUP"
echo "Env:          $ENV"
echo "Steps:        $STEPS"
echo "Anneal frac:  $ANNEAL_FRAC"
echo "Coefficients: ${COEFFS[*]}"
echo ""

for COEFF in "${COEFFS[@]}"; do
    EXP_NAME="crl_explore_c${COEFF}_s${SEED}"
    echo "--- Starting coeff=$COEFF  (exp: $EXP_NAME) ---"

    python run.py \
        crl_explore \
        --env "$ENV" \
        --total-env-steps "$STEPS" \
        --seed "$SEED" \
        --wandb-group "$GROUP" \
        --exp-name "$EXP_NAME" \
        --exploration-coeff "$COEFF" \
        --exploration-anneal-frac "$ANNEAL_FRAC"

    echo "--- Finished coeff=$COEFF ---"
    echo "GPU memory after run:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    sleep 5
    echo "GPU memory after 5s cooldown:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    echo ""
done

echo "=== Sweep complete ==="
