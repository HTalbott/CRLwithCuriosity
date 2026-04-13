#!/usr/bin/env bash
# Sequential beta sweep for crl_plus (PER-style reweighting) on Humanoid, 10M steps each.
# Betas: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
# Runs serially to avoid GPU contention.

set -euo pipefail

# Use platform allocator so GPU memory is freed between runs.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

GROUP="crl_plus_per_beta_sweep_v1"
ENV="humanoid"
STEPS=10000000
SEED=0

BETAS=(1.0 1.5 2.0 2.5)  # 0.0, 0.5 already completed

echo "=== CRL_PLUS (PER-style) beta sweep ==="
echo "Group:  $GROUP"
echo "Env:    $ENV"
echo "Steps:  $STEPS"
echo "Betas:  ${BETAS[*]}"
echo ""

for BETA in "${BETAS[@]}"; do
    EXP_NAME="crl_plus_per_beta${BETA}_s${SEED}"
    echo "--- Starting beta=$BETA  (exp: $EXP_NAME) ---"

    python run.py \
        crl_plus \
        --env "$ENV" \
        --total-env-steps "$STEPS" \
        --seed "$SEED" \
        --wandb-group "$GROUP" \
        --exp-name "$EXP_NAME" \
        --hard-positive-beta "$BETA"

    echo "--- Finished beta=$BETA ---"
    echo "GPU memory after run:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    sleep 5
    echo "GPU memory after 5s cooldown:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    echo ""
done

echo "=== Sweep complete ==="
