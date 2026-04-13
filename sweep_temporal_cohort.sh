#!/usr/bin/env bash
# Temporal cohort negatives sweep for crl_plus on Humanoid, 10M steps each.
# Compares vanilla CRL (beta=0) with temporal cohort batching,
# and a few beta values to test the interaction.

set -euo pipefail

export XLA_PYTHON_CLIENT_ALLOCATOR=platform

GROUP="crl_plus_temporal_cohort_v1"
ENV="humanoid"
STEPS=10000000
SEED=0

# beta=0.0 with cohort is the pure batching effect (no reweighting).
# beta=0.5,1.0 test whether reweighting helps when negatives are harder.
BETAS=(0.0 0.5 1.0)

echo "=== CRL_PLUS temporal cohort negatives sweep ==="
echo "Group:  $GROUP"
echo "Env:    $ENV"
echo "Steps:  $STEPS"
echo "Betas:  ${BETAS[*]}"
echo ""

for BETA in "${BETAS[@]}"; do
    EXP_NAME="crl_plus_cohort_beta${BETA}_s${SEED}"
    echo "--- Starting beta=$BETA  (exp: $EXP_NAME) ---"

    python run.py \
        crl_plus \
        --env "$ENV" \
        --total-env-steps "$STEPS" \
        --seed "$SEED" \
        --wandb-group "$GROUP" \
        --exp-name "$EXP_NAME" \
        --hard-positive-beta "$BETA" \
        --temporal-cohort-negatives

    echo "--- Finished beta=$BETA ---"
    echo "GPU memory after run:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    sleep 5
    echo "GPU memory after 5s cooldown:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
    echo ""
done

echo "=== Sweep complete ==="
