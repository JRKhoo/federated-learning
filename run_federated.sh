#!/usr/bin/env bash
# Run federated simulation (default: 50 rounds)
# Usage: ./run_federated.sh [rounds] [evaluate_every] [local_epochs]

set -euo pipefail

ROUNDS=${1:-50}
EVAL_EVERY=${2:-10}
LOCAL_EPOCHS=${3:-3}
SEED=${4:-33}

PYTHON=python3

echo "Running federated simulation: rounds=${ROUNDS}, eval_every=${EVAL_EVERY}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
if [ -n "${SEED}" ]; then
	${PYTHON} src/run_federated.py "${ROUNDS}" "${EVAL_EVERY}" "${LOCAL_EPOCHS}" "${SEED}"
else
	${PYTHON} src/run_federated.py "${ROUNDS}" "${EVAL_EVERY}" "${LOCAL_EPOCHS}"
fi