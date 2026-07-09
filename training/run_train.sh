#!/usr/bin/env bash
# Run GNN embedding training — works interactively and as a condor job.
#
# Usage:
#   ./run_train.sh                                    # configs/default.yaml
#   ./run_train.sh --config configs/default.yaml --epochs 100
#   ./run_train.sh --resume

set -eo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── environment ──────────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash 2>/dev/null)"
micromamba activate caloembed

source /vols/software/cuda/setup.sh 2>/dev/null || true

# ── info ─────────────────────────────────────────────────────────────────────
cd "${TRAIN_DIR}"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Args : ${*:-'(none — using defaults)'}"

set +e
GPU_QUERY="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"
GPU_QUERY_STATUS=$?
set -e
if [[ ${GPU_QUERY_STATUS} -eq 0 && -n "${GPU_QUERY}" ]]; then
    echo "GPU  : ${GPU_QUERY}"
else
    echo "GPU  : none detected — training will run on CPU"
fi

exec python train.py "$@"
