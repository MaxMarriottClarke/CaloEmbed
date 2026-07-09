#!/usr/bin/env bash
# Run caloembed-run-agglo (GNN embedding + agglomerative clustering) — works
# both interactively and as a condor job. All arguments are forwarded.
#
# Usage:
#   ./run_agglo.sh --run-dir results/training/edgeconv_infonce \
#       --data /path/to/hdf5 --output results/dataframes/d5/gnn_agglo --threshold 0.052

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── environment ──────────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash 2>/dev/null)"
micromamba activate caloembed

source /vols/software/cuda/setup.sh 2>/dev/null || true
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# ── info ─────────────────────────────────────────────────────────────────────
cd "${REPO_DIR}"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Args : ${*}"

# The condor sub requests a GPU. Inference falls back to CPU cleanly, but a GPU
# is ~10x faster for the embedding step, so report what we landed on.
set +e
GPU_QUERY="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)"
GPU_STATUS=$?
set -e
if [[ ${GPU_STATUS} -eq 0 && -n "${GPU_QUERY}" ]]; then
    echo "GPU  : $(echo "${GPU_QUERY}" | head -1 | xargs)"
else
    echo "GPU  : none detected — embedding will run on CPU"
fi
echo ""

# ── run ──────────────────────────────────────────────────────────────────────
exec python -m caloembed.scripts.run_agglomerative "$@"
