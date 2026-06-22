#!/usr/bin/env bash
# Run caloembed-tune — works both interactively and as a condor job.
#
# Usage:
#   ./run_tune.sh                                  # uses configs/tune.yaml
#   ./run_tune.sh --config configs/tune.yaml
#   ./run_tune.sh --config configs/tune.yaml --resume

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
echo "Args : ${*:-'(none — using defaults)'}"

# A GPU was requested (request_GPUs=1 in condor/tune.sub) — fail fast if this
# node can't actually provide a working one, rather than silently grinding
# through the tuning run on CPU for tens of hours.
set +e
GPU_QUERY="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1)"
GPU_QUERY_STATUS=$?
set -e
if [[ ${GPU_QUERY_STATUS} -ne 0 ]] || [[ -z "${GPU_QUERY}" ]]; then
    echo "ERROR: GPU was requested but nvidia-smi failed on this node:"
    echo "${GPU_QUERY}"
    exit 1
fi
GPU_NAME="$(echo "${GPU_QUERY}" | head -1 | cut -d',' -f1 | xargs)"
echo "GPU  : ${GPU_NAME}"

# ── verify the pre-built CLUEstering fat binary runs on this GPU ──────────────
# CLUEstering is built ahead of time by build_clustering.sh into a single .so
# that embeds SASS for every GPU in the pool (P100/V100/A6000), so there is no
# in-job recompile. This just confirms the 'gpu cuda' backend actually runs on
# whatever node we landed on — failing fast on a surprise (missing arch, driver
# mismatch) instead of grinding for hours.
if ! python -c "from caloembed.clustering.clue import probe_backend; probe_backend('gpu cuda')"; then
    echo "ERROR: the 'gpu cuda' backend failed on this node (${GPU_NAME})."
    echo "       Rebuild the portable binary on the submit node: ./build_clustering.sh"
    exit 1
fi
echo "GPU backend verified working."
echo ""

# ── run ──────────────────────────────────────────────────────────────────────
caloembed-tune --config configs/tune.yaml "$@"
