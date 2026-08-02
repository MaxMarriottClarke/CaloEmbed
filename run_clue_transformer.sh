#!/usr/bin/env bash
# Run caloembed-run-clue-transformer (geo_transformer embeddings + CLUE at a
# fixed working point) — works both interactively and as a condor job. All
# arguments are forwarded.
#
# Usage:
#   ./run_clue_transformer.sh --run-dir results/training/geo_transformer_a6000_500k \
#       --data /vols/cms/mm1221/cms/Data/d5/100k_mix_sep/hdf5 \
#       --output results/dataframes/d5/transformer_clue_500k_knee \
#       --dc 0.2424 --rhoc 0.01 --do 1.1769 --dm 3.7764

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── environment ──────────────────────────────────────────────────────────────
# Same env as the transformer tune: needs torch + PyG for the embedder and the
# prebuilt CLUEstering .so. 'caloembed' is the env that has both.
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash 2>/dev/null)"
micromamba activate caloembed

source /vols/software/cuda/setup.sh 2>/dev/null || true
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "${REPO_DIR}"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Args : $*"

# The legacy GPU pool is often fully occupied by multi-day jobs, so this also
# runs as a CPU job. '--device cpu' in the arguments selects that mode and skips
# the GPU checks below; otherwise a GPU was requested and we fail fast rather
# than silently grinding through 100k events on CPU.
if [[ " $* " == *" cpu "* ]]; then
    echo "Mode : CPU (threads: ${OMP_NUM_THREADS:-unset})"
else
    set +e
    GPU_QUERY="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1)"
    GPU_QUERY_STATUS=$?
    set -e
    if [[ ${GPU_QUERY_STATUS} -ne 0 ]] || [[ -z "${GPU_QUERY}" ]]; then
        echo "ERROR: GPU was requested but nvidia-smi failed on this node:"
        echo "${GPU_QUERY}"
        exit 1
    fi
    echo "GPU  : $(echo "${GPU_QUERY}" | head -1 | cut -d',' -f1 | xargs)"

    if ! python -c "
from caloembed.clustering.clue import probe_backend
probe_backend('gpu cuda')
print('CLUE backend verified working.')
import torch
print('torch CUDA :', 'yes' if torch.cuda.is_available() else 'NO — embedding falls back to CPU')
"; then
        echo "ERROR: the 'gpu cuda' backend failed on this node."
        echo "       Rebuild the portable binary on the submit node: ./build_clustering.sh"
        exit 1
    fi
fi
echo ""

exec python -m caloembed.scripts.run_clue_transformer "$@"
