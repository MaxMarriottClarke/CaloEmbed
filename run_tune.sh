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
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none detected')"
echo ""

# ── run ──────────────────────────────────────────────────────────────────────
caloembed-tune --config configs/tune.yaml "$@"
