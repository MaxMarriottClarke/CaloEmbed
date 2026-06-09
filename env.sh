#!/usr/bin/env bash
# Source this file to set up the CaloEmbed environment:
#   source env.sh
#
# Two contexts exist:
#   caloembed env  — PyTorch, PyG, h5py, umap, CLUEstering, this package
#   system python3 — ROOT (preprocessing only, one-off)
#
# Preprocessing command (uses system python3, not this env):
#   python3 -m caloembed.scripts.preprocess_root --input-dir ... --output-dir ... --jobs 32

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── micromamba ────────────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash)"
micromamba activate caloembed

source /vols/software/cuda/setup.sh
# ── make the local package importable ────────────────────────────────────────
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

# ── convenience aliases ───────────────────────────────────────────────────────
alias preprocess="/usr/bin/python3 -m caloembed.scripts.preprocess_root"   # system python3 (ROOT)
alias caloembed-test="python -m pytest ${REPO_DIR}/tests/ -v"

echo "caloembed env active | repo: ${REPO_DIR}"
echo "  ML pipeline : python ($(python --version 2>&1 | cut -d' ' -f2))"
echo "  Preprocessing: python3 (system, has ROOT) — use: preprocess --help"
