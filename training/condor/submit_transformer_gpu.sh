#!/usr/bin/env bash
# Submit geometric-transformer training to the best currently-idle GPU node.
#
# Thin wrapper over submit_best_gpu.sh (see that script for why host pinning is
# needed at all). Capability 8.0+ because the config trains in bf16, which in
# this pool means the A6000 nodes — also the only ones with enough GPU memory
# for the O(n_lc^2) attention.
#
# Usage:
#   training/condor/submit_transformer_gpu.sh
#   training/condor/submit_transformer_gpu.sh config=configs/geo_transformer.yaml
#   training/condor/submit_transformer_gpu.sh extra_args=--resume

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUB_FILE="training/condor/train_transformer.sub"
export MIN_CAPABILITY="${MIN_CAPABILITY:-8.0}"
exec "${HERE}/submit_best_gpu.sh" "$@"
