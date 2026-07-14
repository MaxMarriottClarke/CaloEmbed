#!/usr/bin/env bash
# Submit training/condor/train.sub pinned to the best currently-idle GPU node.
#
# Why this exists: this pool's negotiator ranks whole-node GPU jobs by
# machine Cpus/Memory (fewest wins), which always favours the P100 nodes
# regardless of the job's own `rank` expression. Pinning to a specific
# host via `requirements` sidesteps that policy entirely.
#
# Usage:
#   training/condor/submit_best_gpu.sh                          # configs/default.yaml
#   training/condor/submit_best_gpu.sh config=configs/foo.yaml
#   training/condor/submit_best_gpu.sh extra_args=--resume

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BEST_HOST="$(condor_status -constraint 'TotalGpus > 0 && State == "Unclaimed"' \
    -af Machine GPUs_Capability GPUs_GlobalMemoryMb \
    | sort -k2,2 -k3,3 -nr \
    | head -n1 \
    | awk '{print $1}')"

if [[ -z "${BEST_HOST}" ]]; then
    echo "No idle GPU node found — submitting without a host pin (may queue)." >&2
    exec condor_submit training/condor/train.sub "$@"
fi

echo "Best idle GPU node: ${BEST_HOST}"
exec condor_submit training/condor/train.sub target_gpu_host="\"${BEST_HOST}\"" "$@"
