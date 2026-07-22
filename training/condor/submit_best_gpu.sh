#!/usr/bin/env bash
# Submit a training job pinned to the best currently-idle GPU node.
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
#
# Environment:
#   SUB_FILE        submit file to use (default training/condor/train.sub), e.g.
#                   SUB_FILE=training/condor/train_transformer.sub ...
#   MIN_CAPABILITY  skip GPUs below this compute capability when choosing the
#                   host (default 0). The transformer submit file needs 8.0 for
#                   bf16 and sets it automatically.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SUB_FILE="${SUB_FILE:-training/condor/train.sub}"
MIN_CAPABILITY="${MIN_CAPABILITY:-0}"
[[ -f "${SUB_FILE}" ]] || { echo "No such submit file: ${SUB_FILE}" >&2; exit 1; }

BEST_HOST="$(condor_status \
    -constraint "TotalGpus > 0 && State == \"Unclaimed\" && GPUs_Capability >= ${MIN_CAPABILITY}" \
    -af Machine GPUs_Capability GPUs_GlobalMemoryMb \
    | sort -k2,2 -k3,3 -nr \
    | head -n1 \
    | awk '{print $1}')"

if [[ -z "${BEST_HOST}" ]]; then
    echo "No idle GPU node with capability >= ${MIN_CAPABILITY} found —" \
         "submitting without a host pin (may queue)." >&2
    exec condor_submit "${SUB_FILE}" "$@"
fi

echo "Submitting ${SUB_FILE} to best idle GPU node: ${BEST_HOST}"
exec condor_submit "${SUB_FILE}" target_gpu_host="\"${BEST_HOST}\"" "$@"
