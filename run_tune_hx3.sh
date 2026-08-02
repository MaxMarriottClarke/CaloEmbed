#!/usr/bin/env bash
# Node-side runner for the hep-hx3-batch H100/H200 pool. Unpacks the bundle
# staged by make_hx3_bundle.sh into the job's scratch directory and runs the raw
# CLUE tuner entirely from local disk — nothing here touches /vols, a shared
# home, or CVMFS, none of which exist on these nodes.
#
# Invoked by condor/tune_hx3.sub as:  run_tune_hx3.sh <seed> <config> [extra args]
#
# Everything is relative to $PWD (the Condor scratch dir), which is where
# payload.tar.gz and data/ have been transferred.

set -eo pipefail

SEED="${1:?usage: run_tune_hx3.sh <seed> <config> [extra args]}"
CONFIG="${2:?usage: run_tune_hx3.sh <seed> <config> [extra args]}"
shift 2

echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Seed : ${SEED}"
echo "Scratch: $(df -h . | tail -1)"

# ── GPU check ────────────────────────────────────────────────────────────────
set +e
GPU_QUERY="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1)"
GPU_STATUS=$?
set -e
if [[ ${GPU_STATUS} -ne 0 ]] || [[ -z "${GPU_QUERY}" ]]; then
    echo "ERROR: GPU requested but nvidia-smi failed:"; echo "${GPU_QUERY}"; exit 1
fi
echo "GPU  : ${GPU_QUERY}"

# ── unpack ───────────────────────────────────────────────────────────────────
echo "Unpacking payload ..."
tar xzf payload.tar.gz
rm -f payload.tar.gz          # don't let Condor try to transfer it back

export PYTHONPATH="${PWD}/repo:${PWD}/repo/CLUEstering:${PWD}/repo/patatune/src"
export LD_LIBRARY_PATH="${PWD}/lib:${LD_LIBRARY_PATH:-}"
PY="${PWD}/env/bin/python"

# ── verify the fat binary runs on this GPU ───────────────────────────────────
# The whole point of build_clustering.sh embedding sm_90 is that this does not
# have to JIT. If sm_90 is missing the driver can still JIT from the sm_86 PTX,
# which works but adds a slow first call — so report which path was taken
# instead of silently eating it.
if ! "${PY}" -c "
from caloembed.clustering.clue import probe_backend
probe_backend('gpu cuda')
print('CLUE gpu cuda backend verified working.')
"; then
    echo "ERROR: the 'gpu cuda' backend failed on this node."
    echo "       Rebuild with sm_90: condor_submit condor/build_clustering.sub"
    echo "       then re-stage:      ./make_hx3_bundle.sh"
    exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────
# --data points at the transferred local copy; --output writes into scratch,
# from where condor/tune_hx3.sub transfers it back to results/param_tuning/.
OUT="tune_out_s${SEED}"
exec "${PY}" -m caloembed.scripts.tune_clue \
    --config "repo/${CONFIG}" \
    --data   "${PWD}/data" \
    --output "${OUT}" \
    --seed   "${SEED}" \
    --backend "gpu cuda" \
    "$@"
