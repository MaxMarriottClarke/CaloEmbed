#!/usr/bin/env bash
# Stage a self-contained bundle for the hep-hx3-batch H100/H200 pool.
#
#   ./make_hx3_bundle.sh [config]        # default configs/tune_raw_d5_strat.yaml
#
# Unlike every other node in this pool, the hx3 nodes are containers with NO
# shared filesystem: no /vols, no shared $HOME, no CVMFS (verified — they see
# only an overlay root, a 252GB local scratch and the internet). So the repo,
# the Python environment and the input events all have to be shipped in over
# Condor file transfer. This script builds that payload once; condor/tune_hx3.sub
# then transfers it for every job.
#
# What goes in:
#   env/    the caloembed-tune-slim micromamba env (~900MB, ~300MB packed) —
#           numpy/h5py/pandas/pyarrow/sklearn/matplotlib/dill/pyyaml. Deliberately
#           NOT the 7.4GB 'caloembed' env: torch and PyG are only needed by the
#           transformer tuner, not by the raw one.
#   repo/   caloembed, CLUEstering (incl. the prebuilt .so), patatune, configs
#   lib/    libboost_atomic.so.1.75.0 — the ONLY shared library the CLUE CUDA
#           binary needs that the hx3 image lacks (its CUDA runtime is statically
#           linked, so no libcudart is required). Everything else it wants
#           (libstdc++, libm, libgcc_s, libc) is present on the node.
#   data/   just the HDF5 files the config's select_by_ncp picks (~94MB), not
#           the whole 1000-file directory.

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/tune_raw_d5_strat.yaml}"
STAGE="${STAGE:-/vols/cms/mm1221/hx3stage}"
SLIM_ENV="${HOME}/micromamba/envs/caloembed-tune-slim"
BOOST_LIB="/lib64/libboost_atomic.so.1.75.0"

[[ -d "${SLIM_ENV}" ]] || { echo "ERROR: slim env not found at ${SLIM_ENV}"; exit 1; }
[[ -f "${BOOST_LIB}" ]] || { echo "ERROR: ${BOOST_LIB} not found"; exit 1; }

rm -rf "${STAGE}"
mkdir -p "${STAGE}/payload"/{repo,lib} "${STAGE}/data"

echo "Staging into ${STAGE} ..."

# ── environment ──────────────────────────────────────────────────────────────
# Copied wholesale rather than conda-packed: nothing here is launched through a
# shebang (the runner calls env/bin/python directly), and conda's .so files use
# $ORIGIN-relative RPATHs, so the tree works from any unpack location.
echo "  env  ..."
cp -a "${SLIM_ENV}" "${STAGE}/payload/env"

# ── repo ─────────────────────────────────────────────────────────────────────
echo "  repo ..."
cp -a "${REPO_DIR}/caloembed"              "${STAGE}/payload/repo/"
cp -a "${REPO_DIR}/configs"                "${STAGE}/payload/repo/"
mkdir -p "${STAGE}/payload/repo/CLUEstering" "${STAGE}/payload/repo/patatune/src"
cp -a "${REPO_DIR}/CLUEstering/CLUEstering" "${STAGE}/payload/repo/CLUEstering/"
cp -a "${REPO_DIR}/patatune/src/patatune"   "${STAGE}/payload/repo/patatune/src/"
find "${STAGE}/payload/repo" -name "__pycache__" -type d -prune -exec rm -rf {} +

cp "${BOOST_LIB}" "${STAGE}/payload/lib/"

# ── the exact HDF5 files this config selects ─────────────────────────────────
# select_files_by_n_cp is re-run on the node over the transferred directory. It
# lands on the same set because the pool there contains exactly these files and
# nothing else, so every bucket is taken whole.
echo "  data ..."
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
"${SLIM_ENV}/bin/python" - "$CONFIG" "${STAGE}/data" <<'PY'
import shutil, sys, yaml
from pathlib import Path
from caloembed.data.loader import select_files_by_n_cp

cfg  = yaml.safe_load(open(sys.argv[1]))
dest = Path(sys.argv[2])
data = cfg["data"]
if "select_by_ncp" not in data:
    raise SystemExit("This bundler only supports configs using data.select_by_ncp.")

counts = {int(k): int(v) for k, v in data["select_by_ncp"].items()}
files  = select_files_by_n_cp(data["dir"], counts, seed=data["select_seed"])
for fp in files:
    shutil.copy2(fp, dest / fp.name)
print(f"    {len(files)} files, {sum(f.stat().st_size for f in dest.iterdir()) / 1e6:.0f} MB")
PY

# ── pack ─────────────────────────────────────────────────────────────────────
echo "  packing ..."
tar czf "${STAGE}/payload.tar.gz" -C "${STAGE}/payload" .
rm -rf "${STAGE}/payload"

echo ""
echo "Bundle : ${STAGE}/payload.tar.gz  ($(du -h "${STAGE}/payload.tar.gz" | cut -f1))"
echo "Data   : ${STAGE}/data           ($(du -sh "${STAGE}/data" | cut -f1))"
echo ""
echo "Submit with:  condor_submit condor/tune_hx3.sub seeds=\"42 43 44\""
