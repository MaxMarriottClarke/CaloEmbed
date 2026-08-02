#!/usr/bin/env bash
# Build CLUEstering ONCE into a portable, multi-GPU "fat" binary.
#
# Run whenever CLUEstering or the CUDA toolkit changes. The resulting
# CLUE_GPU_CUDA*.so embeds SASS for every GPU in the HEP Condor pool, so the
# Condor job (run_tune.sh) never has to recompile in-the-moment for whatever
# node it lands on. No GPU is needed — nvcc cross-compiles.
#
# SUBMIT VIA CONDOR, not directly on a login node:
#
#     condor_submit condor/build_clustering.sub
#
# The login nodes cap the user slice at 16 GiB, and nvcc's 'cicc' frontend
# needs more than that to compile the alpaka CUDA translation unit for all four
# architectures. Running this here dies with "'cicc' died due to signal 9",
# which looks like a compiler bug and is really the OOM killer.
#
# Pool GPUs (condor_status -af GPUs_DeviceName GPUs_Capability) → CUDA arch:
#   Tesla P100  (lxbgpu)        : capability 6.0 → 60
#   Tesla V100  (lxcgpu)        : capability 7.0 → 70
#   RTX A6000   (lxdgpu)        : capability 8.6 → 86
#   NVIDIA H200 (hep-hx3-batch) : capability 9.0 → 90
# Override with: ARCHS="60;70;86" ./build_clustering.sh
#
#   ./build_clustering.sh

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHS="${ARCHS:-60;70;86;90}"

# ── environment ──────────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash 2>/dev/null)"
micromamba activate caloembed
source /vols/software/cuda/setup.sh 2>/dev/null || true

# The node's /tmp is tiny (~1GB) — the alpaka/CUDA template build blows past it.
# Build in a scratch dir on the big repo filesystem instead.
export TMPDIR="${REPO_DIR}/.buildtmp"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "Host  : $(hostname)"
echo "nvcc  : $(command -v nvcc)"
echo "Archs : ${ARCHS}"
echo "Building portable CLUEstering (this takes ~10 min) ..."

# CMAKE_DISABLE_PRECOMPILE_HEADERS=ON: the PCH step is fragile here and buys
# nothing for a one-off build; the output binary is identical without it.
pip install --force-reinstall --no-build-isolation \
    -e "${REPO_DIR}/CLUEstering/" \
    -C cmake.define.BUILD_PYTHON=ON \
    -C "cmake.define.CMAKE_CUDA_ARCHITECTURES=${ARCHS}" \
    -C cmake.define.CMAKE_DISABLE_PRECOMPILE_HEADERS=ON

# ── verify every requested arch made it into the fat binary ──────────────────
# Pin the ABI tag to the env that was just built against: lib/ also holds .so
# files from other Python versions (the py312 training env), and a bare glob
# happily reports one of those instead — which looks like the build silently
# ignored the new arch.
PYTAG="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SO="${REPO_DIR}/CLUEstering/CLUEstering/lib/CLUE_GPU_CUDA${PYTAG}"
[[ -f "${SO}" ]] || { echo "ERROR: expected binary not found: ${SO}"; exit 1; }
echo ""
echo "Embedded GPU architectures in $(basename "${SO}"):"
cuobjdump "${SO}" | grep -iE "arch =" | sort -u
echo ""
echo "Done. Submit jobs with:  condor_submit condor/tune.sub"
