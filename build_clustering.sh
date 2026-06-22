#!/usr/bin/env bash
# Build CLUEstering ONCE into a portable, multi-GPU "fat" binary.
#
# Run this on the submit node (no GPU needed — nvcc cross-compiles) whenever
# CLUEstering or the CUDA toolkit changes. The resulting CLUE_GPU_CUDA*.so
# embeds SASS for every GPU in the HEP Condor pool, so the Condor job
# (run_tune.sh) never has to recompile in-the-moment for whatever node it lands
# on.
#
# Pool GPUs (condor_status -af GPUs_DeviceName GPUs_Capability) → CUDA arch:
#   Tesla P100  (lxbgpu) : capability 6.0 → 60
#   Tesla V100  (lxcgpu) : capability 7.0 → 70
#   RTX A6000   (lxdgpu) : capability 8.6 → 86
# Override with: ARCHS="60;70;86;90" ./build_clustering.sh
#
#   ./build_clustering.sh

set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHS="${ARCHS:-60;70;86}"

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
SO="$(compgen -G "${REPO_DIR}/CLUEstering/CLUEstering/lib/CLUE_GPU_CUDA*.so" | head -1)"
echo ""
echo "Embedded GPU architectures in $(basename "${SO}"):"
cuobjdump "${SO}" | grep -iE "arch =" | sort -u
echo ""
echo "Done. Submit jobs with:  condor_submit condor/tune.sub"
