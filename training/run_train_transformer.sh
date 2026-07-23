#!/usr/bin/env bash
# Run geometric-transformer embedding training 
# works interactively and as a condor job. 
# Usage:
#   ./run_train_transformer.sh                                  # configs/geo_transformer.yaml
#   ./run_train_transformer.sh --epochs 100 --batch-size 8
#   ./run_train_transformer.sh --resume

set -eo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# environment 
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/bin/micromamba" shell hook --shell bash 2>/dev/null)"
micromamba activate caloembed

#Needed for lx04
source /vols/software/cuda/setup.sh 2>/dev/null || true

# info 
cd "${TRAIN_DIR}"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Args : ${*:-'(none: using defaults)'}"

set +e
GPU_QUERY="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)"
GPU_QUERY_STATUS=$?
set -e
if [[ ${GPU_QUERY_STATUS} -ne 0 || -z "${GPU_QUERY}" ]]; then
    echo "ERROR: GPU was requested but nvidia-smi failed on this node:"
    echo "${GPU_QUERY}"
    exit 1
fi
echo "GPU  : ${GPU_QUERY}"

# bf16 needs real hardware support (Ampere, capability 8.0+). Do NOT trust
# torch.cuda.is_bf16_supported() for this: it returns True on V100/P100, where
# bf16 is emulated — measured on a V100, bf16 matmul runs at 9.9 TFLOP/s against
# 92.1 for fp16, and a full training step costs 410 ms/event instead of 19.4.
# Silently taking that path would waste a 20x factor, so refuse it outright.
PRECISION="$(python -c "
import sys, yaml
args = sys.argv[1:]
cfg_path = 'configs/geo_transformer.yaml'
if '--config' in args:
    cfg_path = args[args.index('--config') + 1]
print(yaml.safe_load(open(cfg_path))['train'].get('precision', 'fp16'))
" "${@}")"

python -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('ERROR: torch cannot see the GPU on this node.')
major, minor = torch.cuda.get_device_capability()
print(f'torch: {torch.__version__} | CUDA cap {major}.{minor} | precision: ${PRECISION}')
if '${PRECISION}' == 'bf16' and major < 8:
    sys.exit(
        f'ERROR: train.precision is bf16 but this GPU is capability {major}.{minor}. '
        'bf16 is emulated below 8.0 and ~20x slower than fp16 here. '
        'Set train.precision: fp16 in the config, or submit to an Ampere+ node.')
"

# run
ARGS=("$@")
if [[ ! " ${ARGS[*]} " =~ " --config " ]]; then
    ARGS=(--config configs/geo_transformer.yaml "${ARGS[@]}")
fi
exec python train.py "${ARGS[@]}"
