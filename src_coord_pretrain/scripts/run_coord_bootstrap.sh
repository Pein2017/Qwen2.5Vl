#!/bin/bash
set -euo pipefail

# Module: src_coord_pretrain bootstrap trainer (Multi‑GPU + DeepSpeed zero2)

# Dynamically determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is current directory
PROJECT_ROOT="."

MODULE_ROOT="src_coord_pretrain"
CONFIG_PATH="src_coord_pretrain/config/coord_bootstrap.yaml"
LOG_PATH="$MODULE_ROOT/run_coord.log"

# Devices (edit this to control number of GPUs)
GPU_DEVICES="2,3,4,5,6,7"  # e.g., "0" or "0,1,2,3"

setup_environment() {
    echo "🌍 Setting up environment for coord bootstrap..."
    export PYTHONPATH="$PROJECT_ROOT"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_MODULES_CACHE="$PROJECT_ROOT/model_cache"
    export HF_HOME="$PROJECT_ROOT/model_cache"
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

    # Disable tqdm/progress bars (align with src_new style)
    export DISABLE_TQDM=1
    export TQDM_DISABLE=1
    export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    eval "$(conda shell.bash hook)"
    conda activate ms

    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMBA_NUM_THREADS=4

    cd "$PROJECT_ROOT"
    echo "✅ Environment configured (Python: $(which python))"
}

random_port() { echo $((20001 + RANDOM % 9999)); }

launch() {
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    MASTER_PORT=$(random_port)
    echo "🚀 Launch: $NUM_GPUS GPUs (port $MASTER_PORT)"

    torchrun \
      --master_port="$MASTER_PORT" \
      --nproc_per_node="$NUM_GPUS" \
      "$MODULE_ROOT/training/trainer.py" \
      --config "$CONFIG_PATH"
}

main() {
    # Redirect all output to log
    mkdir -p "$MODULE_ROOT"
    exec > "$LOG_PATH" 2>&1

    setup_environment
    launch
}

main "$@"
