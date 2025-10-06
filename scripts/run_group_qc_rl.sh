#!/bin/bash
set -euo pipefail

# =============================================================================
# 🤖 Group QC RL Training Launcher
# =============================================================================
#
# USAGE:
#   bash scripts/run_group_qc_rl.sh
#   RL_CONFIG_PATH=configs/rl/custom_grpo.yaml bash scripts/run_group_qc_rl.sh
#   RL_GPU_DEVICES=0,1,2,3 RL_TO_CONSOLE=true bash scripts/run_group_qc_rl.sh
#
# ENVIRONMENT VARIABLES:
#   RL_CONFIG_PATH       RL config file (default: configs/rl/group_qc_grpo.yaml)
#   RL_GPU_DEVICES       GPUs to use (default: 0)
#   RL_TO_CONSOLE       Output to console: true|false (default: false)
#   RL_MASTER_PORT      Master port for distributed training (default: 29511)
#   RL_PYTHON_BIN       Python interpreter path (default: /root/miniconda3/envs/ms/bin/python)
#   RL_LOG_NAME         Log file name (default: run_group_qc_rl.log)
#   RL_SKIP_SAVE        Skip saving checkpoints: 1|0 (default: 1)
# =============================================================================

show_help() {
    cat << EOF
🤖 Group QC RL Training Launcher

USAGE:
    $0
    RL_CONFIG_PATH=configs/rl/custom_grpo.yaml $0
    RL_GPU_DEVICES=0,1,2,3 RL_TO_CONSOLE=true $0

ENVIRONMENT VARIABLES:
    RL_CONFIG_PATH       RL config file (default: configs/rl/group_qc_grpo.yaml)
    RL_GPU_DEVICES       GPUs to use (default: 0)
    RL_TO_CONSOLE       Output to console: true|false (default: false)
    RL_MASTER_PORT      Master port for distributed training (default: 29511)
    RL_PYTHON_BIN       Python interpreter path (default: /root/miniconda3/envs/ms/bin/python)
    RL_LOG_NAME         Log file name (default: run_group_qc_rl.log)
    RL_SKIP_SAVE        Skip saving checkpoints: 1|0 (default: 1)

EXAMPLES:
    $0                                        # Default training
    RL_CONFIG_PATH=configs/rl/custom_grpo.yaml $0  # Custom config
    RL_GPU_DEVICES=0,1,2,3 $0                 # Multi-GPU training
    RL_TO_CONSOLE=true $0                     # Output to console

FEATURES:
    🎯 Group-level quality control RL training
    🚀 Multi-GPU distributed training support
    📊 Configurable logging and output options
    ⚙️  Environment-aware configuration
EOF
}

# Parse help argument
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

# =====================
# Configuration with environment variables and sensible defaults
# =====================
CONFIG_PATH="${RL_CONFIG_PATH:-configs/rl/group_qc_grpo.yaml}"
GPU_DEVICES="${RL_GPU_DEVICES:-0}"                   # e.g. "0" for single GPU; "0,1,2,3" for 4 GPUs; use "cpu" to force CPU
TO_CONSOLE="${RL_TO_CONSOLE:-false}"                 # true to output to console, false to output to log file
MASTER_PORT="${RL_MASTER_PORT:-29511}"
PY="${RL_PYTHON_BIN:-/root/miniconda3/envs/ms/bin/python}"
LOG_NAME="${RL_LOG_NAME:-run_group_qc_rl.log}"
DS_CONFIG="scripts/zero2.json"
# Runtime toggles
SKIP_SAVE="${RL_SKIP_SAVE:-1}"             # 1 to skip saving checkpoints; 0 to save

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  echo "Hint: set RL_CONFIG_PATH environment variable to an absolute path." >&2
  exit 1
fi

# =====================
# Prepare logging/output (best-effort)
# =====================
if [[ "$TO_CONSOLE" == "false" ]]; then
  echo "Redirecting output to log file: $LOG_NAME"
  exec > "$LOG_NAME" 2>&1
fi

echo "🚀 Group QC RL Runner (config=${CONFIG_PATH})"
echo "   🖥️  GPUs: $GPU_DEVICES"
echo "   📝  Console output: $TO_CONSOLE"

echo "Using DeepSpeed ZeRO-2 config: $DS_CONFIG"

# =====================
# Self-test mode (optional): run single-GPU and multi-GPU (2 GPUs) quick checks
# Enable by setting SELFTEST=1 environment variable
# =====================
if [[ "${SELFTEST:-0}" == "1" ]]; then
  echo "Self-test mode enabled"
  GPU_COUNT=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  fi
  if [[ "$GPU_COUNT" -ge 1 ]]; then
    echo "[SELFTEST] Running single GPU (device 0)"
    echo "Command: CUDA_VISIBLE_DEVICES=0 $PY -m src_post.runner --config \"$CONFIG_PATH\""
    CUDA_VISIBLE_DEVICES="0" SKIP_SAVE=$SKIP_SAVE $PY -m src_post.runner --config "$CONFIG_PATH"
  else
    echo "[SELFTEST] No GPUs found; running CPU"
    echo "Command: CPU $PY -m src_post.runner --config \"$CONFIG_PATH\""
    CUDA_VISIBLE_DEVICES="" SKIP_SAVE=$SKIP_SAVE $PY -m src_post.runner --config "$CONFIG_PATH"
  fi
  if [[ "$GPU_COUNT" -ge 2 ]]; then
    echo "[SELFTEST] Running multi GPU (2 GPUs)"
    echo "Command: CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node 2 --master_port $MASTER_PORT -m src_post.runner --config \"$CONFIG_PATH\""
    CUDA_VISIBLE_DEVICES="0,1" SKIP_SAVE=$SKIP_SAVE $PY -m torch.distributed.run --nproc_per_node 2 --master_port $MASTER_PORT -m src_post.runner --config "$CONFIG_PATH"
  else
    echo "[SELFTEST] Skipping multi-GPU self-test (need >=2 GPUs, found $GPU_COUNT)"
  fi
  echo "[SELFTEST] Completed"
  exit 0
fi

# =====================
# Decide device visibility
# =====================
if [ -z "${GPU_DEVICES}" ] || [ "$GPU_DEVICES" = "cpu" ] || [ "$GPU_DEVICES" = "none" ]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "Running on CPU: $PY -m src_post.runner --config \"$CONFIG_PATH\""
  eval "SKIP_SAVE=$SKIP_SAVE $PY -m src_post.runner --config \"$CONFIG_PATH\""
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_DEVICES"
NPROC=${#GPU_ARR[@]}

# =====================
# Launch (torchrun DDP only)
# =====================
if [ "$NPROC" -le 1 ]; then
  echo "Running (single GPU): $PY -m src_post.runner --config \"$CONFIG_PATH\""
  eval "SKIP_SAVE=$SKIP_SAVE $PY -m src_post.runner --config \"$CONFIG_PATH\""
else
  echo "Running (multi GPU, nproc=$NPROC, DDP): $PY -m torch.distributed.run --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.runner --config \"$CONFIG_PATH\""
  eval "SKIP_SAVE=$SKIP_SAVE $PY -m torch.distributed.run --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.runner --config \"$CONFIG_PATH\""
fi
