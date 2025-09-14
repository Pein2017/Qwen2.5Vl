#!/usr/bin/env bash
set -euo pipefail

# =====================
# Fixed, self-contained configuration (edit here if needed)
# Also accepts optional CLI args:
#   $1 = CONFIG_PATH (file path) OR TO_CONSOLE when set to 'true'/'false'
#   $2 = TO_CONSOLE (true|false) when $1 is a config path
# =====================
CONFIG_PATH="configs/rl/group_qc_grpo.yaml"
GPU_DEVICES="${GPU_DEVICES:-0}"                   # e.g. "0" for single GPU; "0,1,2,3" for 4 GPUs; use "cpu" to force CPU
TO_CONSOLE="${1:-false}"            # true to output to console, false to output to log file
MASTER_PORT=29511
PY="/root/miniconda3/envs/ms/bin/python"
LOG_NAME="run_group_qc_rl.log"
DS_CONFIG="scripts/zero2.json"
# Runtime toggles
SKIP_SAVE=1             # 1 to skip saving checkpoints; 0 to save

# CLI parsing: allow first arg to be a config path
if [[ -n "${1:-}" ]]; then
  if [[ -f "$1" ]]; then
    CONFIG_PATH="$1"
    # Shift positional args so $2 becomes $1 for TO_CONSOLE detection
    shift
    if [[ -n "${1:-}" ]]; then
      TO_CONSOLE="$1"
    else
      TO_CONSOLE="false"
    fi
  else
    # If $1 is not a file, interpret it strictly as TO_CONSOLE flag
    case "${1,,}" in
      true|false) TO_CONSOLE="$1" ;;
      *) echo "[WARN] First arg '$1' is not a file; treating it as TO_CONSOLE (true|false)." ;;
    esac
  fi
fi

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  echo "Hint: pass an absolute path as the first argument or edit CONFIG_PATH at the top of scripts/run_group_qc_rl.sh." >&2
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
