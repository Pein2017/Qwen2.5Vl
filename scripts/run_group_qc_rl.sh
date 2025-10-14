#!/bin/bash
set -euo pipefail

# =====================
# Group QC RL Runner Script
# 
# Environment variables (lowercase, short names):
#   config      - Path to YAML config file (default: configs/group_qc_rl/standard.yaml)
#   gpus        - GPU devices (default: "0", use "cpu" for CPU-only)
#   console     - Output to console vs log file (default: "false")
#   port        - Master port for DDP (default: 29511)
#   python      - Python binary path (default: /root/miniconda3/envs/ms/bin/python)
#   log         - Log file name (default: run_group_qc_rl.log)
#   skip_save   - Skip checkpoint saving (default: "1")
#   launcher    - Launch method: "ddp" or "accelerate" (default: "ddp")
#   procs       - Number of processes (default: auto-detect from gpus)
#   precision   - Mixed precision: "bf16", "fp16", "no" (default: "bf16")
#   selftest    - Run self-test mode (default: "0")
# =====================

# =====================
# Configuration with environment variables and sensible defaults
# =====================
CONFIG_PATH="${config:-configs/group_qc_rl/standard.yaml}"
GPU_DEVICES="${gpus:-0}"                   # e.g. "0" for single GPU; "0,1,2,3" for 4 GPUs; use "cpu" to force CPU
TO_CONSOLE="${console:-false}"                 # true to output to console, false to output to log file
MASTER_PORT="${port:-29511}"
PY="${python:-/root/miniconda3/envs/ms/bin/python}"
LOG_NAME="${log:-run_group_qc_rl.log}"
DS_CONFIG="scripts/zero2.json"
# Runtime toggles
SKIP_SAVE="${skip_save:-1}"             # 1 to skip saving checkpoints; 0 to save
LAUNCHER="${launcher:-ddp}"
NUM_PROCS_ENV="${procs:-}"
MIXED_PRECISION="${precision:-bf16}"

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  echo "Hint: set 'config' environment variable to an absolute path." >&2
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
echo "Launcher: $LAUNCHER"

# =====================
# Self-test mode (optional): run single-GPU and multi-GPU (2 GPUs) quick checks
# Enable by setting selftest=1 environment variable
# =====================
if [[ "${selftest:-0}" == "1" ]]; then
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
if [[ -n "$NUM_PROCS_ENV" ]]; then
  NPROC="$NUM_PROCS_ENV"
fi

# =====================
# Launch
# =====================
if [[ "$LAUNCHER" == "accelerate" ]]; then
  echo "Running (accelerate, nproc=$NPROC, mp=$MIXED_PRECISION): accelerate launch --num_processes $NPROC --mixed_precision $MIXED_PRECISION $PY -m src_post.runner --config \"$CONFIG_PATH\""
  eval "SKIP_SAVE=$SKIP_SAVE accelerate launch --num_processes $NPROC --mixed_precision $MIXED_PRECISION $PY -m src_post.runner --config \"$CONFIG_PATH\""
else
  if [ "$NPROC" -le 1 ]; then
    echo "Running (single GPU): $PY -m src_post.runner --config \"$CONFIG_PATH\""
    eval "SKIP_SAVE=$SKIP_SAVE $PY -m src_post.runner --config \"$CONFIG_PATH\""
  else
    echo "Running (multi GPU, nproc=$NPROC, DDP): $PY -m torch.distributed.run --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.runner --config \"$CONFIG_PATH\""
    eval "SKIP_SAVE=$SKIP_SAVE $PY -m torch.distributed.run --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.runner --config \"$CONFIG_PATH\""
  fi
fi
