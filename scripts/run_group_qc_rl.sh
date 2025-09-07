#!/usr/bin/env bash
set -euo pipefail

# =====================
# User-configurable vars
# =====================
CONFIG_PATH="configs/rl/group_qc_grpo.yaml"
GPU_DEVICES="0"                      # e.g. "0" single GPU; "0,1,2,3" 4 GPUs; "cpu" to force CPU
MASTER_PORT=29501
PY="/root/miniconda3/envs/ms/bin/python"

# Runtime overrides (optional)
TO_CONSOLE=true
LOG_NAME="run_group_qc_rl.log"
LIMIT_GROUPS_OVERRIDE=""      # e.g. "--limit_groups 10"
DEVICE_OVERRIDE=""            # e.g. "--device cuda:0"

# =====================
# Prepare logging/output dir (best-effort)
# =====================
if [[ "$TO_CONSOLE" == "false" ]]; then
  exec > "$LOG_NAME" 2>&1
fi

echo "🚀 Group QC RL Runner (config=${CONFIG_PATH})"
echo "   🖥️  GPUs: $GPU_DEVICES"

# =====================
# Build ARGS
# =====================
ARGS="--config \"$CONFIG_PATH\""
if [[ -n "$LIMIT_GROUPS_OVERRIDE" ]]; then
  ARGS+=" $LIMIT_GROUPS_OVERRIDE"
fi
if [[ -n "$DEVICE_OVERRIDE" ]]; then
  ARGS+=" $DEVICE_OVERRIDE"
fi

# =====================
# Decide single vs multi-GPU by GPU_DEVICES
# =====================
if [ -z "${GPU_DEVICES}" ] || [ "$GPU_DEVICES" = "cpu" ] || [ "$GPU_DEVICES" = "none" ]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "Running on CPU: $PY -m src_post.grpo_runner $ARGS"
  eval "$PY -m src_post.grpo_runner $ARGS"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_DEVICES"
NPROC=${#GPU_ARR[@]}

if [ "$NPROC" -le 1 ]; then
  echo "Running (single GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES): $PY -m src_post.grpo_runner $ARGS"
  eval "$PY -m src_post.grpo_runner $ARGS"
else
  echo "Running (multi GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc=$NPROC): torchrun --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.grpo_runner $ARGS"
  torchrun --nproc_per_node $NPROC --master_port $MASTER_PORT -m src_post.grpo_runner $(eval echo $ARGS)
fi
