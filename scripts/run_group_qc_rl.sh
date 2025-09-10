#!/usr/bin/env bash
set -euo pipefail

# =====================
# Fixed, self-contained configuration (edit here if needed)
# =====================
CONFIG_PATH="configs/rl/group_qc_grpo.yaml"
GPU_DEVICES="0,1"                    # e.g. "0" for single GPU; "0,1,2,3" for 4 GPUs; use "cpu" to force CPU
MASTER_PORT=29511
PY="/root/miniconda3/envs/ms/bin/python"
TO_CONSOLE=false
LOG_NAME="run_group_qc_rl.log"
DS_CONFIG="scripts/zero2.json"
# Runtime toggles
SKIP_SAVE=1             # 1 to skip saving checkpoints; 0 to save

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  echo "Hint: set CONFIG_PATH at the top of scripts/run_group_qc_rl.sh (absolute path preferred)." >&2
  exit 1
fi

# =====================
# Prepare logging/output (best-effort)
# =====================
if [[ "$TO_CONSOLE" == "false" ]]; then
  exec > "$LOG_NAME" 2>&1
fi

echo "🚀 Group QC RL Runner (config=${CONFIG_PATH})"
echo "   🖥️  GPUs: $GPU_DEVICES"

echo "Using DeepSpeed ZeRO-2 config: $DS_CONFIG"

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
