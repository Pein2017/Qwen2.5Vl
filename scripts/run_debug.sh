#!/bin/bash

# =============================================================================
# 🐛 Debug Mode Wrapper for Qwen2.5-VL Training
# =============================================================================
#
# USAGE:
#   bash scripts/run_debug.sh
#   DEBUG_CONFIG_NAME=phase_3/standard bash scripts/run_debug.sh
#   DEBUG_MAX_STEPS=20 DEBUG_GPU_DEVICES=0,1 bash scripts/run_debug.sh
#
# ENVIRONMENT VARIABLES:
#   DEBUG_CONFIG_NAME     Config to use (default: summary)
#   DEBUG_MAX_STEPS      Max steps override (default: 10)
#   DEBUG_GPU_DEVICES    GPUs to use (default: 0)
#   DEBUG_TO_CONSOLE     Output to console (default: true)
# =============================================================================

set -euo pipefail

show_help() {
    cat << EOF
🐛 Debug Mode Wrapper for Qwen2.5-VL Training

USAGE:
    $0
    DEBUG_CONFIG_NAME=phase_3/standard $0
    DEBUG_MAX_STEPS=20 DEBUG_GPU_DEVICES=0,1 $0

ENVIRONMENT VARIABLES:
    DEBUG_CONFIG_NAME     Config file to use (default: summary)
    DEBUG_MAX_STEPS      Max training steps (default: 10)
    DEBUG_GPU_DEVICES    GPUs to use (default: 0)
    DEBUG_TO_CONSOLE     Output to console (default: true)

EXAMPLES:
    $0                                        # Default debug run
    DEBUG_CONFIG_NAME=debug $0                # Use debug config
    DEBUG_CONFIG_NAME=phase_3/standard $0     # Phase 3 standard config
    DEBUG_MAX_STEPS=20 DEBUG_GPU_DEVICES=0,1 $0  # Custom steps and multi-GPU

FEATURES:
    🎯 Single GPU optimized for quick validation
    📊 DEBUG log level with detailed output
    ⏱️  Limited steps for fast iteration
    📺 Console output enabled by default
    🏗️  Uses standard architecture (src_new)
EOF
}

# Parse help argument
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration with environment variables and sensible defaults
CONFIG_NAME="${DEBUG_CONFIG_NAME:-summary}"
MAX_STEPS="${DEBUG_MAX_STEPS:-10}"
GPU_DEVICES="${DEBUG_GPU_DEVICES:-0}"
TO_CONSOLE="${DEBUG_TO_CONSOLE:-true}"

echo "🐛 DEBUG WRAPPER: Quick training validation mode"
echo "   📄 Config: $CONFIG_NAME"
echo "   🏗️  Architecture: standard (src_new)"
echo "   ⏱️  Max Steps: $MAX_STEPS"
echo "   🖥️  GPUs: $GPU_DEVICES"
echo "   📺 Console Output: $TO_CONSOLE"
echo ""

# Export debug configuration as environment variables for run_new_train.sh
export DEBUG_MODE=true
export CONFIG_NAME="$CONFIG_NAME"
export ARCH="standard"
export MAX_STEPS="$MAX_STEPS"
export GPU_DEVICES="$GPU_DEVICES"
export LOG_LEVEL="DEBUG"
export to_console="$TO_CONSOLE"

# Enable detailed debug dumps
export BBU_DEBUG_DETAILED=1

# Execute the main training script with debug configuration
exec bash "${SCRIPT_DIR}/run_new_train.sh"
