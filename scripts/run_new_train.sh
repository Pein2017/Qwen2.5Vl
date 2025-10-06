#!/bin/bash

# =============================================================================
# 🚀 Qwen2.5-VL SFT Training Launcher
# =============================================================================
# 
# USAGE:
#   bash scripts/run_new_train.sh
#   CONFIG_NAME=phase_3/standard bash scripts/run_new_train.sh
#   DEBUG_MODE=true CONFIG_NAME=debug bash scripts/run_new_train.sh
#
# FEATURES:
#   - Auto-selects single GPU vs multi-GPU training
#   - Debug mode for quick validation (DEBUG_MODE=true)
#   - Uses standard architecture (src_new)
#   - Flexible GPU selection and logging options
#   - All settings tunable via environment variables
#
# ENVIRONMENT VARIABLES:
#   DEBUG_MODE      true|false - Enable debug mode (default: false)
#   CONFIG_NAME     Config file name without .yaml (default: summary)
#   MAX_STEPS      Override max training steps (default: from config)
#   GPU_DEVICES    GPUs to use, e.g. "0,1,2,3" (default: auto-select)
#   LOG_LEVEL      Log level: DEBUG|INFO|WARN|ERROR (default: auto-select)
#   to_console     true|false - Output to console vs log file (default: false)
#   LOG_NAME       Log file name (default: run.log)
# =============================================================================

set -euo pipefail

show_help() {
    cat << EOF
🚀 Qwen2.5-VL SFT Training Launcher

USAGE:
    $0
    CONFIG_NAME=phase_3/standard $0
    DEBUG_MODE=true CONFIG_NAME=debug $0

ENVIRONMENT VARIABLES:
    DEBUG_MODE         Enable debug mode: true|false (default: false)
    CONFIG_NAME        Config file name without .yaml (default: summary)
    MAX_STEPS         Override max training steps (default: from config)
    GPU_DEVICES       GPUs to use, e.g. "0,1,2,3" (default: auto-select)
    LOG_LEVEL         Log level: DEBUG|INFO|WARN|ERROR (default: auto-select)
    to_console        Output to console: true|false (default: false)
    LOG_NAME          Log file name (default: run.log)

EXAMPLES:
    $0                                        # Default training (summary config)
    DEBUG_MODE=true $0                        # Debug mode training
    CONFIG_NAME=phase_3/standard $0           # Use phase 3 standard config
    GPU_DEVICES=0,1 LOG_LEVEL=INFO $0         # Multi-GPU with custom log level

FEATURES:
    🎯 Auto-selects single GPU vs multi-GPU training
    🐛 Debug mode for quick validation (10 steps, console output)
    🏗️  Standard architecture (src_new)
    🖥️  Flexible GPU selection and logging options
    ⚙️  All settings tunable via environment variables
    📊 Distributed training with DeepSpeed for multi-GPU
EOF
}

# Parse help argument
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

# =============================================================================
# CONFIGURATION - TUNABLE VIA ENVIRONMENT VARIABLES
# =============================================================================

export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1

# ==================== MAIN CONFIGURATION ====================
# These settings can be overridden via environment variables:

# Training mode
DEBUG_MODE="${DEBUG_MODE:-false}"              # true: single GPU, DEBUG logs, 10 steps | false: full training

# Training configuration  
CONFIG_NAME="${CONFIG_NAME:-summary}"         # Config file to use (without .yaml extension)
                                             # Examples: "summary", "phase_3/standard", "debug"
ARCH="${ARCH:-standard}"                      # Architecture: always standard (src_new)
MAX_STEPS="${MAX_STEPS:-}"                    # Override max steps (empty = use config default)
                                             # Examples: "10", "100", "1000"

# GPU configuration
GPU_DEVICES="${GPU_DEVICES:-}"                # GPUs to use (empty = auto-select)
                                             # Examples: "0", "0,1", "0,1,2,3"

# Logging configuration
LOG_LEVEL="${LOG_LEVEL:-}"                    # Log level (empty = auto-select based on mode)
                                             # Options: DEBUG, INFO, WARN, ERROR
to_console="${to_console:-false}"             # Output to console vs log file
LOG_NAME="${LOG_NAME:-run.log}"               # Log file name

# ==================== DERIVED CONFIGURATION ====================
# Auto-configuration based on DEBUG_MODE

if [[ "$DEBUG_MODE" == "true" ]]; then
    echo "🐛 DEBUG MODE ENABLED"
    
    # Auto-configure for debug mode
    [[ -z "$GPU_DEVICES" ]] && GPU_DEVICES="0"
    [[ -z "$LOG_LEVEL" ]] && LOG_LEVEL="DEBUG"
    [[ -z "$MAX_STEPS" ]] && MAX_STEPS="10"
    to_console="true"
    
    echo "   📄 Config: $CONFIG_NAME"
    echo "   🏗️  Architecture: standard (src_new)"
    echo "   🖥️  GPUs: $GPU_DEVICES"
    echo "   📊 Log Level: $LOG_LEVEL"
    echo "   ⏱️  Max Steps: $MAX_STEPS"
    echo "   📺 Console Output: enabled"
else
    echo "🚀 PRODUCTION MODE"
    
    # Set defaults for production mode
    [[ -z "$GPU_DEVICES" ]] && GPU_DEVICES="0,1,2,3,4,5,6,7"
    [[ -z "$LOG_LEVEL" ]] && LOG_LEVEL="INFO"
    
    echo "   📄 Config: $CONFIG_NAME"
    echo "   🏗️  Architecture: standard (src_new)"
    echo "   🖥️  GPUs: $GPU_DEVICES"
    echo "   📊 Log Level: $LOG_LEVEL"
fi

# Fixed configuration
DEEPSPEED_CONFIG="scripts/zero2.json"


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "🌍 Setting up DEBUG environment..."
    else
        echo "🌍 Setting up environment for new architecture..."
    fi

    # Ensure Python path is available
    if [[ ! -x "$PY" ]]; then
        echo "❌ Python interpreter not found at: $PY"
        echo "💡 Please ensure the 'ms' conda environment is installed"
        exit 1
    fi

    # Core environment variables
    export HF_MODULES_CACHE="${PROJECT_ROOT}/model_cache"
    export HF_HOME="${PROJECT_ROOT}/model_cache"
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
    
    # Global logging level for rank-aware logging
    export BBU_LOG_LEVEL="$LOG_LEVEL"

    # Distributed training coordination
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=$(generate_random_port)
    
    # NCCL: enable async error handling to avoid hang on multi-GPU failures
    export NCCL_ASYNC_ERROR_HANDLING=1

    # Memory optimization
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # FlashAttention v2 and Triton compatibility
    export TRITON_CACHE_DIR="/tmp/triton_cache"
    export TORCH_COMPILE_DISABLE=1
    export FLASH_ATTENTION_FORCE_CUDNN=0
    # IMPORTANT: Do not set TRITON_CACHE_MANAGER to arbitrary values.
    # Triton expects a value in the form "module_path:ClassName". Invalid values cause a crash
    # like: ValueError: not enough values to unpack (expected 2, got 1)
    # We explicitly unset it so Triton uses its default file cache manager and TRITON_CACHE_DIR.
    if [[ -n "${TRITON_CACHE_MANAGER:-}" && "$TRITON_CACHE_MANAGER" != *:* ]]; then
        echo "⚠️  Ignoring invalid TRITON_CACHE_MANAGER=$TRITON_CACHE_MANAGER (expected 'module_path:ClassName'). Unsetting."
        unset TRITON_CACHE_MANAGER
    else
        unset TRITON_CACHE_MANAGER
    fi

    # CPU Threading Optimization (56 cores, 8 GPUs)
    if [[ "$DEBUG_MODE" == "true" ]]; then
        export OMP_NUM_THREADS=2                    # Reduced for debug mode
        export MKL_NUM_THREADS=2
        export OPENBLAS_NUM_THREADS=2
        export NUMBA_NUM_THREADS=2
    else
        export OMP_NUM_THREADS=4                    # OpenMP threading (vs restrictive 1)
        export MKL_NUM_THREADS=4                    # Intel MKL threading
        export OPENBLAS_NUM_THREADS=4               # OpenBLAS threading
        export NUMBA_NUM_THREADS=4                  # Numba threading
    fi

    # I/O and System Optimization
    export PYTHONUNBUFFERED=1                  # Immediate stdout/stderr (already set)
    export MALLOC_TRIM_THRESHOLD_=100000       # Aggressive memory trimming

    # Quieter framework logs (reduce overhead from verbose shutdown warnings)
    export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
    export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
    # Keep NCCL_DEBUG at WARN by default to avoid massive logs; set INFO only when debugging
    if [[ "$DEBUG_MODE" == "true" ]]; then
        export NCCL_DEBUG=${NCCL_DEBUG:-INFO}  # More verbose for debug
    else
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    fi

    # Create Triton cache directory if it doesn't exist
    mkdir -p "$TRITON_CACHE_DIR"

    cd "$PROJECT_ROOT"

    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "✅ DEBUG environment configured (Python: $PY)"
    else
        echo "✅ Environment configured for new architecture (Python: $PY)"
        echo "🚀 Optimized setup - removed performance-limiting environment variables"
    fi
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

generate_random_port() {
    # Generate random port between 20001 and 29999 to avoid conflicts
    echo $((20001 + RANDOM % 9999))
}

determine_deepspeed_usage() {
    # Count GPUs from GPU_DEVICES
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    # Auto-determine DeepSpeed usage based on GPU count
    if [[ $NUM_GPUS -gt 1 ]]; then
        DEEPSPEED_ENABLED=true
        echo "🚀 Multi-GPU detected ($NUM_GPUS GPUs) - DeepSpeed ENABLED"
    else
        DEEPSPEED_ENABLED=false
        echo "🖥️  Single GPU detected - DeepSpeed DISABLED (testing new architecture)"
    fi
    
    # Export for Python scripts to access
    export BBU_DEEPSPEED_ENABLED="$DEEPSPEED_ENABLED"
    export BBU_DEEPSPEED_CONFIG="$DEEPSPEED_CONFIG"
    export BBU_NUM_GPUS="$NUM_GPUS"
    
    echo "   🔧 BBU_DEEPSPEED_ENABLED=$BBU_DEEPSPEED_ENABLED"
    echo "   ⚙️  BBU_DEEPSPEED_CONFIG=$BBU_DEEPSPEED_CONFIG"
    echo "   🖥️  BBU_NUM_GPUS=$BBU_NUM_GPUS"
}

validate_config() {
    echo "🔍 Validating configuration for standard architecture..."
    
    # Resolve config path (support legacy phase_x/y names)
    local config_file="configs/${CONFIG_NAME}.yaml"
    local alt_config_file="configs/${CONFIG_NAME//\//_}.yaml"
    if [[ -f "$config_file" ]]; then
        RESOLVED_CONFIG_PATH="$config_file"
    elif [[ -f "$alt_config_file" ]]; then
        RESOLVED_CONFIG_PATH="$alt_config_file"
    else
        echo "❌ Configuration file not found: $config_file"
        echo "💡 Available configs:"
        ls -1 configs/*.yaml | sed 's/configs\///g' | sed 's/\.yaml//g' | sed 's/^/   - /'
        exit 1
    fi
    
    # Test config loading with src_new
    echo "🧪 Testing config loading with src_new..."
    "$PY" - <<EOF
from src_new.config.config import load_config
try:
    config = load_config('${CONFIG_NAME}')
    print('✅ Config loading successful (src_new)')
    print(f'   Model path: {config.model_path}')
    print(f'   Teacher ratio: {config.teacher_ratio}')
except Exception as e:
    print(f'❌ Config loading failed (src_new): {e}')
    raise
EOF
    
    # Check if DeepSpeed config exists (if enabled)
    if [[ $DEEPSPEED_ENABLED == true ]]; then
        if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
            echo "❌ DeepSpeed config file not found: $DEEPSPEED_CONFIG"
            echo "💡 Please ensure the DeepSpeed config file exists"
            exit 1
        fi
        echo "✅ DeepSpeed config validated: $DEEPSPEED_CONFIG"
    fi
    
    echo "✅ Configuration validation passed for standard architecture"
}



# =============================================================================
# TRAINING LAUNCH FUNCTIONS
# =============================================================================

launch_single_gpu() {
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "🐛 DEBUG: Single GPU Training (GPU: ${GPU_DEVICES%%,*})"
    else
        echo "🖥️  Single GPU Training with Standard Architecture (GPU: ${GPU_DEVICES%%,*})"
    fi
    
    # Use standard training script (src_new)
    local TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_new.py"
    
    # Build command with conditional max_steps
    local cmd_args=(
        "$TRAIN_SCRIPT"
        --config "$CONFIG_NAME"
        --log_level "$LOG_LEVEL"
    )
    
    if [[ -n "$MAX_STEPS" ]]; then
        cmd_args+=(--max_steps "$MAX_STEPS")
    fi
    
    "$PY" "${cmd_args[@]}"
}

launch_deepspeed() {
    # Count GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "🐛 DEBUG: Multi-GPU Training + DeepSpeed"
    else
        echo "🚀 Multi-GPU Training with Standard Architecture + DeepSpeed"
    fi
    echo "   🖥️  GPUs: $NUM_GPUS devices ($GPU_DEVICES)"
    echo "   ⚙️  DeepSpeed Config: $DEEPSPEED_CONFIG"
    echo "   📄 Training Config: $CONFIG_NAME"
    echo "   📊 Log Level: $LOG_LEVEL"
    echo "   🔗 Master Port: $MASTER_PORT"
    if [[ -n "$MAX_STEPS" ]]; then
        echo "   ⏱️  Max Steps: $MAX_STEPS"
    fi
    
    # Use standard training script (src_new)
    local TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_new.py"

    # Build distributed command with conditional max_steps
    local dist_args=(
        -m torch.distributed.run
        --master_port "$MASTER_PORT"
        --nproc_per_node "$NUM_GPUS"
        "$TRAIN_SCRIPT"
        --config "$CONFIG_NAME"
        --log_level "$LOG_LEVEL"
    )
    
    if [[ -n "$MAX_STEPS" ]]; then
        dist_args+=(--max_steps "$MAX_STEPS")
    fi
    
    "$PY" "${dist_args[@]}"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Redirect output to log file based on configuration
    if [[ "$to_console" == "false" ]]; then
        exec > $LOG_NAME 2>&1
    fi
    
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "🐛 DEBUG MODE: Standard Architecture Training Launcher (src_new)"
    else
        echo "🚀 Standard Architecture Training Launcher (src_new)"
    fi
    echo "   📄 Config: $CONFIG_NAME"
    echo "   🖥️  GPUs: $GPU_DEVICES"
    echo "   📊 Log level: $LOG_LEVEL"
    echo "   🏗️  Architecture: src_new (standard)"
    if [[ -n "$MAX_STEPS" ]]; then
        echo "   ⏱️  Max steps override: $MAX_STEPS"
    fi
    if [[ "$to_console" == "true" ]]; then
        echo "   📺 Console output enabled"
    else
        echo "   📄 Log file: $LOG_NAME"
    fi
    echo ""
    
    setup_environment
    determine_deepspeed_usage
    validate_config
    
    # After validation, decide whether to keep FlashAttention/Triton-specific env
    FLASH_ATTENTION_ENABLED=$("$PY" - <<EOF
try:
    from src_new.config.config import load_config
    cfg = load_config('${CONFIG_NAME}')
    print('1' if getattr(cfg, 'attn_implementation', 'eager') == 'flash_attention_2' else '0')
except Exception:
    print('0')
EOF
)
    if [[ "$FLASH_ATTENTION_ENABLED" == "1" ]]; then
        echo "✓ Using flash_attention_2 per config — keeping Triton/FlashAttention env"
        # Ensure Triton cache dir exists
        mkdir -p "$TRITON_CACHE_DIR"
    else
        echo "ℹ️ Not using flash_attention_2 — unsetting Triton/FlashAttention env"
        unset FLASH_ATTENTION_FORCE_CUDNN
        unset TRITON_CACHE_DIR
    fi

    
    # Launch training
    if [[ $DEEPSPEED_ENABLED == true ]]; then
        launch_deepspeed
    else
        launch_single_gpu
    fi
    
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "✅ DEBUG training completed successfully!"
    else
        echo "✅ Standard architecture training completed successfully!"
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Run main function
main
