#!/bin/bash

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1

# Project paths
PROJECT_ROOT="."
# CONFIG_NAME="phase_2/standard"
CONFIG_NAME="phase_1/debug"
ARCH="json"   # json | legacy
MAX_STEPS=""  # e.g., 10 for quick sanity; empty means use config default
to_console=true

# Python interpreter - use ms environment directly
PY="/root/miniconda3/envs/ms/bin/python"


# Set configuration based on experiment number (skipped if GPU_DEVICES provided via env)
if [[ -z "${GPU_DEVICES:-}" ]]; then
    if [[ $# -eq 0 ]]; then
        # Default configuration when no arguments provided
        GPU_DEVICES="0,1"
        LOG_NAME="${LOG_NAME:-run.log}"
        echo "🚀 Default run: GPUs 0,1,2,3,4,5,6,7 → ${LOG_NAME}"
    else
        EXP_NUM="$1"
        case "$EXP_NUM" in
            1)
                GPU_DEVICES="0,1,2,3"
                LOG_NAME="${LOG_NAME:-run_exp_1.log}"
                echo "🚀 Experiment 1: GPUs 0,1,2,3 → ${LOG_NAME}"
                ;;
            2)
                GPU_DEVICES="4,5,6,7"
                LOG_NAME="${LOG_NAME:-run_exp_2.log}"
                echo "🚀 Experiment 2: GPUs 4,5,6,7 → ${LOG_NAME}"
                ;;
            *)
                echo "❌ Invalid experiment number: $EXP_NUM"
                echo "💡 Usage: $0 [1|2]"
                echo "  Default: GPUs 1,2,3,4 → run.log"
                echo "  1: GPUs 0,1,2,3 → run_exp_1.log"
                echo "  2: GPUs 4,5,6,7 → run_exp_2.log"
                exit 1
                ;;
        esac
    fi
else
    echo "🧰 Using GPU devices from environment: $GPU_DEVICES"
    LOG_NAME="${LOG_NAME:-run.log}"
fi

# Fixed configuration
DEEPSPEED_CONFIG="scripts/zero2.json"
LOG_LEVEL="INFO"


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    echo "🌍 Setting up environment for new architecture..."

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
    export OMP_NUM_THREADS=4                    # OpenMP threading (vs restrictive 1)
    export MKL_NUM_THREADS=4                    # Intel MKL threading
    export OPENBLAS_NUM_THREADS=4               # OpenBLAS threading
    export NUMBA_NUM_THREADS=4                  # Numba threading

    # I/O and System Optimization
    export PYTHONUNBUFFERED=1                  # Immediate stdout/stderr (already set)
    export MALLOC_TRIM_THRESHOLD_=100000       # Aggressive memory trimming

    # Quieter framework logs (reduce overhead from verbose shutdown warnings)
    export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
    export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
    # Keep NCCL_DEBUG at WARN by default to avoid massive logs; set INFO only when debugging
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

    # Create Triton cache directory if it doesn't exist
    mkdir -p "$TRITON_CACHE_DIR"

    cd "$PROJECT_ROOT"

    echo "✅ Environment configured for new architecture (Python: $PY)"
    echo "🚀 Optimized setup - removed performance-limiting environment variables"
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
    echo "🔍 Validating configuration for new architecture..."
    
    # Check if config file exists
    if [[ ! -f "configs/${CONFIG_NAME}.yaml" ]]; then
        echo "❌ Configuration file not found: configs/${CONFIG_NAME}.yaml"
        echo "💡 Available configs:"
        ls -1 configs/*.yaml | sed 's/configs\///g' | sed 's/\.yaml//g' | sed 's/^/   - /'
        exit 1
    fi
    
    # Test config loading with selected architecture
    if [[ "$ARCH" == "json" ]]; then
        echo "🧪 Testing config loading with src_new_json..."
        "$PY" - <<EOF
from src_new_json.config.config import load_config
try:
    config = load_config('configs/${CONFIG_NAME}.yaml')
    print('✅ Config loading successful (src_new_json)')
    print(f'   Model path: {config.model_path}')
    print(f'   Teacher ratio: {config.teacher_ratio}')
except Exception as e:
    print(f'❌ Config loading failed (src_new_json): {e}')
    raise
EOF
    else
        echo "🧪 Testing config loading with src_new..."
        "$PY" - <<EOF
from src_new.config.config import load_config
try:
    config = load_config('configs/${CONFIG_NAME}.yaml')
    print('✅ Config loading successful (src_new)')
    print(f'   Model path: {config.model_path}')
    print(f'   Teacher ratio: {config.teacher_ratio}')
except Exception as e:
    print(f'❌ Config loading failed (src_new): {e}')
    raise
EOF
    fi
    
    # Check if DeepSpeed config exists (if enabled)
    if [[ $DEEPSPEED_ENABLED == true ]]; then
        if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
            echo "❌ DeepSpeed config file not found: $DEEPSPEED_CONFIG"
            echo "💡 Please ensure the DeepSpeed config file exists"
            exit 1
        fi
        echo "✅ DeepSpeed config validated: $DEEPSPEED_CONFIG"
    fi
    
    echo "✅ Configuration validation passed for new architecture"
}



# =============================================================================
# TRAINING LAUNCH FUNCTIONS
# =============================================================================

launch_single_gpu() {
    echo "🖥️  Single GPU Training with New Architecture (GPU: ${GPU_DEVICES%%,*})"
    
    if [[ "$ARCH" == "json" ]]; then
        if [[ -n "$MAX_STEPS" ]]; then
            "$PY" "${PROJECT_ROOT}/scripts/train_new_json.py" \
                --config "$CONFIG_NAME" \
                --log_level "$LOG_LEVEL" \
                --max_steps "$MAX_STEPS"
        else
            "$PY" "${PROJECT_ROOT}/scripts/train_new_json.py" \
                --config "$CONFIG_NAME" \
                --log_level "$LOG_LEVEL"
        fi
    else
        "$PY" "${PROJECT_ROOT}/scripts/train_new.py" \
            --config "$CONFIG_NAME" \
            --log_level "$LOG_LEVEL"
    fi
}

launch_deepspeed() {
    # Count GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    echo " Multi-GPU Training with New Architecture + DeepSpeed"
    echo "   🖥️  GPUs: $NUM_GPUS devices ($GPU_DEVICES)"
    echo "   ⚙️  DeepSpeed Config: $DEEPSPEED_CONFIG"
    echo "   📄 Training Config: $CONFIG_NAME"
    echo "   📊 Log Level: $LOG_LEVEL"
    echo "   🔗 Master Port: $MASTER_PORT"
    
    # Select script based on ARCH
    local TRAIN_SCRIPT
    if [[ "$ARCH" == "json" ]]; then
        TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_new_json.py"
    else
        TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_new.py"
    fi
    
    # Launch with torchrun
    torchrun \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$NUM_GPUS" \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG_NAME" \
        --log_level "$LOG_LEVEL" \
        ${MAX_STEPS:+--max_steps "$MAX_STEPS"}
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Redirect output to log file
    if [[ "$to_console" == "false" ]]; then
        exec > $LOG_NAME 2>&1
    fi
    
    echo "🚀 New Architecture Training Launcher (src_new)"
    echo "   📄 Config: $CONFIG_NAME"
    echo "   🖥️  GPUs: $GPU_DEVICES"
    echo "   📄 Log file: $LOG_NAME"
    echo "   📊 Log level: $LOG_LEVEL"
    if [[ "$ARCH" == "json" ]]; then
        echo "   🏗️  Architecture: src_new_json (pure JSON geometry)"
    else
        echo "   🏗️  Architecture: src_new (legacy)"
    fi
    if [[ -n "$MAX_STEPS" ]]; then
        echo "   ⏱️  Max steps override: $MAX_STEPS"
    fi
    if [[ "$to_console" == "true" ]]; then
        echo "   🐛 Console output enabled"
    else
        echo "   📄 Output will be redirected to $LOG_NAME"   
    fi
    echo ""
    
    setup_environment
    determine_deepspeed_usage
    validate_config
    
    # After validation, decide whether to keep FlashAttention/Triton-specific env
    FLASH_ATTENTION_ENABLED=$("$PY" - <<EOF
try:
    from src_new_json.config.config import load_config
except Exception:
    from src_new.config.config import load_config
try:
    cfg = load_config(f"configs/${CONFIG_NAME}.yaml")
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
    
    echo "✅ New architecture training completed successfully!"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Run main function
main
