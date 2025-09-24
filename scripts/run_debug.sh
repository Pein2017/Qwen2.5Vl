#!/bin/bash


set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1

# Project paths
PROJECT_ROOT="."
CONFIG_NAME="debug"                      # Config to use: bbu_v2
ARCH="legacy"   # json | legacy
GPU_DEVICES="0"                             # Single GPU for robust debugging
DEEPSPEED_CONFIG="scripts/zero2.json"    # DeepSpeed configuration file

# Logging configuration
LOG_LEVEL="DEBUG"                          # Logging level: INFO (production) | DEBUG (development)
to_console=true                             # true: console output, false: log to run_new.log


setup_environment() {
    echo "🌍 Setting up environment for new architecture..."

    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate ms

    # Core environment variables
    export HF_MODULES_CACHE="${PROJECT_ROOT}/model_cache"
    export HF_HOME="${PROJECT_ROOT}/model_cache"
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

    # Rank-aware logging: enable DEBUG from process start
    export BBU_LOG_LEVEL="$LOG_LEVEL"
    # Enable detailed grouping debug dumps (one-time per phase)
    export BBU_DEBUG_DETAILED=1

    # Distributed training coordination
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=$(generate_random_port)

    # Memory optimization
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # FlashAttention v2 and Triton compatibility
    export TRITON_CACHE_DIR="/tmp/triton_cache"
    export TORCH_COMPILE_DISABLE=1
    export FLASH_ATTENTION_FORCE_CUDNN=0
    # IMPORTANT: Do not set TRITON_CACHE_MANAGER to arbitrary values.
    # Triton expects a value in the form "module_path:ClassName". Invalid values cause crashes.
    if [[ -n "${TRITON_CACHE_MANAGER:-}" && "$TRITON_CACHE_MANAGER" != *:* ]]; then
        echo "⚠️  Ignoring invalid TRITON_CACHE_MANAGER=$TRITON_CACHE_MANAGER (expected 'module_path:ClassName'). Unsetting."
        unset TRITON_CACHE_MANAGER
    else
        unset TRITON_CACHE_MANAGER
    fi

    # Create Triton cache directory if it doesn't exist
    mkdir -p "$TRITON_CACHE_DIR"

    cd "$PROJECT_ROOT"

    echo "✅ Environment configured for new architecture (Python: $(which python))"
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
    
    # Test config loading with selected architecture
    if [[ "$ARCH" == "json" ]]; then
        echo "🧪 Testing config loading with src_new_json..."
        python - <<EOF
from src_new_json.config.config import load_config
try:
    config = load_config('${CONFIG_NAME}')
    print('✅ Config loading successful (src_new_json)')
    try:
        print(f'   Coordinate tokens: {config.coordinate_tokens_enabled}')
    except Exception:
        pass
    print(f'   Teacher ratio: {config.teacher_ratio}')
except Exception as e:
    print(f'❌ Config loading failed (src_new_json): {e}')
    exit(1)
EOF
    else
        echo "🧪 Testing config loading with src_new..."
        python - <<EOF
from src_new.config.config import load_config
try:
    config = load_config('${CONFIG_NAME}')
    print('✅ Config loading successful (src_new)')
    try:
        print(f'   Coordinate tokens: {config.coordinate_tokens_enabled}')
    except Exception:
        pass
    print(f'   Teacher ratio: {config.teacher_ratio}')
except Exception as e:
    print(f'❌ Config loading failed (src_new): {e}')
    exit(1)
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
    
    local TRAIN_SCRIPT
    if [[ "$ARCH" == "json" ]]; then
        TRAIN_SCRIPT="scripts/train_new_json.py"
    else
        TRAIN_SCRIPT="scripts/train_new.py"
    fi
    
    python "$TRAIN_SCRIPT" \
        --config "$CONFIG_NAME" \
        --log_level "$LOG_LEVEL" \
        --max_steps 10
}

launch_deepspeed() {
    # Count GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    echo "🚀 Multi-GPU Training with New Architecture + DeepSpeed"
    echo "   🖥️  GPUs: $NUM_GPUS devices ($GPU_DEVICES)"
    echo "   ⚙️  DeepSpeed Config: $DEEPSPEED_CONFIG"
    echo "   📄 Training Config: $CONFIG_NAME"
    echo "   📊 Log Level: $LOG_LEVEL"
    echo "   🔗 Master Port: $MASTER_PORT"
    
    local TRAIN_SCRIPT
    if [[ "$ARCH" == "json" ]]; then
        TRAIN_SCRIPT="scripts/train_new_json.py"
    else
        TRAIN_SCRIPT="scripts/train_new.py"
    fi
    
    # Launch with torchrun
    torchrun \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$NUM_GPUS" \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG_NAME" \
        --log_level "$LOG_LEVEL" \
        --max_steps 10
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Redirect output based on to_console mode
    if [[ "$to_console" == "false" ]]; then
        exec > run_debug.log 2>&1
    fi
    
    echo "🚀 New Architecture Training Launcher (src_new)"
    echo "   📄 Config: $CONFIG_NAME | 🖥️ GPUs: $GPU_DEVICES | 📊 Log: $LOG_LEVEL"
    if [[ "$ARCH" == "json" ]]; then
        echo "   🏗️  Architecture: src_new_json (pure JSON geometry)"
    else
        echo "   🏗️  Architecture: src_new (legacy)"
    fi
    if [[ "$to_console" == "true" ]]; then
        echo "   🐛 DEBUG MODE: Console output enabled"
    else
        echo "   📄 Output redirected to run_debug.log"
    fi
    
    setup_environment
    determine_deepspeed_usage
    validate_config
    
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

# Run main function with all arguments
main "$@"
