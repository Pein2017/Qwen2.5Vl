#!/bin/bash

# =============================================================================
# Unified BBU Training Launch Script
# Handles ALL environment variables and GPU/distributed configuration
# Clean separation: Environment (bash) vs Training Parameters (YAML)
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

export PYTHONPATH=/data3/Qwen2.5-VL-main

# Project paths
PROJECT_ROOT="/data3/Qwen2.5-VL-main"

# Training configuration
CONFIG_NAME="base_flat_det"                   # Config to use: base_flat_v2 | base_flat_det
GPU_DEVICES="0,1,2,3,4,5,6,7"               # GPU devices (comma-separated)
DEEPSPEED_CONFIG="scripts/zero2.json"    # DeepSpeed configuration file

# Simplified: Single configuration system

# Logging configuration
LOG_LEVEL="INFO"                          # Logging level: INFO (production) | DEBUG (development)
to_console=false                               # true: console output, false: log to run.log
export TRANSFORMERS_NO_TQDM=1
export DISABLE_TQDM=1
# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    echo "🌍 Setting up environment..."

    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate ms
    
    # Core environment variables
    export HF_MODULES_CACHE="/data3/Qwen2.5-VL-main/model_cache"
    export HF_HOME="/data3/Qwen2.5-VL-main/model_cache"
    export TOKENIZERS_PARALLELISM=false
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
    export TRANSFORMERS_NO_TQDM=1
    export DISABLE_TQDM=1
    
    # Distributed training
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=$(generate_random_port)
    
    # Performance optimizations
    export OMP_NUM_THREADS=1
    export NCCL_DEBUG=WARN
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    # CUDA to_console mode for detailed error info
    # export CUDA_LAUNCH_BLOCKING=1
    # export TORCH_USE_CUDA_DSA=1
    
    
    cd "$PROJECT_ROOT"
    
    echo "✅ Environment configured (Python: $(which python))"
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
        echo "🖥️  Single GPU detected - DeepSpeed DISABLED"
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
    echo "🔍 Validating configuration..."
    
    # Check if config file exists
    if [[ ! -f "configs/${CONFIG_NAME}.yaml" ]]; then
        echo "❌ Configuration file not found: configs/${CONFIG_NAME}.yaml"
        echo "💡 Available configs:"
        ls -1 configs/*.yaml | sed 's/configs\///g' | sed 's/\.yaml//g' | sed 's/^/   - /'
        exit 1
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
    
    echo "✅ Configuration validation passed"
}

# =============================================================================
# TRAINING LAUNCH FUNCTIONS
# =============================================================================

launch_single_gpu() {
    echo "🖥️  Single GPU Training (GPU: ${GPU_DEVICES%%,*})"
    
    python scripts/train.py \
        --config "$CONFIG_NAME" \
        --log_level "$LOG_LEVEL"
}

launch_deepspeed() {
    # Count GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    echo "🚀 Multi-GPU Training with DeepSpeed"
    echo "   🖥️  GPUs: $NUM_GPUS devices ($GPU_DEVICES)"
    echo "   ⚙️  DeepSpeed Config: $DEEPSPEED_CONFIG"
    echo "   📄 Training Config: $CONFIG_NAME"
    echo "   📊 Log Level: $LOG_LEVEL (rank-aware filtering enabled)"
    echo "   🔗 Master Port: $MASTER_PORT (randomly generated)"
    
    # Launch with torchrun (official approach)
    torchrun \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$NUM_GPUS" \
        scripts/train.py \
        --config "$CONFIG_NAME" \
        --log_level "$LOG_LEVEL"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Redirect output based on to_console mode
    if [[ "$to_console" == "false" ]]; then
        exec > run.log 2>&1
    fi
    
    echo "🚀 BBU Training Launcher"
    echo "   📄 Config: $CONFIG_NAME | 🖥️ GPUs: $GPU_DEVICES | 📊 Log: $LOG_LEVEL"
    echo "   🔧 Rank-aware logging: INFO/DEBUG on rank0 only, ERROR/WARNING on all ranks"
    if [[ "$to_console" == "true" ]]; then
        echo "   🐛 DEBUG MODE: Console output enabled"
    else
        echo "   📄 Output redirected to run.log"
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
    
    echo "✅ Training completed successfully!"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Run main function with all arguments
main "$@" 