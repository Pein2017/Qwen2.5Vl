#!/usr/bin/env python3
"""
Test script to verify direct configuration setting functionality.
"""
import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.global_config import DirectConfig, set_config, reset_config, get_config


def test_direct_config_setting():
    """Test that we can set global config directly with a DirectConfig object."""
    
    # Reset any existing config
    reset_config()
    
    # Create a minimal DirectConfig object similar to the prototype
    config_dict = {
        # Required core paths and model settings
        "data_root": "/tmp/test_data",
        "model_path": "/tmp/test_model",
        "model_size": "3B",
        "model_max_length": 2048,
        "train_data_path": "/tmp/test_data/train.jsonl",
        "val_data_path": "/tmp/test_data/val.jsonl",
        "teacher_pool_file": "/tmp/test_data/teacher.jsonl",
        
        # Required training parameters
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        
        # Differential learning rates
        "learning_rate": 5e-6,
        "vision_lr": 5e-7,
        "merger_lr": 0,
        "llm_lr": 5e-6,
        "coordinate_lr": 5e-6,
        "adapter_lr": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.001,
        "max_grad_norm": 2.0,
        "lr_scheduler_type": "cosine",
        "max_total_length": 2048,
        
        # Required model configuration
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "use_cache": False,
        "use_cache_inference": True,
        "bf16": True,
        "fp16": False,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "use_flash_attention": True,
        
        # Required vision processing
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        
        # Model architecture
        "model_hidden_size": 3584,
        "model_num_layers": 28,
        "model_num_attention_heads": 28,
        "model_vocab_size": 153713,
        
        # Coordinate token configuration
        "coordinate_config_enable_coordinate_tokens": True,
        "coordinate_config_max_coord_value": 2048,
        "coordinate_config_coord_token_init_std": 0.02,
        "coordinate_config_coordinate_loss_weight": 0.05,
        "coordinate_config_regular_loss_weight": 1.0,
        "coordinate_config_soft_expectation_temperature": 1.0,
        "coordinate_config_use_official_box_tokens": True,
        "coordinate_config_enable_multi_geometry": True,
        "coordinate_config_max_line_coordinates": 50,
        "coordinate_config_box_start_id": 151648,
        "coordinate_config_box_end_id": 151649,
        "coordinate_config_square_start_id": 151650,
        "coordinate_config_square_end_id": 151651,
        "coordinate_config_line_start_id": 151652,
        "coordinate_config_line_end_id": 151653,
        "coordinate_config_enable_validation": True,
        "coordinate_config_enable_caching": True,
        "coordinate_config_batch_processing": True,
        "coordinate_config_strict_coordinate_validation": True,
        "coordinate_config_log_raw_text_on_error": True,
        
        # Chat processor settings
        "chat_processor_max_coord_value": 2048,
        
        # Teacher-student training
        "teacher_ratio": 0.5,
        "num_teacher_samples": 1,
        "teacher_loss_weight": 0.3,
        "student_loss_weight": 1.0,
        
        # Data processing
        "language": "chinese",
        "max_examples": 1,
        "collator_type": "packed",
        
        # Training control
        "detection_freeze_epochs": 0,
        "training_prompt_style": True,
        "use_consistent_prompts": True,
        "remove_unused_columns": False,
        
        # Logging and evaluation (simplified)
        "logging_steps": 1,
        "eval_strategy": "no",
        "eval_steps": 5,
        "save_strategy": "no",
        "save_steps": 100,
        "save_total_limit": 1,
        "log_level": "INFO",
        "report_to": "none",
        "disable_tqdm": False,
        "verbose": True,
        
        # Hardware optimization
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
        
        # Required output directories
        "output_dir": "/tmp/test_output",
        "run_name": "test_direct_config",
        "tb_dir": "/tmp/test_output/tensorboard",
        "logging_dir": "/tmp/test_output/logs",
    }
    
    # Create DirectConfig object
    direct_config = DirectConfig(**config_dict)
    
    # Set the global config directly
    set_config(direct_config)
    
    # Test that we can access the config
    config = get_config()
    
    print("✅ SUCCESS: Direct config setting works!")
    print(f"   Model path: {config.model_path}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Run output dir: {config.run_output_dir}")
    print(f"   Coordinate tokens enabled: {config.coordinate_config_enable_coordinate_tokens}")
    
    # Test auto-derived properties
    print(f"\nAuto-derived properties:")
    print(f"   Tune vision: {config.tune_vision}")
    print(f"   Tune LLM: {config.tune_llm}")
    print(f"   Use differential LR: {config.use_differential_lr}")
    
    # Cleanup
    reset_config()
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_direct_config_setting()