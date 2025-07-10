"""
Configuration Module for Qwen2.5-VL Training

This module provides both the new domain-specific configuration system and
backward compatibility with the legacy DirectConfig approach.

New Domain-Specific Usage:
    # Initialize domain-specific configs
    from src.config import init_config_manager
    manager = init_config_manager("configs/base_flat.yaml")
    # Access domain-specific configs
    model_settings = manager.model
    training_params = manager.training
    data_config = manager.data

Legacy Direct Access (Backward Compatible):
    # Initialize legacy config
    from src.config import init_config
    init_config("configs/base_flat.yaml")

    # Access anywhere in the codebase
    from src.config import config
    learning_rate = config.learning_rate
"""

# New domain-specific configuration system
from .config_manager import (
    ConfigManager,
    get_config_manager,
    init_config_manager,
    reset_config_manager,
)
from .domain_configs import (
    DataConfig,
    DetectionConfig,
    InfrastructureConfig,
    ModelConfig,
    TrainingConfig,
)

# Legacy configuration system (backward compatibility)
from .global_config import (
    DirectConfig,
    get_config,
    init_config,
    reset_config,
)


class ConfigAccessor:
    """
    Module-level accessor that supports both new and legacy config systems.

    Provides seamless backward compatibility while enabling migration to
    the new domain-specific configuration approach.
    """

    def __getattr__(self, name):
        # Try new config manager first
        try:
            manager = get_config_manager()
            return getattr(manager, name)
        except RuntimeError:
            pass

        # Fall back to legacy config system
        from .global_config import config as _config

        if _config is None:
            # Auto-initialize with default config for testing/simple usage
            self._auto_initialize_config()
            from .global_config import config as _config

        return getattr(_config, name)

    def _auto_initialize_config(self):
        """Auto-initialize config with default values for testing/simple usage."""
        from pathlib import Path

        # Look for config files in common locations
        config_candidates = [
            "configs/base_flat_v2.yaml",
        ]

        config_path = None
        for candidate in config_candidates:
            if Path(candidate).exists():
                config_path = candidate
                break

        if config_path:
            print(f"🔧 Auto-initializing config from: {config_path}")
            init_config(config_path)
        else:
            # Create minimal config for testing
            print("🔧 Auto-initializing with minimal test config")
            self._create_minimal_config()

    def _create_minimal_config(self):
        """Create minimal config for testing purposes."""
        from .global_config import DirectConfig

        # Create minimal config with only essential values
        minimal_config = DirectConfig(
            # Model settings
            model_path="/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
            model_size="3B",
            model_max_length=8192,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
            use_cache=True,
            # Training settings (minimal)
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            vision_lr=1e-6,
            merger_lr=1e-5,
            llm_lr=1e-6,
            detection_lr=1e-4,
            adapter_lr=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            gradient_checkpointing=False,
            bf16=True,
            fp16=False,
            # Data settings (minimal)
            train_data_path="data/train.jsonl",
            val_data_path="data/val.jsonl",
            data_root="data",
            max_total_length=8192,
            use_candidates=False,
            teacher_pool_file="data/teacher.jsonl",
            num_teacher_samples=0,
            collator_type="standard",
            teacher_ratio=0.0,
            max_examples=1000,
            language="english",
            # Evaluation settings
            eval_strategy="no",
            eval_steps=500,
            save_strategy="no",
            save_steps=500,
            save_total_limit=1,
            # Logging settings
            logging_steps=10,
            logging_dir="logs",
            log_level="INFO",
            report_to="none",
            verbose=False,
            disable_tqdm=False,
            # Detection settings
            detection_enabled=True,
            detection_num_queries=100,
            detection_max_caption_length=50,
            detection_decoder_dim_feedforward_factor=4.0,
            detection_decoder_num_layers=3,
            detection_caption_decoder_dim_feedforward_factor=4.0,
            detection_caption_decoder_num_layers=3,
            detection_head_dropout=0.1,
            detection_adapter_bottleneck_ratio=4,
            detection_adapter_num_layers=2,
            detection_bbox_weight=5.0,
            detection_giou_weight=2.0,
            detection_objectness_weight=1.0,
            detection_caption_weight=1.0,
            detection_focal_loss_gamma=2.0,
            detection_focal_loss_alpha=0.25,
            # Model architecture
            model_hidden_size=2048,
            model_num_layers=28,
            model_num_attention_heads=16,
            model_vocab_size=152064,
            # Performance settings
            use_flash_attention=True,
            mixed_precision="bf16",
            dataloader_num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            batching_strategy="standard",
            remove_unused_columns=False,
            # Output settings
            output_dir="output",
            run_name="test_run",
            tb_dir="tensorboard",
            # Stability settings
            max_consecutive_nan=5,
            max_consecutive_zero=10,
            max_nan_ratio=0.1,
            nan_monitoring_window=100,
            allow_occasional_nan=True,
            nan_recovery_enabled=True,
            learning_rate_reduction_factor=0.5,
            gradient_clip_reduction_factor=0.5,
            # Debug settings
            test_samples=10,
            test_forward_pass=False,
            # Training schedule
            detection_freeze_epochs=0,
            # Vision processing
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            # Teacher-Student weights
            teacher_loss_weight=1.0,
            student_loss_weight=1.0,
        )

        # Manually set the global config
        import src.config.global_config as gc

        gc.config = minimal_config

    def __bool__(self):
        # Check if either config system is initialized
        try:
            manager = get_config_manager()
            return manager.is_initialized()
        except RuntimeError:
            pass

        from .global_config import config as _config

        return _config is not None

    @property
    def manager(self):
        """Access to the new domain-specific config manager."""
        return get_config_manager()


# Create the module-level config accessor
config = ConfigAccessor()

__all__ = [
    # New domain-specific system
    "ConfigManager",
    "init_config_manager",
    "get_config_manager",
    "reset_config_manager",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "DetectionConfig",
    "InfrastructureConfig",
    # Legacy system (backward compatibility)
    "init_config",
    "reset_config",
    "get_config",
    "DirectConfig",
    # Unified access
    "config",
]
