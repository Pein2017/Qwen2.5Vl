#!/usr/bin/env python3
"""
BBU Training Script - Using Direct Configuration System

This training script uses the new direct configuration system that eliminates
parameter passing and provides flat, direct access to all config values.
Clean separation of concerns: Environment (bash) vs Training (Python).

Usage:
    python scripts/train.py --config base_flat --log_level INFO --log_verbose true
"""

# Standard library imports
import argparse
import pathlib
import shutil
import sys
import warnings
from pathlib import Path

# Suppress the specific deprecation warning about Trainer.tokenizer
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is deprecated.*")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import config, init_config
from src.config.config_manager import ConfigManager
from src.logger_utils import (
    configure_global_logging,
    get_training_logger,
)


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    import torch

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BBU Training Script")

    # Core arguments
    parser.add_argument(
        "--config", required=True, help="Config name (e.g., 'base_flat')"
    )

    # Operational modes
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate config only"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print config and exit"
    )
    parser.add_argument(
        "--use-new-config", action="store_true", 
        help="Use new domain-specific configuration system (experimental)"
    )

    # Logging configuration
    parser.add_argument(
        "--log_level", required=True, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--log_verbose", required=True, choices=["true", "false"])
    parser.add_argument(
        "--console_log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    return parser.parse_args()


def create_training_arguments_with_deepspeed():
    """Create TrainingArguments with DeepSpeed configuration using direct config access."""
    import os

    from transformers import TrainingArguments

    # Check if DeepSpeed is enabled via environment variable
    deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    deepspeed_config = os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json")

    # Create training arguments with direct config access
    training_args = TrainingArguments(
        # Output settings
        output_dir=config.run_output_dir,
        run_name=config.run_name,
        # Training parameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        # Training optimizations
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        # Evaluation settings
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # Logging settings
        logging_steps=config.logging_steps,
        logging_dir=config.tensorboard_dir,
        report_to=config.report_to,
        disable_tqdm=config.disable_tqdm,
        # Performance settings
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_prefetch_factor=config.prefetch_factor,
        remove_unused_columns=config.remove_unused_columns,
        # DeepSpeed configuration
        deepspeed=deepspeed_config if deepspeed_enabled else None,
    )

    return training_args


def main():
    """Main training function using direct configuration system."""
    args = parse_args()

    try:
        # =====================================================================
        # CONFIGURATION LOADING
        # =====================================================================
        rank0_print("📄 Loading configuration...")

        # Initialize config system
        config_source_path = f"configs/{args.config}.yaml"
        
        if args.use_new_config:
            rank0_print("🔧 Using new domain-specific configuration system")
            from src.config import init_config_manager
            config_manager = init_config_manager(config_source_path)
            rank0_print(f"✅ New config system loaded: {config_source_path}")
        else:
            rank0_print("📄 Using legacy configuration system")
            init_config(config_source_path)
            rank0_print(f"✅ Legacy config loaded: {config_source_path}")

        # Print config if requested
        if args.print_config:
            if args.use_new_config:
                manager = config.manager
                rank0_print(f"Model: {manager.model.model_path}")
                rank0_print(
                    f"LR: {manager.training.learning_rate}, Epochs: {manager.training.num_train_epochs}"
                )
                rank0_print(
                    f"Batch: {manager.training.per_device_train_batch_size}, Output: {manager.infrastructure.run_output_dir}"
                )
            else:
                rank0_print(f"Model: {config.model_path}")
                rank0_print(
                    f"LR: {config.learning_rate}, Epochs: {config.num_train_epochs}"
                )
                rank0_print(
                    f"Batch: {config.per_device_train_batch_size}, Output: {config.run_output_dir}"
                )
            return 0

        # =====================================================================
        # LOGGING SETUP
        # =====================================================================
        log_verbose = args.log_verbose.lower() == "true"
        console_log_level = (
            args.console_log_level if args.console_log_level else args.log_level
        )

        rank0_print(
            f"📊 Configuring logging: Level={args.log_level}, Verbose={log_verbose}, Console={console_log_level}"
        )
        configure_global_logging(
            log_dir=config.log_file_dir,
            log_level=args.log_level,
            verbose=log_verbose,
            is_training=True,
            console_level=console_log_level,
        )

        logger = get_training_logger()
        logger.info("🚀 BBU Training Started - Direct Configuration System")
        logger.info(f"📄 Config: {args.config}")
        logger.info(
            f"📊 Logging: Level={args.log_level}, Verbose={log_verbose}, Console={console_log_level}"
        )
        logger.info("🌍 Environment: All variables handled by launcher script")

        # =====================================================================
        # CONFIGURATION READY TO USE
        # =====================================================================
        logger.info("🔧 Configuration loaded and validated successfully")

        # Validation only mode
        if args.validate_only:
            logger.info("✅ Configuration validation passed!")
            return 0

        # ---------------------------------------------------------------------
        # Delayed heavy imports (trainer + helpers) *after* config + logging
        # are fully initialised.  This prevents early logger configuration
        # attempts that previously caused the "Config not initialised" error.
        # ---------------------------------------------------------------------

        from src.training.trainer_factory import (
            create_trainer_with_coordinator,
            safe_save_model_for_hf_trainer,
        )  # noqa: E402
        from src.training.trainer import (
            create_trainer,
        )  # noqa: E402

        # =====================================================================
        # TRAINING SETUP - Using Direct Config Access
        # =====================================================================
        logger.info("🔧 Creating TrainingArguments with DeepSpeed configuration...")
        training_args = create_training_arguments_with_deepspeed()

        # Create output directory
        pathlib.Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("🏋️ Creating unified BBU trainer...")
        if args.use_new_config:
            trainer = create_trainer_with_coordinator(training_args=training_args, use_new_config=True)
        else:
            trainer = create_trainer(training_args=training_args)

        # Save configuration for reproducibility by copying the original file
        config_dest_path = pathlib.Path(config.run_output_dir) / f"{args.config}.yaml"
        shutil.copy(config_source_path, config_dest_path)
        logger.info(f"💾 Configuration saved to: {config_dest_path}")

        # =====================================================================
        # TRAINING EXECUTION - Following Official Structure
        # =====================================================================
        logger.info("🚀 Starting training...")

        # Start fresh training from predefined model checkpoint
        logger.info("🆕 Starting fresh training from predefined model checkpoint")
        trainer.train()

        # =====================================================================
        # POST-TRAINING CLEANUP - Following Official Structure
        # =====================================================================

        # Save trainer state
        trainer.save_state()
        logger.info("💾 Trainer state saved")

        # Save image processor - following official approach
        if hasattr(trainer, "processing_class"):
            # Get image processor from the trainer's model setup
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(config.model_path)
            processor.image_processor.save_pretrained(training_args.output_dir)
            logger.info(f"💾 Image processor saved to: {training_args.output_dir}")

        # Re-enable cache after training - following official approach
        trainer.model.config.use_cache = True

        # Safe model saving - following official approach
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
        logger.info(f"💾 Model saved to: {training_args.output_dir}")

        logger.info("✅ Training completed successfully!")
        return 0

    except Exception as e:
        rank0_print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
