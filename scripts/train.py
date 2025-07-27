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
import os
import pathlib
import shutil
import warnings

# Apply compatibility patches early (before any torch/flash_attn imports)
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

# Suppress the specific deprecation warning about Trainer.tokenizer
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is deprecated.*")


from src.config import load_config
from src.logger_utils import (
    configure_global_logging,
    get_training_logger,
)


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
    # Removed --use-new-config flag - using single config system

    # Logging configuration - simplified with rank-aware logging
    parser.add_argument(
        "--log_level",
        required=True,
        choices=["DEBUG", "INFO"],
        help="Logging level: INFO (production) | DEBUG (development)",
    )

    return parser.parse_args()


def create_training_arguments_with_deepspeed(config):
    # Runtime DeepSpeed configuration from environment variables set by run_train.sh
    # These are runtime decisions based on GPU count, not static config values
    deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    deepspeed_config = (
        os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json")
        if deepspeed_enabled
        else None
    )

    from transformers.training_args import TrainingArguments

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
        dataloader_prefetch_factor=config.prefetch_factor
        if config.dataloader_num_workers > 0
        else None,
        remove_unused_columns=config.remove_unused_columns,
        # Parallel training configuration
        # Force single GPU mode when DeepSpeed is disabled to prevent DataParallel
        local_rank=-1 if not deepspeed_enabled else -1,
        # DeepSpeed configuration
        deepspeed=deepspeed_config if deepspeed_enabled else None,
    )

    return training_args


def main():
    """Main training function using direct configuration system."""
    args = parse_args()

    # Get logger early (will be rank-aware after logging configuration)
    logger = get_training_logger()

    try:
        # =====================================================================
        # CONFIGURATION LOADING
        # =====================================================================
        logger.info("Loading configuration...")

        # Initialize config system
        config_name: str = args.config
        config_source_path = f"configs/{config_name}.yaml"
        logger.info("Loading explicit configuration system...")
        config = load_config(config_source_path)
        logger.info(f"Configuration loaded: {config_source_path}")

        # Print config if requested
        if args.print_config:
            logger.info(f"Model: {config.model_path}")
            logger.info(
                f"LR: {config.learning_rate}, Epochs: {config.num_train_epochs}"
            )
            logger.info(
                f"Batch: {config.per_device_train_batch_size}, Output: {config.run_output_dir}"
            )
            return 0

        # =====================================================================
        # LOGGING SETUP
        # =====================================================================
        print(f"📊 Configuring rank-aware logging: Level={args.log_level}")
        configure_global_logging(
            log_dir=config.log_file_dir,
            log_level=args.log_level,
        )

        logger = get_training_logger()
        logger.info("🚀 BBU Training Started - Rank-Aware Logging System")
        logger.info(f"📄 Config: {config_name}")
        logger.info(
            f"📊 Logging: Level={args.log_level} (rank-aware filtering enabled)"
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

        # =====================================================================
        # TRAINING SETUP - Using Direct Config Access
        # =====================================================================
        logger.info("🔧 Creating TrainingArguments with DeepSpeed configuration...")
        training_args = create_training_arguments_with_deepspeed(config)

        # Create output directory
        if training_args.output_dir is not None:
            pathlib.Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(
                "output_dir cannot be None. Please check your configuration."
            )

        logger.info("🏋️ Creating unified BBU trainer...")
        trainer = create_trainer_with_coordinator(
            training_args=training_args, config=config
        )

        # Save configuration for reproducibility by copying the original file
        config_dest_path = pathlib.Path(config.run_output_dir) / f"{config_name}.yaml"
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
            try:
                # Get image processor from the trainer's model setup
                from transformers.models.auto.processing_auto import AutoProcessor

                processor = AutoProcessor.from_pretrained(config.model_path)

                # Check if image processor exists and is not None
                if (
                    hasattr(processor, "image_processor")
                    and processor.image_processor is not None
                ):
                    processor.image_processor.save_pretrained(training_args.output_dir)
                    logger.info(
                        f"💾 Image processor saved to: {training_args.output_dir}"
                    )
                else:
                    logger.warning("⚠️  Image processor is None, skipping save")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save image processor: {e}")

        # Re-enable cache after training - using configured value
        trainer.model.config.use_cache = config.use_cache_inference

        # Safe model saving - following official approach
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
        logger.info(f"💾 Model saved to: {training_args.output_dir}")

        logger.info("✅ Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
