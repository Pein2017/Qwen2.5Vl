#!/usr/bin/env python3
"""
New Architecture Training Script - Using BBUTrainer

This training script uses the new src_new architecture with BBUTrainer
for reliable local loss component aggregation that eliminates NCCL timeout issues.
The problematic DistributedLossTrainer has been replaced with the proven BBUTrainer pattern.

Key Features:
- BBUTrainer: Local loss aggregation with TrainingStateManager (no distributed conflicts)
- No NCCL timeouts: Eliminates custom distributed communication conflicts
- Automatic loss component tracking: Teacher/student loss breakdown without distributed operations
- DeepSpeed compatible: Works seamlessly with ZeRO-2 optimization
- Proven pattern: Based on working src/ implementation

Usage:
    python scripts/train_new.py --config bbu_v2 --log_level INFO
"""

import argparse
import logging
import os
import pathlib
import shutil
import warnings
from typing import TYPE_CHECKING

import torch

# Apply compatibility patches early
from src_new.models.patches import apply_comprehensive_qwen25_fixes


# Type checking imports
if TYPE_CHECKING:
    from transformers import TrainingArguments

    from src_new.config.config import Config


apply_comprehensive_qwen25_fixes()

# Import BBUTrainer after patches are applied to avoid import issues


# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is deprecated.*")


def get_logger():
    """Get logger for training script."""
    from src_new.config.config import _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL

    logger = logging.getLogger("train_new")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_GLOBAL_LOG_LEVEL)
        _CONFIGURED_LOGGERS.add("train_new")
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="New Architecture Training Script")

    # Core arguments
    parser.add_argument("--config", required=True, help="Config name (e.g., 'bbu_v2')")

    # Operational modes
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate config only"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print config and exit"
    )
    parser.add_argument(
        "--test-trainer", action="store_true", help="Test trainer creation and exit"
    )

    # Logging configuration
    parser.add_argument(
        "--log_level",
        required=True,
        help="Logging level: INFO (production) | DEBUG (development)",
    )

    # Training control
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of training steps (for testing)",
    )

    return parser.parse_args()


def create_training_arguments_with_deepspeed(config: "Config", max_steps=None):
    """Create TrainingArguments with DeepSpeed configuration."""
    # Runtime DeepSpeed configuration from environment variables
    deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    deepspeed_config = (
        os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json")
        if deepspeed_enabled
        else None
    )

    from transformers import TrainingArguments

    # Create training arguments with direct config access
    training_args = TrainingArguments(
        # Output settings
        output_dir=config.output_dir,
        run_name=config.run_name,
        # Training parameters
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
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
        logging_dir=config.logging_dir,
        report_to=config.report_to,
        disable_tqdm=config.disable_tqdm,
        # Performance settings
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_prefetch_factor=config.prefetch_factor
        if config.dataloader_num_workers > 0
        else None,
        remove_unused_columns=config.remove_unused_columns,
        # Model saving settings - disable safe serialization to avoid shared tensor issues
        save_safetensors=False,
        # NCCL timeout optimization - reduce evaluation frequency during distributed training
        eval_delay=0,  # No delay before first evaluation
        eval_accumulation_steps=1,  # Reduce evaluation accumulation to minimize memory
        # Device settings - force single GPU to avoid DataParallel issues
        local_rank=-1,  # Disable distributed training
        # Use default distributed training timeout (1800 seconds) for stability
        # Removed ddp_timeout=10 override that caused NCCL timeout issues
        # DeepSpeed configuration
        deepspeed=deepspeed_config if deepspeed_enabled else None,
    )

    return training_args


def create_trainer_with_new_architecture(
    training_args: "TrainingArguments", config: "Config"
):
    """
    Create trainer using BBUTrainer with local loss aggregation.

    This function creates a BBUTrainer that uses TrainingStateManager for local
    loss component aggregation, eliminating the NCCL timeout issues that plagued
    the DistributedLossTrainer approach.

    Returns:
        BBUTrainer: Trainer with local loss aggregation and no distributed conflicts
    """
    from transformers import AutoTokenizer

    # Import from collator module
    from src_new.data.collator import create_data_collator
    from src_new.data.dataset import Dataset
    from src_new.data.teacher_pool import TeacherPoolManager
    from src_new.models.wrapper import DetectionModel

    logger = get_logger()

    # Load tokenizer and processor
    logger.info(f"Loading tokenizer and processor from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=False
    )

    # Load image processor separately (following src pattern)
    from transformers import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # Extend tokenizer vocabulary if coordinate tokens are enabled
    if config.coordinate_tokens_enabled:
        logger.info("🔧 Extending tokenizer vocabulary for coordinate tokens...")
        from src_new.processing.token_processor import TokenConfig, TokenProcessor

        token_config = TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=config.max_coord_value,
        )
        token_processor = TokenProcessor(token_config)
        tokenizer = token_processor.extend_tokenizer_vocabulary(tokenizer)
        logger.info("✅ Tokenizer vocabulary extended")

    # Load teacher pool
    teacher_pool_manager = TeacherPoolManager(
        teacher_pool_file=config.teacher_pool_file
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = Dataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=teacher_pool_manager,
        config=config,
    )

    val_dataset = Dataset(
        data_path=config.val_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=teacher_pool_manager,
        config=config,
    )

    # Create data collator
    data_collator = create_data_collator(
        collator_type=config.collator_type, tokenizer=tokenizer, config=config
    )

    # Load and wrap model
    logger.info(f"Loading model from {config.model_path}")
    from transformers import Qwen2_5_VLForConditionalGeneration

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=getattr(torch, config.torch_dtype),
        attn_implementation=config.attn_implementation,
        trust_remote_code=False,
    )

    # Wrap model with detection capabilities
    model = DetectionModel(base_model=base_model, config=config, tokenizer=tokenizer)

    # Import BBUTrainer locally to ensure it's available in distributed training
    BBUTrainer = __import__("src_new.training", fromlist=["BBUTrainer"]).BBUTrainer

    # Create trainer with local loss aggregation (no distributed conflicts)
    trainer = BBUTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    return trainer


def main():
    """Main training function using new src_new architecture."""
    args = parse_args()
    logger = get_logger()

    try:
        # Force single GPU usage to avoid DataParallel issues
        import os

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("🖥️ Set CUDA_VISIBLE_DEVICES=0 to force single GPU usage")

        # Load configuration
        logger.info("Loading configuration...")
        from src_new.config.config import load_config

        config_path = f"configs/{args.config}.yaml"
        config = load_config(config_path)
        logger.info(f"Configuration loaded: {config_path}")

        # Print config if requested
        if args.print_config:
            logger.info(f"Model: {config.model_path}")
            logger.info(
                f"LR: {config.learning_rate}, Epochs: {config.num_train_epochs}"
            )
            logger.info(
                f"Batch: {config.per_device_train_batch_size}, Output: {config.output_dir}"
            )
            logger.info(
                f"Coordinate tokens: {'enabled' if config.coordinate_tokens_enabled else 'disabled'}"
            )
            return 0

        # Validation only mode
        if args.validate_only:
            logger.info("✅ Configuration validation passed!")
            return 0

        # Test trainer creation mode
        if args.test_trainer:
            logger.info("🧪 Testing trainer creation...")

            # Create training arguments
            training_args = create_training_arguments_with_deepspeed(
                config, args.max_steps
            )

            # Create trainer using new architecture
            trainer = create_trainer_with_new_architecture(training_args, config)

            logger.info("✅ Trainer creation test passed!")
            logger.info(f"   - Model: {type(trainer.model).__name__}")
            logger.info("   - Datasets and model loaded successfully")
            return 0

        # Setup logging level using centralized system
        from src_new.config.config import set_global_log_level

        set_global_log_level(args.log_level)
        logger.info(f"🔧 Global logging level set to: {args.log_level}")

        # Create training arguments
        logger.info("Creating TrainingArguments with DeepSpeed configuration...")
        training_args = create_training_arguments_with_deepspeed(config, args.max_steps)

        # Create output directory
        output_dir = training_args.output_dir or "."
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create trainer using new architecture
        logger.info("Creating trainer with new architecture...")
        trainer = create_trainer_with_new_architecture(training_args, config)

        # Save configuration for reproducibility
        config_dest_path = pathlib.Path(config.output_dir) / f"{args.config}.yaml"
        shutil.copy(config_path, config_dest_path)
        logger.info(f"💾 Configuration saved to: {config_dest_path}")

        # Start training
        logger.info("🚀 Starting training with new architecture...")
        trainer.train()

        # Post-training cleanup
        # Custom state saving (replacing trainer.save_state())
        import os

        state_dict = {
            "optimizer": getattr(trainer, "optimizer", None),
            "lr_scheduler": getattr(trainer, "lr_scheduler", None),
            "epoch": getattr(getattr(trainer, "state", None), "epoch", None),
        }

        # Ensure output_dir exists and is a string
        output_dir = training_args.output_dir or "."
        state_path = os.path.join(output_dir, "trainer_state.pt")
        torch.save(state_dict, state_path)
        logger.info(f"💾 Trainer state saved to {state_path}")

        # Re-enable cache after training
        trainer.model.base_model.config.use_cache = config.use_cache_inference

        # No need to save model again if custom Trainer already handles it
        logger.info("✅ Training completed successfully with new architecture!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
