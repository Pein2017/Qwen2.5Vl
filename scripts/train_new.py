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


# Initialize rank-aware logging from environment BEFORE importing patches
# so any import-time logs (e.g., compatibility patches) respect configured level.
try:
    from src_new.utils.rank_aware_logging import initialize_logging_from_env

    initialize_logging_from_env()
except Exception:
    # Safe to ignore; will be configured later in main
    pass

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
    """Get rank-aware logger for training script."""
    try:
        from src_new.utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("train_new")
    except ImportError:
        # Fallback to config system
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
    from transformers import TrainingArguments

    logger = get_logger()

    # Runtime DeepSpeed configuration from environment variables
    deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    deepspeed_config = (
        os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json")
        if deepspeed_enabled
        else None
    )

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
        # Checkpoint optimization settings
        save_safetensors=True,  # Always use SafeTensors for faster loading
        # Best checkpoint tracking settings
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
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
        # Model saving settings - enable SafeTensors for faster inference loading
        save_on_each_node=getattr(
            config, "save_on_each_node", False
        ),  # EFFICIENCY: Only rank 0 saves checkpoints
        # Checkpoint optimization settings for faster training
        dataloader_drop_last=True,  # Reduce coordination overhead
        save_only_model=False,  # Keep full checkpoints for training resumption
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

    # Add fast checkpoint mode configuration (default: True for inference-ready checkpoints)
    logger.info(
        "🚀 Fast checkpoint mode enabled - inference-ready checkpoints only (30s vs 200-400s)"
    )

    return training_args


def perform_pre_distributed_expansion(base_model, tokenizer, config):
    """
    Perform ALL tokenizer and model expansion operations BEFORE distributed training.

    This function centralizes all expensive expansion operations to happen once
    in the main process, preventing conflicts with HuggingFace Trainer's distributed
    coordination mechanisms.

    Args:
        base_model: The base Qwen2.5-VL model
        tokenizer: The base tokenizer
        config: Training configuration

    Returns:
        tuple: (expanded_tokenizer, expanded_model)
    """
    from src_new.processing.token_processor import TokenConfig, TokenProcessor

    logger = get_logger()
    logger.info("🔧 Starting pre-distributed expansion operations...")

    # Only perform expansion if coordinate tokens are enabled
    if not config.coordinate_tokens_enabled:
        logger.info("📋 Coordinate tokens disabled - skipping expansion")
        return tokenizer, base_model

    # Create token processor for expansion
    token_config = TokenConfig(
        coordinate_tokens_enabled=config.coordinate_tokens_enabled,
        max_coord_value=config.max_coord_value,
    )
    token_processor = TokenProcessor(token_config)

    # Record initial vocabulary size
    vocab_size_before = len(tokenizer.get_vocab())
    logger.info(f"📊 Initial vocabulary size: {vocab_size_before}")

    # Check if expansion is needed
    has_extended_vocab = vocab_size_before > 151665
    skip_extension = (
        getattr(config, "skip_vocab_extension", False) or has_extended_vocab
    )

    if skip_extension:
        if has_extended_vocab:
            logger.info(
                f"🚀 Vocabulary already extended ({vocab_size_before} tokens) - skipping expansion"
            )
        else:
            logger.info("🚀 Skipping vocabulary extension (manual override)")
        return tokenizer, base_model

    # Perform tokenizer vocabulary expansion
    logger.info("🔧 Expanding tokenizer vocabulary...")
    expanded_tokenizer = token_processor.extend_tokenizer_vocabulary(tokenizer)

    # Perform model embedding extension
    logger.info("🔧 Expanding model embeddings...")
    expanded_model = token_processor.extend_model_embeddings(
        base_model, expanded_tokenizer
    )

    # Log final vocabulary size
    vocab_size_after = len(expanded_tokenizer.get_vocab())
    logger.info(
        f"✅ Pre-distributed expansion completed: {vocab_size_before} → {vocab_size_after} tokens"
    )

    return expanded_tokenizer, expanded_model


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

    logger = get_logger()
    # Import required components for datasets, collator, and model wrapper
    from src_new.data.collator import create_data_collator
    from src_new.data.dataset import Dataset
    from src_new.data.teacher_pool import TeacherPoolManager
    from src_new.models.wrapper import DetectionModel

    # Load tokenizer and processor
    logger.info(f"Loading tokenizer and processor from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,  # FIXED: Enable fast tokenizer for offset mapping
    )

    # Load image processor separately (following src pattern)
    from transformers import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # Override max_pixels with configured value
    if hasattr(config, "max_pixels"):
        logger.info(f"🖼️ Setting image processor max_pixels to {config.max_pixels}")
        image_processor.max_pixels = config.max_pixels

    # Note: Tokenizer vocabulary extension is now handled by DetectionModel
    # to avoid duplicate processing and ensure consistency

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

    # Load and wrap model with optimized loading strategy
    logger.info(f"Loading model from {config.model_path}")
    from transformers import Qwen2_5_VLForConditionalGeneration

    # Optimized loading parameters for faster initialization
    loading_kwargs = {
        "torch_dtype": getattr(torch, config.torch_dtype),
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
    }

    # Add device_map for direct GPU loading if available and not using DeepSpeed
    if (
        torch.cuda.is_available()
        and not hasattr(config, "deepspeed")
        and torch.cuda.device_count() == 1
    ):
        loading_kwargs["device_map"] = "auto"
        logger.info("🚀 Using direct GPU loading for faster initialization")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path, **loading_kwargs
    )

    # CRITICAL: Perform ALL expansion operations BEFORE distributed training
    # This prevents conflicts with HuggingFace Trainer's distributed coordination
    tokenizer, base_model = perform_pre_distributed_expansion(
        base_model, tokenizer, config
    )

    # Wrap model with detection capabilities (expansion already completed)
    model = DetectionModel(
        base_model=base_model,
        config=config,
        tokenizer=tokenizer,
        skip_expansion=True,  # Skip expansion since it's already done
    )

    # Import BBUTrainer locally to ensure it's available in distributed training
    BBUTrainer = __import__("src_new.training", fromlist=["BBUTrainer"]).BBUTrainer

    # Create trainer with unified checkpoint management (no callback needed)
    # Best checkpoint functionality is now integrated directly into BBUTrainer
    trainer = BBUTrainer(
        model=model,
        processing_class=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # callbacks=[]  # No BestCheckpointCallback needed - integrated into trainer
    )

    # Create and set processor for checkpoint saving with updated components
    from transformers import Qwen2VLProcessor

    # Load processor from pretrained to get the chat template, then update components
    processor = Qwen2VLProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # Update processor with our extended tokenizer and image processor
    # Note: We need to create a new processor instance with updated components
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=processor.chat_template,  # Preserve the original chat template
    )

    # CRITICAL: Set tokenizer as trainer's processing_class for automatic saving
    # HuggingFace Trainer expects processing_class to have get_vocab() method (tokenizer has it, processor doesn't)
    trainer.processing_class = tokenizer
    logger.info(
        "🔧 Set expanded tokenizer as processing_class for automatic checkpoint saving"
    )

    # Set processor for the trainer
    trainer.set_processor(processor)
    logger.info("✅ Processor configured for checkpoint saving with chat template")

    return trainer


def main():
    """Main training function using new src_new architecture."""
    args = parse_args()

    # Setup centralized rank-aware logging system
    from src_new.utils.rank_aware_logging import (
        configure_rank_aware_logging,
        initialize_logging_from_env,
        log_distributed_info,
    )

    # Initialize from environment (BBU_LOG_LEVEL/BBU_LOG_FORMAT), then apply CLI override
    initialize_logging_from_env()
    configure_rank_aware_logging(log_level=args.log_level)

    logger = get_logger()

    try:
        # Force single GPU usage to avoid DataParallel issues
        import os

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("🖥️ Set CUDA_VISIBLE_DEVICES=0 to force single GPU usage")

        # Log distributed training information (rank-aware)
        log_distributed_info(logger)

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

        # Setup centralized rank-aware logging system
        # Logging was configured at the top; no reconfiguration needed here
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

        # Best checkpoint is maintained automatically by unified checkpoint management
        # No need for manual save_final_model call

        # Post-training cleanup
        # Re-enable cache after training
        trainer.model.base_model.config.use_cache = config.use_cache_inference

        logger.info("✅ Training completed successfully with new architecture!")
        logger.info("🔗 Best checkpoint maintained by unified checkpoint management")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
