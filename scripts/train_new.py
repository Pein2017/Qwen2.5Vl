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
import os
import warnings
from typing import TYPE_CHECKING

import torch

from src_new.utils.seeding import seed_everything


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
    from src_new.utils.logger_factory import get_training_logger

    return get_training_logger("train_new")


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
        output_dir=config.run_output_dir,  # Use computed property for output/{run_name}/
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
        logging_dir=config.tensorboard_dir,  # TensorBoard events go to tb/{run_name}/
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
        # Device settings - rely on torchrun/DeepSpeed env for ranks
        # Use default distributed training timeout (1800 seconds) for stability
        # Removed ddp_timeout=10 override that caused NCCL timeout issues
        # DeepSpeed configuration
        deepspeed=deepspeed_config if deepspeed_enabled else None,
        # Reproducibility
        seed=int(getattr(config, "seed", 17)),
        data_seed=int(getattr(config, "seed", 17)),
    )

    # Add fast checkpoint mode configuration (default: True for inference-ready checkpoints)
    logger.info(
        "🚀 Fast checkpoint mode enabled - inference-ready checkpoints only (30s vs 200-400s)"
    )

    return training_args


def _validate_preexpanded_checkpoint(tokenizer, model, config) -> None:
    """Strictly validate that checkpoint is pre-expanded when coords are enabled.

    Raises ValueError/AssertionError with actionable messages if validation fails.
    """
    logger = get_logger()
    if not getattr(config, "coordinate_tokens_enabled", False):
        return

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    if vocab_size <= 151665:
        raise ValueError(
            f"Coordinate tokens are enabled but tokenizer is not extended: vocab_size={vocab_size} (expected > 151665). "
            f"Please run the migration script to export an expanded checkpoint and point model_path to it."
        )

    # Required tokens
    missing = []
    for tok in ("<|line_start|>", "<|line_end|>"):
        if tok not in vocab:
            missing.append(tok)
    if missing:
        raise ValueError(
            f"Missing required geometry tokens in tokenizer: {missing}. Ensure you used the official base model and migration script."
        )

    # Coordinate token ID range check
    max_coord = int(getattr(config, "max_coord_value", -1))
    if max_coord < 0:
        raise ValueError(
            "max_coord_value must be set when coordinate tokens are enabled"
        )

    coord_ids = []
    for i in range(max_coord + 1):
        tok = f"<|coord_{i}|>"
        if tok not in vocab:
            raise ValueError(
                f"Missing coordinate token '{tok}' in tokenizer. Expected full range 0..{max_coord}."
            )
        coord_ids.append(vocab[tok])

    hard_start = 151667
    if min(coord_ids) != hard_start or max(coord_ids) != hard_start + max_coord:
        raise ValueError(
            "Tokenizer coordinate token range mismatch. "
            f"Expected inclusive range [{hard_start}, {hard_start + max_coord}] but found "
            f"[{min(coord_ids)}, {max(coord_ids)}]. Ensure the checkpoint was exported via the migration script."
        )

    # Embedding shape checks
    in_emb = model.get_input_embeddings()
    num_rows = int(in_emb.weight.shape[0])
    if num_rows < vocab_size:
        raise AssertionError(
            f"Model input embeddings smaller than tokenizer: rows={num_rows}, vocab={vocab_size}. "
            f"Expanded checkpoint must include resized embeddings."
        )
    if (num_rows % 128) != 0:
        raise AssertionError(
            f"Model embeddings must be padded to a multiple of 128 rows; got {num_rows}."
        )

    logger.info(
        f"✅ Pre-expanded checkpoint validated: tokenizer={vocab_size}, embeddings_rows={num_rows}, max_coord={max_coord}"
    )


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

    # Override image processor configuration explicitly from config (fail-fast)
    if hasattr(config, "max_pixels") and int(config.max_pixels) > 0:
        if not hasattr(image_processor, "max_pixels"):
            raise ValueError("Qwen2VLImageProcessor missing 'max_pixels' attribute")
        image_processor.max_pixels = int(config.max_pixels)
        logger.info(f"🔧 Set image_processor.max_pixels={image_processor.max_pixels}")

    # Create teacher pool manager (lazy loading of teachers)
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

    # Strictly require pre-expanded checkpoints if coordinate tokens are enabled
    _validate_preexpanded_checkpoint(tokenizer, base_model, config)

    # Wrap model with detection capabilities (no expansion performed here)
    model = DetectionModel(
        base_model=base_model,
        config=config,
        tokenizer=tokenizer,
        skip_expansion=True,  # Expansion is externalized via migration script
    )

    # Create necessary directories for logging and TensorBoard
    from pathlib import Path

    Path(config.run_output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_file_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"📁 Created directories: output={config.run_output_dir}, tb={config.tensorboard_dir}"
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

    # Pre-create optimizer so that all params are registered once
    try:
        trainer.create_optimizer()
        logger.info("✅ Pre-created optimizer before staged freezing")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-create optimizer: {e}")

    # Register progressive unfreeze callback when enabled via YAML
    try:
        if getattr(config, "prog_unfreeze_enabled", False):
            from src_new.training.callbacks import ProgressiveUnfreezeCallback

            callback = ProgressiveUnfreezeCallback(
                freeze_vision_llm_epochs=int(
                    getattr(config, "prog_unfreeze_epoch_stage1_end", 1)
                    or getattr(config, "num_train_epochs", 1)
                ),
                coord_slice_only=bool(
                    getattr(config, "prog_unfreeze_coord_slice_only", True)
                ),
                stage0_end_epoch=getattr(
                    config, "prog_unfreeze_epoch_stage0_end", None
                ),
                stage1_end_epoch=getattr(
                    config, "prog_unfreeze_epoch_stage1_end", None
                ),
                top_k_layers=getattr(config, "prog_unfreeze_top_k_layers", None),
            )
            # Store trainer reference for HF compatibility
            callback._trainer_ref = trainer
            trainer.add_callback(callback)
            logger.info(
                "✅ Registered ProgressiveUnfreezeCallback (staged unfreeze enabled)"
            )
        else:
            logger.info(
                "🔧 Progressive unfreeze disabled by config - training with standard setup"
            )
    except Exception as e:
        logger.warning(f"⚠️ Could not register ProgressiveUnfreezeCallback: {e}")

    # Create and set processor for checkpoint saving with updated components
    from transformers import Qwen2VLProcessor

    # Load processor from pretrained to get the chat template, then update components
    processor = Qwen2VLProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    # STRICT VALIDATION: Ensure chat template is present (no fallbacks)
    processor_chat_template = getattr(processor, "chat_template", None)
    tokenizer_chat_template = getattr(tokenizer, "chat_template", None)

    # Prefer tokenizer's chat template (authoritative), else use processor's
    authoritative_chat_template = tokenizer_chat_template or processor_chat_template

    if not authoritative_chat_template:
        raise ValueError(
            "chat_template is missing. Ensure your checkpoint contains a valid chat_template.json "
            f"or tokenizer_config.json with 'chat_template'. Checked path: {config.model_path}"
        )

    # Update processor with our tokenizer and image processor
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=authoritative_chat_template,
    )

    logger.info("✅ Chat template validated and set from checkpoint/tokenizer")

    # CRITICAL: Set tokenizer as trainer's processing_class for automatic saving
    # HuggingFace Trainer expects processing_class to have get_vocab() method (tokenizer has it, processor doesn't)
    trainer.processing_class = tokenizer
    logger.info("🔧 Set tokenizer as processing_class for automatic checkpoint saving")

    # Provide processor to trainer and datasets for HF-first pipeline
    trainer.set_processor(processor)
    logger.info("✅ Set HuggingFace processor on trainer and datasets")

    return trainer


def main():
    from src_new.config.config import load_config, set_global_log_level

    args = parse_args()
    set_global_log_level(args.log_level)
    logger = get_logger()

    # Load configuration
    config = load_config(f"configs/{args.config}.yaml")

    # Seed data-related RNGs without forcing deterministic backends (keeps efficiency)
    try:
        seed_everything(
            getattr(config, "seed", 17), deterministic=False, set_hf_seed=True
        )
        logger.info(f"🔧 Seeded RNGs with seed={getattr(config, 'seed', 17)}")
    except Exception as e:
        logger.warning(f"⚠️ Could not seed RNGs: {e}")

    if args.print_config:
        from pprint import pformat

        logger.info("Loaded config:\n" + pformat(config))
        return

    # Create training arguments (DeepSpeed-aware)
    training_args = create_training_arguments_with_deepspeed(
        config, max_steps=args.max_steps
    )

    # Create trainer and optionally run tests
    trainer = create_trainer_with_new_architecture(training_args, config)

    if args.test_trainer:
        logger.info("Trainer creation successful - exiting due to --test-trainer")
        return

    # Begin training
    trainer.train()


if __name__ == "__main__":
    main()
