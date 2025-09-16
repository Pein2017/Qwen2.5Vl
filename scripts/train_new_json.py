#!/usr/bin/env python3
"""
JSON-mode Training Script (src_new_json) - Using BBUTrainer

This script mirrors scripts/train_new.py but is wired to src_new_json/* modules
that implement the pure-JSON geometry pipeline.

Usage:
    python scripts/train_new_json.py --config phase_1/standard --log_level INFO --max_steps 3
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import TYPE_CHECKING

import torch

from src_new_json.utils.seeding import seed_everything
from src_new_json.utils.rank_aware_logging import initialize_logging_from_env


# Initialize logging early
initialize_logging_from_env()

# Apply Qwen2.5-VL compatibility patches
from src_new_json.models.patches import apply_comprehensive_qwen25_fixes

apply_comprehensive_qwen25_fixes()


if TYPE_CHECKING:
    from transformers.training_args import TrainingArguments
    from src_new_json.config.config import Config


def get_logger():
    from src_new_json.utils.logger_factory import get_training_logger

    return get_training_logger("train_new_json")


def parse_args():
    parser = argparse.ArgumentParser(description="src_new_json Training Script")

    parser.add_argument("--config", required=True, help="Config name or path (e.g., 'phase_1/standard')")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate config only and exit"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print merged config and exit"
    )
    parser.add_argument(
        "--test-trainer", action="store_true", help="Test trainer creation and exit"
    )
    parser.add_argument(
        "--log_level",
        required=True,
        help="Logging level: INFO | DEBUG | WARNING | ERROR | CRITICAL",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of training steps (for quick sanity runs)",
    )
    return parser.parse_args()


def create_training_arguments_with_deepspeed(config: "Config", max_steps=None):
    from transformers.training_args import TrainingArguments

    logger = get_logger()

    deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    deepspeed_config = (
        os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json") if deepspeed_enabled else None
    )

    training_args = TrainingArguments(
        output_dir=config.run_output_dir,
        run_name=config.run_name,
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
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        fp16=config.fp16,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy='no',
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_safetensors=True,
        load_best_model_at_end=False,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        logging_steps=config.logging_steps,
        logging_dir=config.tensorboard_dir,
        report_to=config.report_to,
        disable_tqdm=config.disable_tqdm,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.pin_memory,
        dataloader_prefetch_factor=(
            config.prefetch_factor if config.dataloader_num_workers > 0 else None
        ),
        dataloader_persistent_workers=(
            True if config.dataloader_num_workers > 0 else False
        ),
        remove_unused_columns=config.remove_unused_columns,
        dataloader_drop_last=True,
        deepspeed=deepspeed_config if deepspeed_enabled else None,
        seed=int(config.seed),
        data_seed=int(config.seed),
    )

    logger.info(
        "🚀 Fast checkpoint mode enabled - inference-ready checkpoints only (json mode)"
    )
    return training_args


def create_trainer_with_new_architecture(training_args: "TrainingArguments", config: "Config"):
    from transformers import AutoTokenizer

    logger = get_logger()

    # Data/collator/model components (JSON mode)
    from src_new_json.data.collator import create_data_collator
    from src_new_json.data.dataset import Dataset
    from src_new_json.data.teacher_pool import TeacherPoolManager
    from src_new_json.models.wrapper import DetectionModel

    # Optional phase freeze manager
    try:
        from src_new_json.training.phase_freeze_manager import PhaseFreezeManager
    except Exception:
        PhaseFreezeManager = None  # type: ignore

    logger.info(f"Loading tokenizer and image processor from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Image processor
    from transformers import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if int(config.max_pixels) > 0:
        if not hasattr(image_processor, "max_pixels"):
            raise ValueError("Qwen2VLImageProcessor missing 'max_pixels' attribute")
        image_processor.max_pixels = int(config.max_pixels)
        logger.info(f"🔧 Set image_processor.max_pixels={image_processor.max_pixels}")

    teacher_pool_manager = None
    try:
        # Check if dynamic pairing is enabled
        dynamic_pairing_enabled = config.dynamic_pairing_enabled
        teacher_pool_file = config.teacher_pool_file
        
        if teacher_pool_file and not dynamic_pairing_enabled:
            # Traditional teacher pool approach
            teacher_pool_manager = TeacherPoolManager(
                teacher_pool_file=teacher_pool_file, config=config
            )
            logger.info("✅ Teacher pool manager loaded (traditional teacher-student training)")
        elif dynamic_pairing_enabled:
            # Dynamic pairing approach (teacher_pool_file optional)
            teacher_pool_manager = TeacherPoolManager(
                teacher_pool_file=teacher_pool_file, config=config  # Can be None
            )
            if teacher_pool_file:
                logger.info("✅ Teacher pool manager loaded (hybrid: dynamic pairing + teacher pool)")
            else:
                logger.info("✅ Teacher pool manager initialized for dynamic pairing only")
        else:
            logger.info("ℹ️ No teacher_pool_file and dynamic_pairing_enabled=False; training without teacher examples")
    except Exception as e:
        logger.warning(f"⚠️ Teacher pool manager unavailable ({e}); proceeding with dynamic pairing only")

    logger.info("Creating datasets (JSON mode)...")
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

    if getattr(val_dataset, "augmentation_pipeline", None) is not None:
        val_dataset.augmentation_pipeline = None
        logger.info("🔧 Validation: augmentation disabled by default")

    data_collator = create_data_collator(
        collator_type=config.collator_type, tokenizer=tokenizer, config=config
    )

    # Load model
    from transformers import Qwen2_5_VLForConditionalGeneration

    dtype_str = str(config.torch_dtype).lower()
    if dtype_str in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif dtype_str in {"fp16", "float16"}:
        torch_dtype = torch.float16
    elif dtype_str in {"fp32", "float32"}:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Invalid torch_dtype in config: {config.torch_dtype}")

    loading_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and not hasattr(config, "deepspeed") and torch.cuda.device_count() == 1:
        loading_kwargs["device_map"] = "auto"
        logger.info("🚀 Using direct GPU loading for faster initialization")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path, **loading_kwargs
    )

    # Tokenizer/model alignment
    try:
        if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        bos_id = tokenizer.pad_token_id
        cfg = base_model.config
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.bos_token_id = bos_id
        if hasattr(base_model, "generation_config") and (base_model.generation_config is not None):
            base_model.generation_config.pad_token_id = cfg.pad_token_id
            base_model.generation_config.eos_token_id = cfg.eos_token_id
            base_model.generation_config.bos_token_id = cfg.bos_token_id
    except Exception:
        pass

    # Wrap
    model = DetectionModel(
        base_model=base_model,
        config=config,
        tokenizer=tokenizer,
    )

    # Phase freeze
    if PhaseFreezeManager is not None:
        try:
            pfm = PhaseFreezeManager()
            phase = str(getattr(config, "phase_name", "off") or "off").lower()
            if phase not in ("off", "phase_1", "phase_2", "phase_3"):
                raise ValueError(
                    f"Invalid phase_name: {phase}. Expected one of: off, phase_1, phase_2, phase_3"
                )
            if phase != "off":
                summary = pfm.apply_phase(
                    model,
                    tokenizer,
                    phase=phase,
                    llm_top_k_block=getattr(config, "llm_top_k_block", None),
                    vision_top_k_block=getattr(config, "vision_top_k_block", None),
                    freeze_patch_embed=getattr(config, "freeze_patch_embed", None),
                    trainable_token_strings=getattr(config, "trainable_token_strings", None),
                )
                logger.info(
                    f"✅ PhaseFreeze applied: phase={summary.phase}, trainable≈{summary.num_trainable_params}"
                )
            else:
                logger.info("PhaseFreeze is off; proceeding with default trainable set")
        except Exception as e:
            logger.warning(
                f"⚠️ PhaseFreezeManager failed, continuing without staged freezes: {e}"
            )

    # Create directories
    from pathlib import Path

    Path(config.run_output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_file_dir).mkdir(parents=True, exist_ok=True)

    # Trainer
    BBUTrainer = __import__("src_new_json.training", fromlist=["BBUTrainer"]).BBUTrainer
    trainer = BBUTrainer(
        model=model,
        processing_class=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Pre-create optimizer
    try:
        trainer.create_optimizer()
        logger.info("✅ Pre-created optimizer before staged freezing")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-create optimizer: {e}")

    # Register epoch callback (augmentation schedule + dynamic pairing)
    try:
        from src_new_json.training.callbacks import AugmentationScheduleCallback
        trainer.add_callback(AugmentationScheduleCallback())
        logger.info("✅ Registered AugmentationScheduleCallback")
    except Exception as e:
        logger.warning(f"⚠️ Could not register AugmentationScheduleCallback: {e}")

    # Processor setup for checkpoint saving
    from transformers import Qwen2VLProcessor, Qwen2VLVideoProcessor

    proc_ckpt = Qwen2VLProcessor.from_pretrained(config.model_path, trust_remote_code=True)
    video_processor = proc_ckpt.video_processor if hasattr(proc_ckpt, "video_processor") and proc_ckpt.video_processor is not None else Qwen2VLVideoProcessor()
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=(tokenizer.chat_template if hasattr(tokenizer, "chat_template") else None),
    )

    trainer.processing_class = tokenizer
    trainer.set_processor(processor)

    return trainer


def main():
    from src_new_json.config.config import load_config, set_global_log_level

    args = parse_args()
    set_global_log_level(args.log_level)
    logger = get_logger()

    # Load configuration (supports base + override)
    config_path = args.config if args.config.endswith(".yaml") else f"configs/{args.config}.yaml"
    config = load_config(config_path)

    # Seed RNGs
    try:
        seed_everything(config.seed, deterministic=False, set_hf_seed=True)
        logger.info(f"🔧 Seeded RNGs with seed={config.seed}")
    except Exception as e:
        logger.warning(f"⚠️ Could not seed RNGs: {e}")

    if args.print_config:
        from pprint import pformat

        logger.info("Loaded config:\n" + pformat(config))
        return

    # Create training arguments
    training_args = create_training_arguments_with_deepspeed(
        config, max_steps=args.max_steps
    )
    try:
        setattr(training_args, "original_config_path", config_path)
    except Exception:
        pass

    # Create trainer
    trainer = create_trainer_with_new_architecture(training_args, config)

    if args.test_trainer:
        logger.info("Trainer creation successful - exiting due to --test-trainer")
        return

    # Train
    trainer.train()

    # Clean distributed teardown to avoid TCPStore/NCCL heartbeat noise on shutdown
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is deprecated.*")
    main()
