"""
Unified BBU Trainer following the official qwen-vl-finetune approach.

This implementation supports both ModelWrapper and direct loading approaches,
closely matching the official train_qwen.py to ensure compatibility.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.config.base import Config
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.attention import replace_qwen2_vl_attention_class
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches


def set_model_training_params(config: Config, model):
    """
    Set model training parameters following the official approach.
    This matches the set_model() function in train_qwen.py exactly.
    """
    logger = get_training_logger()

    # Vision encoder training
    if config.tune_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
        logger.info(f"🔧 Vision encoder: TRAINING (lr={config.vision_lr})")
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
        logger.info("🔧 Vision encoder: FROZEN")

    # MLP connector training
    if config.tune_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
        logger.info(f"🔧 MLP connector: TRAINING (lr={config.mlp_lr})")
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
        logger.info("🔧 MLP connector: FROZEN")

    # LLM training
    if config.tune_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        logger.info(f"🔧 LLM: TRAINING (lr={config.llm_lr})")
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        logger.info("🔧 LLM: FROZEN")


def setup_model_and_tokenizer_with_wrapper(
    config: Config,
) -> Tuple[nn.Module, Any, Any]:
    """
    Setup model and tokenizer using BBU ModelWrapper with patches.
    This ensures all mRoPE fixes and other patches are applied.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model and tokenizer with BBU ModelWrapper...")

    # Use BBU ModelWrapper which includes all necessary patches
    from src.models.wrapper import ModelWrapper

    model_wrapper = ModelWrapper(config, logger)
    model, tokenizer, image_processor = model_wrapper.load_all()

    # CRITICAL: Disable cache for training - following official approach
    model.config.use_cache = False

    # Setup gradient checkpointing if enabled - following official approach
    if config.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Set training parameters
    set_model_training_params(config, model)

    # Log trainable parameters - following official approach
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    elif not torch.distributed.is_initialized():
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    logger.info("✅ Model and tokenizer setup completed with BBU ModelWrapper")
    return model, tokenizer, image_processor


def setup_model_and_tokenizer_direct(config: Config) -> Tuple[nn.Module, Any, Any]:
    """
    Setup model and tokenizer with direct loading and patches.
    This matches the model setup in train_qwen.py exactly.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model and tokenizer with direct loading...")

    # Apply comprehensive Qwen2.5-VL fixes FIRST
    logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
    if not apply_comprehensive_qwen25_fixes():
        raise RuntimeError("Failed to apply Qwen2.5-VL fixes")

    # Load model - following official approach exactly
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        cache_dir=config.cache_dir,
        attn_implementation=config.attn_implementation,
        torch_dtype=(torch.bfloat16 if config.bf16 else None),
    )

    # Verify all patches
    logger.info("🔍 Verifying all patches...")
    if not verify_qwen25_patches():
        raise RuntimeError("Patch verification failed")

    # Load processor and tokenizer - following official approach
    processor = AutoProcessor.from_pretrained(config.model_path)
    image_processor = processor.image_processor

    # CRITICAL FIX: Use pixel constraints from data_conversion/vision_process.py
    try:
        from data_conversion.vision_process import MAX_PIXELS, MIN_PIXELS

        # Apply the exact same pixel constraints used during data conversion
        image_processor.min_pixels = MIN_PIXELS  # 4 * 28 * 28 = 3136
        image_processor.max_pixels = MAX_PIXELS  # 128 * 28 * 28 = 100352

        # Also set size constraints if the processor supports them
        if hasattr(image_processor, "size"):
            if isinstance(image_processor.size, dict):
                image_processor.size["min_pixels"] = MIN_PIXELS
                image_processor.size["max_pixels"] = MAX_PIXELS
            else:
                image_processor.size = {
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }

        logger.info(
            f"✅ Image processor configured with data_conversion pixel constraints:"
        )
        logger.info(f"   min_pixels: {image_processor.min_pixels} (4 * 28 * 28)")
        logger.info(f"   max_pixels: {image_processor.max_pixels} (128 * 28 * 28)")

    except ImportError as e:
        logger.error(f"❌ Failed to import from data_conversion/vision_process.py: {e}")
        logger.warning("   Using default image processor pixel constraints")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # CRITICAL: Disable cache for training - following official approach
    model.config.use_cache = False

    # Setup gradient checkpointing if enabled - following official approach
    if config.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Set training parameters
    set_model_training_params(config, model)

    # Log trainable parameters - following official approach
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    elif not torch.distributed.is_initialized():
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    logger.info("✅ Model and tokenizer setup completed with direct loading")
    return model, tokenizer, image_processor


def setup_model_and_tokenizer(config: Config) -> Tuple[nn.Module, Any, Any]:
    """
    Setup model and tokenizer with configurable approach.
    Chooses between ModelWrapper and direct loading based on config.
    """
    # Check if config specifies which approach to use
    use_model_wrapper = getattr(config, "use_model_wrapper", True)

    if use_model_wrapper:
        return setup_model_and_tokenizer_with_wrapper(config)
    else:
        return setup_model_and_tokenizer_direct(config)


def setup_data_module(config: Config, tokenizer, image_processor) -> Dict[str, Any]:
    """
    Setup data module following the official approach.
    This matches the data setup in train_qwen.py.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up data module...")

    # Setup datasets
    train_dataset = BBUDataset(
        config, tokenizer, image_processor, config.train_data_path
    )
    logger.info(f"✅ Train dataset: {len(train_dataset)} samples")

    eval_dataset = None
    if hasattr(config, "val_data_path") and config.val_data_path:
        eval_dataset = BBUDataset(
            config, tokenizer, image_processor, config.val_data_path
        )
        if len(eval_dataset) > 0:
            logger.info(f"✅ Eval dataset: {len(eval_dataset)} samples")
        else:
            logger.warning("⚠️ Validation dataset is empty")
            eval_dataset = None

    # Setup data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        max_total_length=getattr(config, "max_total_length", config.model_max_length),
        collator_type=getattr(config, "collator_type", "standard"),
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safe model saving following the official approach.
    This matches the safe_save_model_for_hf_trainer in train_qwen.py exactly.
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def create_trainer(
    config: Config, training_args: Optional[TrainingArguments] = None, **kwargs
) -> Trainer:
    """
    Create trainer following the official approach.
    This matches the trainer creation in train_qwen.py exactly.
    """
    logger = get_training_logger()

    # Determine which approach is being used
    use_model_wrapper = getattr(config, "use_model_wrapper", True)
    approach = "ModelWrapper" if use_model_wrapper else "Direct Loading"
    logger.info(f"🔧 Creating unified trainer using {approach}...")

    # Setup flash attention optimization if needed
    if getattr(config, "data_flatten", False):
        replace_qwen2_vl_attention_class()
        logger.info("✅ Flash attention optimization enabled")

    # Setup model and tokenizer (automatically chooses approach based on config)
    model, tokenizer, image_processor = setup_model_and_tokenizer(config)

    # Setup data module
    data_module = setup_data_module(config, tokenizer, image_processor)

    # Create trainer - following official approach exactly
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        args=training_args,
        **data_module,
    )

    logger.info(f"✅ Unified Trainer created successfully using {approach}")
    return trainer
