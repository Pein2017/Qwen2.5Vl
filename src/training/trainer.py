"""
Unified BBU Trainer following the official qwen-vl-finetune approach.

This implementation supports both ModelWrapper and direct loading approaches,
closely matching the official train_qwen.py to ensure compatibility.
"""

from typing import Any, Dict, List, Optional, Tuple

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
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches
from src.training.callbacks import DetectionLossLoggingCallback


class BBUTrainer(Trainer):
    """
    Custom trainer that integrates object detection loss when configured.

    Extends the standard Transformers Trainer to add object detection capabilities
    while maintaining clean separation of concerns.
    """

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.logger = get_training_logger()

        # Initialize detection losses storage for callback integration
        if not hasattr(self.state, "detection_losses"):
            self.state.detection_losses = {}

        # Initialize training monitor if enabled
        self.monitor = None
        if getattr(config, "enable_monitoring", False):
            self._init_monitor()

        # Initialize object detection loss if configured
        self.detection_loss = None
        if hasattr(config, "loss_type") and config.loss_type == "object_detection":
            self._init_detection_loss()

    def _init_monitor(self):
        """Initialize training monitor for prediction and GT logging."""
        from src.training.monitor import create_training_monitor

        monitor_log_dir = getattr(
            self.config, "monitor_log_dir", "logs/training_monitor"
        )
        save_predictions = getattr(self.config, "save_predictions", True)
        save_token_analysis = getattr(self.config, "save_token_analysis", True)
        save_raw_text = getattr(self.config, "save_raw_text", False)

        self.monitor = create_training_monitor(
            log_dir=monitor_log_dir,
            save_predictions=save_predictions,
            save_token_analysis=save_token_analysis,
            save_raw_text=save_raw_text,
        )

        self.logger.info("🔍 Training monitor initialized in BBUTrainer")
        self.logger.info(f"   Monitor log directory: {monitor_log_dir}")
        self.logger.info(f"   Save predictions: {save_predictions}")
        self.logger.info(f"   Save token analysis: {save_token_analysis}")
        self.logger.info(f"   Save raw text: {save_raw_text}")

    def _init_detection_loss(self):
        """Initialize object detection loss with config parameters."""
        from src.losses import ObjectDetectionLoss

        # EXPLICIT: All parameters must be provided in config - no defaults
        self.detection_loss = ObjectDetectionLoss(
            bbox_weight=self.config.bbox_weight,
            giou_weight=self.config.giou_weight,
            class_weight=self.config.class_weight,
            max_generation_length=self.config.max_generation_length,
            hungarian_matching=self.config.hungarian_matching,
            enable_monitoring=getattr(self.config, "enable_monitoring", False),
            monitor=self.monitor,
        )

        self.logger.info("🎯 Object detection loss initialized in BBUTrainer")
        self.logger.info(f"   bbox_weight: {self.config.bbox_weight}")
        self.logger.info(f"   giou_weight: {self.config.giou_weight}")
        self.logger.info(f"   class_weight: {self.config.class_weight}")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with optional object detection integration.

        This method:
        1. Computes standard language modeling loss via model.forward()
        2. Optionally adds object detection loss if configured
        3. Returns combined loss maintaining clean separation
        """
        # CRITICAL FIX: Use new input preparation for forward pass
        from src.utils import prepare_inputs_for_forward

        # Prepare inputs for forward pass (handles shape validation and filtering)
        model_inputs = prepare_inputs_for_forward(inputs)

        # Standard forward pass to get LM loss
        if self.label_smoother is not None and "labels" in model_inputs:
            labels = model_inputs.pop("labels")
        else:
            labels = None

        # Forward pass through model with prepared inputs
        outputs = model(**model_inputs)

        # Extract standard LM loss
        if labels is not None:
            # Put labels back for potential detection loss computation
            model_inputs["labels"] = labels

            # Compute LM loss using label smoother if available
            if self.label_smoother is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, "_get_name"):
                    model_name = unwrapped_model._get_name()
                else:
                    model_name = unwrapped_model.__class__.__name__

                # Use appropriate loss computation based on model type
                if "CausalLM" in model_name:
                    lm_loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    lm_loss = self.label_smoother(outputs, labels)
            else:
                lm_loss = (
                    outputs.loss
                    if hasattr(outputs, "loss") and outputs.loss is not None
                    else outputs[0]
                )
        else:
            # No labels provided, use model's internal loss
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(model_inputs.keys())}."
                )
            lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Initialize total loss with LM loss
        total_loss = lm_loss

        # Add object detection loss if configured
        if self.detection_loss is not None:
            # Extract input/GT split for generation-based detection loss
            input_parts, ground_truth_texts = self._extract_input_gt_split(inputs)

            if not input_parts or not ground_truth_texts:
                raise ValueError(
                    "Failed to extract input/GT split for detection loss computation. "
                    "Check that inputs contain valid 'input_ids' and 'labels'."
                )

            # Compute detection loss using generation - pass original inputs with custom parameters
            detection_loss_value = self.detection_loss.compute(
                model=model,
                tokenizer=self.processing_class,
                inputs=inputs,  # Pass original inputs with image_counts_per_sample for loss computation
                input_parts=input_parts,
                ground_truth_texts=ground_truth_texts,
                batch_idx=getattr(
                    self.state, "global_step", 0
                ),  # Pass batch index for monitoring
            )

            # Add to total loss
            total_loss = total_loss + detection_loss_value

            # Store detection loss components in trainer state for callback integration
            current_step = getattr(self.state, "global_step", 0)
            if not hasattr(self.state, "detection_losses"):
                self.state.detection_losses = {}

            # Store loss components for the DetectionLossLoggingCallback
            self.state.detection_losses[current_step] = {
                "total_loss": detection_loss_value.item(),
                "lm_loss": lm_loss.item(),
            }

            # Try to get detailed loss components from the detection loss object
            if hasattr(self.detection_loss, "last_loss_components"):
                loss_components = self.detection_loss.last_loss_components
                if loss_components:
                    self.state.detection_losses[current_step].update(loss_components)

            # Log loss components for monitoring
            if self.state.global_step % 1 == 0:  # Log every 10 steps
                self.logger.info(
                    f"Step {self.state.global_step}: LM Loss: {lm_loss.item():.6f}, Detection Loss: {detection_loss_value.item():.6f}"
                )

        # Handle token averaging if needed
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            total_loss *= self.accelerator.num_processes

        return (total_loss, outputs) if return_outputs else total_loss

    def on_train_end(self, args, state, control, **kwargs):
        """Finalize monitoring session when training ends."""
        if self.monitor is not None:
            self.monitor.finalize_session()
        super().on_train_end(args, state, control, **kwargs)

    def _extract_input_gt_split(
        self, inputs: Dict
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Extract input/GT split from batch for object detection loss.

        Returns:
            input_parts: List of input_ids tensors (everything before final assistant response)
            ground_truth_texts: List of ground truth text strings (final assistant responses)
        """
        if "input_ids" not in inputs or "labels" not in inputs:
            return [], []

        input_ids = inputs["input_ids"]  # [batch_size, seq_len]
        labels = inputs["labels"]  # [batch_size, seq_len]

        batch_size = input_ids.size(0)
        input_parts = []
        ground_truth_texts = []

        total_input_tokens = 0
        total_gt_tokens = 0
        valid_samples = 0

        for i in range(batch_size):
            sample_input_ids = input_ids[i]  # [seq_len]
            sample_labels = labels[i]  # [seq_len]

            # Find where real labels start (not -100)
            # This indicates the start of the final assistant response
            real_label_mask = sample_labels != -100
            if not real_label_mask.any():
                # No real labels in this sample, skip
                self.logger.warning(f"   Sample {i}: ⚠️ No real labels found, skipping")
                continue

            # Find the first position where real labels start
            gt_start_idx = real_label_mask.nonzero(as_tuple=True)[0][0].item()

            # Split input_ids: everything before GT start is input
            input_part = sample_input_ids[:gt_start_idx]
            gt_part = sample_input_ids[gt_start_idx:]

            # Calculate token lengths
            input_token_length = input_part.size(0)
            gt_token_length = gt_part.size(0)

            # Decode the GT part to text
            gt_text = self.processing_class.decode(gt_part, skip_special_tokens=True)

            # Log GT text preview
            if gt_text:
                preview = gt_text[:100] + "..." if len(gt_text) > 100 else gt_text
                self.logger.debug(f"      GT text preview: '{preview}'")
            else:
                self.logger.warning(f"      ⚠️ Empty GT text")

            input_parts.append(input_part)
            ground_truth_texts.append(gt_text)

            # Accumulate statistics
            total_input_tokens += input_token_length
            total_gt_tokens += gt_token_length
            valid_samples += 1

        return input_parts, ground_truth_texts


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
    # EXPLICIT: No defaults - config must specify use_model_wrapper
    if not hasattr(config, "use_model_wrapper"):
        raise ValueError(
            "Config must specify 'use_model_wrapper' - no defaults allowed"
        )

    if config.use_model_wrapper:
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

    # EXPLICIT: No defaults - config must specify these values
    if not hasattr(config, "max_total_length"):
        raise ValueError("Config must specify 'max_total_length' - no defaults allowed")
    if not hasattr(config, "collator_type"):
        raise ValueError("Config must specify 'collator_type' - no defaults allowed")

    # Setup data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        max_total_length=config.max_total_length,
        collator_type=config.collator_type,
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
) -> BBUTrainer:
    """
    Create BBUTrainer with object detection loss integration.
    This matches the trainer creation in train_qwen.py exactly but uses BBUTrainer.
    """
    logger = get_training_logger()

    # EXPLICIT: No defaults - config must specify use_model_wrapper
    if not hasattr(config, "use_model_wrapper"):
        raise ValueError(
            "Config must specify 'use_model_wrapper' - no defaults allowed"
        )

    approach = "ModelWrapper" if config.use_model_wrapper else "Direct Loading"
    logger.info(f"🔧 Creating BBUTrainer using {approach}...")

    # EXPLICIT: Enable flash attention optimization based on attn_implementation
    if (
        hasattr(config, "attn_implementation")
        and config.attn_implementation == "flash_attention_2"
    ):
        # replace_qwen2_vl_attention_class()
        logger.info(
            "✅DDDDDBUG not using Flash attention optimization enabled (flash_attention_2)"
        )
    else:
        logger.info(
            f"ℹ️ Using standard attention (attn_implementation: {config.attn_implementation})"
        )

    # Setup model and tokenizer (automatically chooses approach based on config)
    model, tokenizer, image_processor = setup_model_and_tokenizer(config)

    # Setup data module
    data_module = setup_data_module(config, tokenizer, image_processor)

    if training_args is None:
        training_args = TrainingArguments(disable_tqdm=True)

    # Create BBUTrainer with object detection loss integration
    trainer = BBUTrainer(
        config=config,  # Pass config to BBUTrainer
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        args=training_args,
        **data_module,
    )

    logger.info(f"✅ BBUTrainer created successfully using {approach}")

    # Add DetectionLossLoggingCallback by default when object detection loss is enabled
    # or when explicitly requested via config
    enable_detection_logging = trainer.detection_loss is not None

    if enable_detection_logging:
        detection_callback = DetectionLossLoggingCallback()
        trainer.add_callback(detection_callback)
        logger.info("📊 DetectionLossLoggingCallback: ADDED")

    if trainer.detection_loss is not None:
        logger.info("🎯 Object detection loss integration: ENABLED")
    else:
        logger.info("📝 Object detection loss integration: DISABLED")

    return trainer
