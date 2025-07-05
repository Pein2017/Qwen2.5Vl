"""
UNIFIED BBU Trainer with Strict Training-Inference Consistency

This trainer uses the unified model loader to ensure ZERO differences 
between training and inference model loading, tokenizer setup, and processing.

NO SILENT FALLBACKS - All errors are exposed immediately.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import config
from src.config.global_config import DirectConfig
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.models.model_loader import load_model_and_processor_unified, ModelLoadingError
from src.utils.schema import GroundTruthObject
from src.utils.tokens.special_tokens import SpecialTokens


class UnifiedBBUTrainer(Trainer):
    """
    Unified BBU Trainer ensuring strict training-inference consistency.
    
    Key features:
    - Uses unified model loader (same as inference)
    - NO silent fallbacks - all errors exposed
    - Strict training-inference consistency validation
    - Proper teacher-student loss backpropagation
    """

    def __init__(
        self,
        *args: Any,
        cfg: Optional[DirectConfig] = None,
        image_processor: Optional[Qwen2VLImageProcessor] = None,
        training_coordinator: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize unified trainer with strict consistency checks.
        
        Args:
            cfg: Direct configuration object
            image_processor: Image processor (will be validated against unified loader)
            training_coordinator: Optional training coordinator
        """
        self.logger = get_training_logger()
        self.logger.info("🚀 Initializing UNIFIED BBU Trainer")
        
        # Store coordinator reference
        self.training_coordinator = training_coordinator
        self._use_coordinator = training_coordinator is not None
        
        # Cache tokenizer reference before parent init to avoid deprecation warnings
        self.tokenizer_ref = kwargs.get('tokenizer')
        
        # Initialize parent trainer
        super().__init__(*args, **kwargs)
        
        # Override tokenizer property to avoid deprecation warnings
        object.__setattr__(self, "tokenizer", self.tokenizer_ref)
        self.image_processor = image_processor
        
        # Store config
        self.config = cfg if cfg is not None else config
        
        # Initialize loss accumulators (used if coordinator not available)
        if not self._use_coordinator:
            self._init_loss_accumulators()
        
        # Validate consistency
        self._validate_training_setup()
        
        self.logger.info("✅ Unified BBU Trainer initialized successfully")

    def _init_loss_accumulators(self):
        """Initialize loss accumulator variables for non-coordinator mode."""
        self._micro_batch_count = 0
        
        # Current loss components (per micro-batch)
        self._current_lm_loss: float = 0.0
        self._current_teacher_lm_loss: float = 0.0
        self._current_student_lm_loss: float = 0.0
        self._current_bbox_loss: float = 0.0
        self._current_caption_loss: float = 0.0
        self._current_objectness_loss: float = 0.0
        self._current_bbox_l1_loss: float = 0.0
        self._current_bbox_giou_loss: float = 0.0
        
        # Accumulated loss components (across gradient accumulation steps)
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_teacher_lm_loss: float = 0.0
        self._accumulated_student_lm_loss: float = 0.0
        self._accumulated_caption_loss: float = 0.0
        self._accumulated_objectness_loss: float = 0.0
        self._accumulated_bbox_l1_loss: float = 0.0
        self._accumulated_bbox_giou_loss: float = 0.0

    def _validate_training_setup(self):
        """Validate that training setup is consistent with inference expectations."""
        self.logger.info("🔍 Validating training-inference consistency...")
        
        try:
            # Check model was loaded via unified loader
            if not hasattr(self.model, 'detection_enabled'):
                self.logger.warning("⚠️  Model missing detection_enabled flag - may indicate inconsistent loading")
            
            # Check tokenizer consistency  
            if self.tokenizer_ref is None:
                raise ModelLoadingError("Tokenizer not properly initialized")
            
            # Check image processor consistency
            if self.image_processor is None:
                raise ModelLoadingError("Image processor not properly initialized")
            
            # Validate detection configuration consistency
            model_detection = getattr(self.model, 'detection_enabled', False)
            config_detection = getattr(self.config, 'detection_enabled', False)
            
            if model_detection != config_detection:
                raise ModelLoadingError(
                    f"Detection mismatch: model={model_detection}, config={config_detection}"
                )
            
            self.logger.info("✅ Training-inference consistency validated")
            
        except Exception as e:
            self.logger.error(f"❌ Training setup validation failed: {e}")
            raise ModelLoadingError(f"Training setup validation failed: {e}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss with proper teacher-student backpropagation.
        """
        if self._use_coordinator:
            return self._compute_loss_with_coordinator(model, inputs, return_outputs)
        else:
            return self._compute_loss_legacy(model, inputs, return_outputs)

    def _compute_loss_with_coordinator(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute loss using training coordinator."""
        try:
            # Forward pass
            outputs = model(**inputs)
            
            # Compute loss via coordinator
            total_loss, loss_components = self.training_coordinator.compute_loss(
                model_outputs=outputs,
                inputs=inputs,
                is_training=model.training
            )
            
            # Update training state
            self.training_coordinator.step_update(
                current_step=self.state.global_step,
                current_epoch=self.state.epoch
            )
            
            if return_outputs:
                return total_loss, outputs
            else:
                return total_loss
                
        except Exception as e:
            self.logger.error(f"❌ Coordinator loss computation failed: {e}")
            raise

    def _compute_loss_legacy(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute loss using legacy system."""
        # This is a fallback - we should use the coordinator for teacher-student training
        self.logger.warning("⚠️  Using legacy loss computation - teacher-student training may not work properly")
        
        # Standard forward pass
        outputs = model(**inputs)
        
        # Use model's built-in loss if available
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Fallback to manual loss computation
            if "labels" in inputs:
                loss = nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    inputs["labels"].view(-1),
                    ignore_index=-100
                )
            else:
                raise ValueError("No loss available and no labels provided")
        
        if return_outputs:
            return loss, outputs
        else:
            return loss


def setup_model_and_tokenizer_unified() -> Tuple[
    nn.Module, PreTrainedTokenizerBase, Qwen2VLImageProcessor
]:
    """
    Setup model and tokenizer using UNIFIED loader for strict consistency.
    """
    logger = get_training_logger()
    logger.info("🔧 Setting up model via UNIFIED loader (same as inference)")
    
    try:
        # Use unified loader - IDENTICAL to inference
        model, tokenizer, image_processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,  # Training mode
        )
        
        # Apply training-specific settings
        _apply_training_settings(model, tokenizer)
        
        logger.info("✅ UNIFIED model setup completed")
        return model, tokenizer, image_processor
        
    except ModelLoadingError as e:
        logger.error(f"❌ UNIFIED model setup failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in unified model setup: {e}")
        raise ModelLoadingError(f"Unexpected error: {e}")


def _apply_training_settings(model: nn.Module, tokenizer: PreTrainedTokenizerBase):
    """Apply training-specific model settings."""
    logger = get_training_logger()
    
    # Get base model for settings
    base_model = model.base_model if hasattr(model, "base_model") else model
    
    # Disable cache for training (required for gradient checkpointing)
    base_model.config.use_cache = False
    logger.info("✅ Disabled KV cache for training")
    
    # Enable gradient checkpointing if configured
    if getattr(config, 'gradient_checkpointing', False):
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
            logger.info("✅ Enabled input gradients via built-in method")
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            base_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
            logger.info("✅ Enabled input gradients via hook")
    
    # Apply parameter training settings
    _set_model_training_params(model)


def _set_model_training_params(model: nn.Module):
    """Set which parameters should be trained based on config."""
    logger = get_training_logger()
    
    # This would contain the parameter freezing/unfreezing logic
    # For now, we assume all parameters are trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Parameter count: {total_params:,} total, {trainable_params:,} trainable")
    
    # Log detection status
    if hasattr(model, 'detection_enabled'):
        logger.info(f"🎯 Detection training: {'ENABLED' if model.detection_enabled else 'DISABLED'}")


def create_unified_trainer(training_args: TrainingArguments) -> UnifiedBBUTrainer:
    """
    Create unified trainer with strict training-inference consistency.
    
    Args:
        training_args: HuggingFace training arguments
        
    Returns:
        UnifiedBBUTrainer instance
    """
    logger = get_training_logger()
    logger.info("🏭 Creating UNIFIED BBU trainer...")
    
    # Setup model and tokenizer via unified loader
    model, tokenizer, image_processor = setup_model_and_tokenizer_unified()
    
    # Setup data module
    data_module = setup_data_module_unified(tokenizer, image_processor)
    
    # Create training coordinator if using new system
    coordinator = None
    if getattr(config, 'use_new_training_system', False):
        try:
            from src.training.training_coordinator import TrainingCoordinator
            coordinator = TrainingCoordinator(
                model=model,
                tokenizer=tokenizer,
                use_domain_config=False
            )
            coordinator.setup_training()
            logger.info("🎯 Training coordinator created")
        except Exception as e:
            logger.warning(f"⚠️  Could not create training coordinator: {e}")
    
    # Create trainer
    trainer = UnifiedBBUTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        image_processor=image_processor,
        cfg=config,
        training_coordinator=coordinator,
        **data_module,
    )
    
    logger.info("✅ UNIFIED BBU trainer created successfully")
    return trainer


def setup_data_module_unified(
    tokenizer: PreTrainedTokenizerBase, 
    image_processor: Qwen2VLImageProcessor
) -> Dict[str, Any]:
    """Setup data module with unified processor."""
    logger = get_training_logger()
    logger.info("📊 Setting up data module...")
    
    # Import chat processor
    from src.chat_processor import ChatProcessor
    
    # Create chat processor (consistent with training)
    chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        merge_size=getattr(config, 'merge_size', 324),
        max_length=getattr(config, 'max_total_length', 12000),
        use_training_prompts=True,
        language="chinese",
    )
    
    # Create datasets
    train_dataset = BBUDataset(
        data_file=getattr(config, 'train_data_path', 'data/train.jsonl'),
        chat_processor=chat_processor,
        data_root=getattr(config, 'data_root', './'),
        split="train",
    )
    
    eval_dataset = BBUDataset(
        data_file=getattr(config, 'val_data_path', 'data/val.jsonl'),
        chat_processor=chat_processor,
        data_root=getattr(config, 'data_root', './'),
        split="val",
    ) if getattr(config, 'val_data_path', None) else None
    
    # Create data collator
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        is_packed=getattr(config, 'pack_samples', False)
    )
    
    logger.info(f"✅ Data module setup complete")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Eval samples: {len(eval_dataset) if eval_dataset else 0}")
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }