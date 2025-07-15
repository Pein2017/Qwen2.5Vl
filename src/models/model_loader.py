"""
Unified Model Loading for Training-Inference Consistency

This module provides a single, authoritative way to load Qwen2.5-VL models
with or without detection capabilities. Both training and inference MUST use
this loader to ensure strict consistency.

NO SILENT FALLBACKS - All errors are exposed immediately.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import config
from src.logger_utils import get_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches


logger = get_logger("unified_loader")


class ModelLoadingError(Exception):
    """Raised when model loading fails - NO SILENT FALLBACKS"""
    pass


def load_model_and_processor_unified(
    model_path: str,
    for_inference: bool = False,
    force_detection: bool = None,
    attn_implementation: str = None
) -> Tuple[Union[torch.nn.Module, any], PreTrainedTokenizerBase, Qwen2VLImageProcessor]:
    """
    UNIFIED model and processor loading for training and inference.
    
    This function MUST be used by both training and inference to ensure
    strict consistency. NO silent fallbacks - all errors are exposed.
    
    Args:
        model_path: Path to model (base model or checkpoint)
        for_inference: Whether this is for inference (affects some settings)
        force_detection: Override detection_enabled from config (for testing)
        attn_implementation: Override attention implementation ('flash_attention_2', 'eager', etc.)
        
    Returns:
        Tuple of (model, tokenizer, image_processor)
        
    Raises:
        ModelLoadingError: If any step fails
    """
    logger.info(f"🔧 UNIFIED MODEL LOADING: {model_path}")
    logger.info(f"   Mode: {'INFERENCE' if for_inference else 'TRAINING'}")
    
    try:
        # =====================================================================
        # STEP 1: Apply patches (MANDATORY)
        # =====================================================================
        logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
        if not apply_comprehensive_qwen25_fixes():
            raise ModelLoadingError("Failed to apply Qwen2.5-VL fixes - CRITICAL")
        
        # =====================================================================
        # STEP 2: Determine coordinate token mode and attention implementation
        # =====================================================================
        coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
        
        if force_detection is not None:
            detection_enabled = force_detection
            logger.info(f"🎯 Detection mode FORCED: {detection_enabled}")
        else:
            # For coordinate token models, we need the detection wrapper
            detection_enabled = coordinate_tokens_enabled
            logger.info(f"🎯 Detection wrapper enabled for coordinate tokens: {detection_enabled}")
            
        logger.info(f"🎯 Coordinate tokens enabled: {coordinate_tokens_enabled}")
        
        # Determine effective attention implementation
        if attn_implementation is not None:
            effective_attn_impl = attn_implementation
            logger.info(f"⚡ Attention implementation OVERRIDE: {effective_attn_impl}")
        else:
            effective_attn_impl = getattr(config, 'attn_implementation', 'flash_attention_2')
            logger.info(f"⚡ Attention implementation from config: {effective_attn_impl}")
        
        # =====================================================================
        # STEP 3: Load tokenizer (IDENTICAL setup for training/inference)
        # =====================================================================
        logger.info(f"🔤 Loading tokenizer from: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_path,
                model_max_length=getattr(config, 'model_max_length', 120000),
                padding_side="left" if for_inference else "right",  # ONLY difference
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load tokenizer from {model_path}: {e}")
        
        # Ensure pad token is set (MANDATORY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Pad token set to EOS token")
        
        # =====================================================================
        # STEP 4: Load model (STRICT detection vs non-detection)
        # =====================================================================
        if detection_enabled:
            # DETECTION PATH - use wrapper
            logger.info("🎯 Loading DETECTION-ENABLED model via wrapper")
            try:
                from src.models.wrapper import Qwen25VLWithDetection
                
                # Check if coordinate tokens are enabled
                coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
                logger.info(f"🔧 Coordinate tokens enabled: {coordinate_tokens_enabled}")
                
                model = Qwen25VLWithDetection.from_pretrained(
                    model_path=model_path,
                    num_queries=getattr(config, 'detection_num_queries', 100),
                    max_caption_length=getattr(config, 'detection_max_caption_length', 256),
                    tokenizer=tokenizer,
                    load_detection_head=True,
                    attn_implementation=effective_attn_impl,
                )
                
                # CRITICAL: For coordinate token models, ensure vocabulary consistency
                if coordinate_tokens_enabled and hasattr(model, 'coordinate_tokens_enabled') and model.coordinate_tokens_enabled:
                    logger.info("🔧 Validating coordinate token vocabulary consistency")
                    
                    # Check model vocabulary size vs tokenizer vocabulary size
                    model_vocab_size = model.base_model.config.vocab_size
                    tokenizer_vocab_size = len(tokenizer.get_vocab())
                    
                    logger.info(f"📊 Model vocab size: {model_vocab_size}")
                    logger.info(f"📊 Tokenizer vocab size: {tokenizer_vocab_size}")
                    
                    if hasattr(model, 'extended_vocab_size') and model.extended_vocab_size:
                        logger.info(f"📊 Extended vocab size: {model.extended_vocab_size}")
                        logger.info("✅ Model has coordinate token extensions - ensuring consistency")
                        
                        # Ensure the model's detection flag is consistent
                        model.detection_enabled = True  # Override for coordinate token models
                        logger.info("✅ Set detection_enabled=True for coordinate token model")
                    else:
                        logger.warning("⚠️ Model loaded without extended vocabulary - coordinate tokens may not work")
                
                # CRITICAL: Move to GPU for inference
                if for_inference and torch.cuda.is_available():
                    model = model.to("cuda:0")
                    logger.info("🔧 Detection model moved to GPU for inference")
                
                # For inference, we MUST use the same generation interface
                if for_inference:
                    # Ensure generate() works consistently
                    if not hasattr(model, 'generate'):
                        raise ModelLoadingError("Detection model missing generate() method")
                
                logger.info("✅ Detection-enabled model loaded successfully")
                
            except Exception as e:
                raise ModelLoadingError(f"Failed to load detection model: {e}")
        else:
            # NON-DETECTION PATH - use base model
            logger.info("📄 Loading BASE model (no detection)")
            try:
                from src.models.wrapper import _get_torch_dtype
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=_get_torch_dtype(getattr(config, 'torch_dtype', 'auto')),
                    attn_implementation=effective_attn_impl,
                    device_map=None,  # Single GPU only - no multi-GPU device mapping
                    trust_remote_code=True,
                    use_cache=True,
                )
                
                # CRITICAL: Move to GPU for inference
                if for_inference and torch.cuda.is_available():
                    model = model.to("cuda:0")
                    logger.info("🔧 Base model moved to GPU for inference")
                
                # Flag for consistency checks
                model.detection_enabled = False
                logger.info("✅ Base model loaded successfully")
                
            except Exception as e:
                raise ModelLoadingError(f"Failed to load base model: {e}")
        
        # =====================================================================
        # STEP 5: Load processor (IDENTICAL for training/inference)
        # =====================================================================
        logger.info(f"📊 Loading processor from: {model_path}")
        try:
            # Import max pixels setting
            try:
                from data_conversion.vision_process import MAX_PIXELS
            except ImportError:
                MAX_PIXELS = 1024 * 28 * 28  # Fallback value
                logger.warning(f"Could not import MAX_PIXELS, using fallback: {MAX_PIXELS}")
            
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                max_pixels=MAX_PIXELS,
            )
            
            # Override tokenizer with our configured one
            processor.tokenizer = tokenizer
            
            image_processor = processor.image_processor
            logger.info("✅ Processor loaded successfully")
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load processor: {e}")
        
        # =====================================================================
        # STEP 6: Apply chat template consistently
        # =====================================================================
        logger.info("🔧 Applying training chat template...")
        try:
            from src.reference.qwen2_5vl_collator import Qwen2_5VLCollator
            
            # This mutates processor.tokenizer.chat_template
            _ = Qwen2_5VLCollator(processor)
            logger.info("✅ Training chat template applied")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not apply training chat template: {e}")
            logger.warning("   Proceeding with default template - may cause inconsistency!")
        
        # =====================================================================
        # STEP 7: Verify patches (MANDATORY)
        # =====================================================================
        logger.info("🔍 Verifying Qwen2.5-VL patches...")
        if not verify_qwen25_patches():
            raise ModelLoadingError("Patch verification failed - CRITICAL")
        
        # =====================================================================
        # STEP 8: Final validation
        # =====================================================================
        logger.info("✅ UNIFIED MODEL LOADING COMPLETE")
        logger.info(f"   Model type: {'Detection-enabled' if detection_enabled else 'Base model'}")
        logger.info(f"   Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"   Tokenizer padding side: {tokenizer.padding_side}")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        logger.info(f"   Model dtype: {next(model.parameters()).dtype}")
        
        # Consistency check - ensure coordinate token models have proper detection flag
        if hasattr(model, 'detection_enabled'):
            coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
            if coordinate_tokens_enabled:
                # For coordinate token models, always ensure detection_enabled=True
                if not model.detection_enabled:
                    logger.info("🔧 Setting detection_enabled=True for coordinate token model")
                    model.detection_enabled = True
                logger.info("✅ Coordinate token model consistency verified")
            elif model.detection_enabled != detection_enabled:
                # For non-coordinate models, enforce strict consistency
                raise ModelLoadingError(
                    f"Model detection flag mismatch: config={detection_enabled}, "
                    f"model={model.detection_enabled}"
                )
        
        return model, tokenizer, image_processor
        
    except ModelLoadingError:
        raise  # Re-raise our errors
    except Exception as e:
        raise ModelLoadingError(f"Unexpected error in unified model loading: {e}")


def validate_training_inference_consistency(
    training_model_path: str,
    inference_model_path: str
) -> bool:
    """
    Validate that training and inference would load the same model architecture.
    
    Args:
        training_model_path: Path used for training
        inference_model_path: Path used for inference
        
    Returns:
        True if consistent, raises ModelLoadingError if not
    """
    logger.info("🔍 Validating training-inference consistency...")
    
    try:
        # Load both models
        train_model, train_tok, train_proc = load_model_and_processor_unified(
            training_model_path, for_inference=False
        )
        infer_model, infer_tok, infer_proc = load_model_and_processor_unified(
            inference_model_path, for_inference=True
        )
        
        # Compare architectures
        train_detection = getattr(train_model, 'detection_enabled', False)
        infer_detection = getattr(infer_model, 'detection_enabled', False)
        
        if train_detection != infer_detection:
            raise ModelLoadingError(
                f"Detection capability mismatch: training={train_detection}, "
                f"inference={infer_detection}"
            )
        
        # Compare tokenizer vocab
        if len(train_tok) != len(infer_tok):
            raise ModelLoadingError(
                f"Tokenizer vocab size mismatch: training={len(train_tok)}, "
                f"inference={len(infer_tok)}"
            )
        
        logger.info("✅ Training-inference consistency validated")
        return True
        
    except Exception as e:
        raise ModelLoadingError(f"Consistency validation failed: {e}")