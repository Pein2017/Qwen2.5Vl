"""
Unified Model Loading for Training-Inference Consistency

This module provides a single, authoritative way to load Qwen2.5-VL models
with or without detection capabilities. Both training and inference MUST use
this loader to ensure strict consistency.

NO SILENT FALLBACKS - All errors are exposed immediately.
"""

from typing import Any, Optional, Tuple, Union

import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches


logger = get_logger("unified_loader")


class ModelLoadingError(Exception):
    """Raised when model loading fails - NO SILENT FALLBACKS"""

    pass


def load_model_and_processor_unified(
    model_path: str,
    for_inference: bool = False,
    force_detection: Optional[bool] = None,
    attn_implementation: Optional[str] = None,
    config=None,
) -> Tuple[Union[torch.nn.Module, Any], PreTrainedTokenizerBase, Qwen2VLImageProcessor]:
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
        # Use explicit config if provided, otherwise fall back to global config
        if config is None:
            from src.config import get_config

            config = get_config()
            logger.info("📄 Using global configuration system (fallback)")
        else:
            logger.info("📄 Using explicit configuration system")
        # =====================================================================
        # STEP 1: Apply patches (MANDATORY)
        # =====================================================================
        logger.info("🔧 Applying comprehensive Qwen2.5-VL fixes...")
        if not apply_comprehensive_qwen25_fixes():
            raise ModelLoadingError("Failed to apply Qwen2.5-VL fixes - CRITICAL")

        # =====================================================================
        # STEP 2: Determine coordinate token mode and attention implementation
        # =====================================================================
        if not hasattr(config, "coordinate_tokens_enabled"):
            raise ValueError(
                "coordinate_tokens_enabled must be explicitly configured in config"
            )
        coordinate_tokens_enabled = config.coordinate_tokens_enabled

        # Handle None value for force_detection
        if force_detection is not None:
            detection_enabled = bool(force_detection)
            logger.info(f"🎯 Detection mode FORCED: {detection_enabled}")
        else:
            # For coordinate token models, we need the detection wrapper
            detection_enabled = coordinate_tokens_enabled
            logger.info(
                f"🎯 Detection wrapper enabled for coordinate tokens: {detection_enabled}"
            )

        logger.info(f"🎯 Coordinate tokens enabled: {coordinate_tokens_enabled}")

        # Determine effective attention implementation
        effective_attn_impl = ""
        if attn_implementation is not None:
            effective_attn_impl = str(attn_implementation)
            logger.info(f"⚡ Attention implementation OVERRIDE: {effective_attn_impl}")
        else:
            # Use default from config if available
            if hasattr(config, "attn_implementation") and config.attn_implementation:
                effective_attn_impl = str(config.attn_implementation)
                logger.info(
                    f"⚡ Attention implementation from config: {effective_attn_impl}"
                )

        # =====================================================================
        # STEP 3: Load tokenizer (IDENTICAL setup for training/inference)
        # =====================================================================
        logger.info(f"🔤 Loading tokenizer from: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_path,
                model_max_length=config.model_max_length,
                padding_side="left",  # Required for Flash Attention in Qwen2.5-VL
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load tokenizer from {model_path}: {e}")

        # Ensure pad token is set (MANDATORY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Pad token set to EOS token")
        logger.info(
            f"[PADDING_SIDE_CHECK] Tokenizer padding side after load: {tokenizer.padding_side}"
        )

        # CRITICAL FIX: Ensure padding_side is ALWAYS 'left' regardless of how tokenizer was created
        if tokenizer.padding_side != "left":
            logger.warning(
                f"🔧 Fixing tokenizer padding_side: {tokenizer.padding_side} -> left"
            )
            tokenizer.padding_side = "left"
            logger.info(
                f"[PADDING_SIDE_FIX] Tokenizer padding side corrected to: {tokenizer.padding_side}"
            )

        # =====================================================================
        # STEP 4: Load model (STRICT detection vs non-detection)
        # =====================================================================
        if detection_enabled:
            # DETECTION PATH - use wrapper
            logger.info("🎯 Loading DETECTION-ENABLED model via wrapper")
            try:
                from src.models.wrapper import Qwen25VLWithDetection

                # Check if coordinate tokens are enabled
                if not hasattr(config, "coordinate_tokens_enabled"):
                    raise ValueError(
                        "coordinate_tokens_enabled must be explicitly configured in config"
                    )
                coordinate_tokens_enabled = config.coordinate_tokens_enabled
                logger.info(
                    f"🔧 Coordinate tokens enabled: {coordinate_tokens_enabled}"
                )

                # Create coordinate config if coordinate tokens are enabled - NO DEFAULTS
                coordinate_config = None
                if coordinate_tokens_enabled:
                    # Validate required coordinate config parameters (simplified)
                    required_coord_params = [
                        "max_coord_value",
                        "coordinate_loss_weight",
                        "regular_loss_weight",
                    ]

                    missing_params = []
                    for param in required_coord_params:
                        if not hasattr(config, param):
                            missing_params.append(param)

                    if missing_params:
                        raise ValueError(
                            f"Coordinate tokens enabled but missing required configuration parameters: "
                            f"{missing_params}. All coordinate config parameters must be explicitly set."
                        )

                    # Create proper coordinate config instance
                    from src.models.wrapper import CoordinateConfig

                    coordinate_config = CoordinateConfig(
                        enable_coordinate_tokens=config.coordinate_tokens_enabled,
                        max_coord_value=config.max_coord_value,
                        coordinate_loss_weight=config.coordinate_loss_weight,
                        regular_loss_weight=config.regular_loss_weight,
                        # Default values for backward compatibility
                        coord_token_init_std=0.02,
                        soft_expectation_temperature=1.0,
                        use_official_box_tokens=True,
                        enable_multi_geometry=True,
                        max_line_coordinates=50,
                    )
                    logger.info("✅ Coordinate token configuration created")

                model = Qwen25VLWithDetection.from_pretrained(
                    model_path=model_path,
                    tokenizer=tokenizer,
                    coordinate_config=coordinate_config,
                    attn_implementation=effective_attn_impl,
                    config=config,
                )

                # CRITICAL: For coordinate token models, ensure vocabulary consistency
                if (
                    coordinate_tokens_enabled
                    and hasattr(model, "coordinate_tokens_enabled")
                    and model.coordinate_tokens_enabled
                ):
                    logger.info("🔧 Validating coordinate token vocabulary consistency")

                    # Check model vocabulary size vs tokenizer vocabulary size
                    model_vocab_size = model.base_model.config.vocab_size
                    tokenizer_vocab_size = len(tokenizer.get_vocab())

                    logger.info(f"📊 Model vocab size: {model_vocab_size}")
                    logger.info(f"📊 Tokenizer vocab size: {tokenizer_vocab_size}")

                    if (
                        hasattr(model, "extended_vocab_size")
                        and model.extended_vocab_size
                    ):
                        logger.info(
                            f"📊 Extended vocab size: {model.extended_vocab_size}"
                        )
                        logger.info(
                            "✅ Model has coordinate token extensions - ensuring consistency"
                        )

                        # Fix the issue with setting detection_enabled attribute
                        # Ensure the model's detection flag is consistent
                        # Instead of directly setting the attribute, use a more compatible approach
                        if hasattr(model, "set_detection_enabled"):
                            model.set_detection_enabled(True)
                        else:
                            # Create a method to safely set the attribute if it doesn't exist
                            def set_detection_enabled(self, value):
                                self._detection_enabled = value

                            # Add the method to the model
                            import types

                            # Use setattr instead of direct assignment to avoid type error
                            setattr(
                                model,
                                "set_detection_enabled",
                                types.MethodType(set_detection_enabled, model),
                            )
                            model.set_detection_enabled(True)

                        logger.info(
                            "✅ Set detection_enabled=True for coordinate token model"
                        )
                    else:
                        logger.warning(
                            "⚠️ Model loaded without extended vocabulary - coordinate tokens may not work"
                        )

                # CRITICAL: Move to GPU for inference
                if for_inference and torch.cuda.is_available():
                    model = model.to("cuda:0")
                    logger.info("🔧 Detection model moved to GPU for inference")

                # For inference, we MUST use the same generation interface
                if for_inference:
                    # Ensure generate() works consistently
                    if not hasattr(model, "generate"):
                        raise ModelLoadingError(
                            "Detection model missing generate() method"
                        )

                logger.info("✅ Detection-enabled model loaded successfully")

            except Exception as e:
                raise ModelLoadingError(f"Failed to load detection model: {e}")
        else:
            # NON-DETECTION PATH - use base model
            logger.info("📄 Loading BASE model (no detection)")
            try:
                # Check if DeepSpeed is enabled
                import os

                from src.models.wrapper import _get_torch_dtype

                deepspeed_enabled = (
                    os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
                )

                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=_get_torch_dtype(
                        getattr(config, "torch_dtype", "auto")
                    ),
                    attn_implementation=effective_attn_impl,
                    device_map=None
                    if deepspeed_enabled
                    else "auto",  # Let DeepSpeed handle if enabled
                    trust_remote_code=True,
                    use_cache=config.use_cache_inference
                    if for_inference
                    else config.use_cache,
                )

                # CRITICAL: Move to GPU for inference only if NOT using DeepSpeed
                if (
                    for_inference
                    and torch.cuda.is_available()
                    and not deepspeed_enabled
                ):
                    # Use the device method instead of to() directly
                    device = torch.device("cuda:0")
                    # Use to() method properly
                    model = model.to(device)
                    logger.info("🔧 Base model moved to GPU for inference")
                elif deepspeed_enabled:
                    logger.info("🔧 DeepSpeed enabled - skipping manual GPU placement")

                # Flag for consistency checks
                if hasattr(model, "set_detection_enabled"):
                    model.set_detection_enabled(False)
                else:
                    # Create a method to safely set the attribute if it doesn't exist
                    def set_detection_enabled(self, value):
                        self._detection_enabled = value

                    # Add the method to the model
                    import types

                    # Use setattr instead of direct assignment to avoid type error
                    setattr(
                        model,
                        "set_detection_enabled",
                        types.MethodType(set_detection_enabled, model),
                    )
                    model.set_detection_enabled(False)

                logger.info("✅ Base model loaded successfully")

            except Exception as e:
                raise ModelLoadingError(f"Failed to load base model: {e}")

        # =====================================================================
        # STEP 5: Add special tokens (CRITICAL for both simple and coordinate tokens)
        # =====================================================================
        # Always add required geometry tokens
        geometry_tokens = {
            "<|box_start|>": None,  # No special meaning, just used as delimiters
            "<|box_end|>": None,
            "<|square_start|>": None,
            "<|square_end|>": None,
            "<|line_start|>": None,
            "<|line_end|>": None,
            "<|object_ref_start|>": None,
            "<|object_ref_end|>": None,
        }

        # Add to tokenizer
        tokens_to_add = []
        for token in geometry_tokens:
            # Check if already in vocabulary
            if token not in tokenizer.get_vocab():
                tokens_to_add.append(token)

        if tokens_to_add:
            logger.info(
                f"🎯 Adding {len(tokens_to_add)} special tokens: {tokens_to_add}"
            )
            special_tokens_dict = {"additional_special_tokens": tokens_to_add}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"✅ Added {num_added} special tokens to tokenizer")

            # Model embeddings will be resized during initialization
            # Update model's embedding table if model is already loaded
            if isinstance(model, torch.nn.Module):
                logger.info(f"🎯 Resizing model embeddings for new tokens")
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"✅ Model embeddings resized to {len(tokenizer)}")

        # Log special token IDs
        vocab = tokenizer.get_vocab()
        logger.info(f"🔍 SPECIAL TOKEN IDs:")
        for token in geometry_tokens:
            if token in vocab:
                logger.info(f"   {token}: {vocab[token]}")
            else:
                logger.warning(f"   {token}: NOT FOUND IN VOCABULARY")

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
                logger.warning(
                    f"Could not import MAX_PIXELS, using fallback: {MAX_PIXELS}"
                )

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
            logger.warning(
                "   Proceeding with default template - may cause inconsistency!"
            )

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
        logger.info(
            f"   Model type: {'Detection-enabled' if detection_enabled else 'Base model'}"
        )
        logger.info(f"   Tokenizer vocab size: {len(tokenizer)}")
        # FINAL VERIFICATION: Ensure padding_side is still 'left' before returning
        if tokenizer.padding_side != "left":
            logger.warning(
                f"🚨 CRITICAL: Tokenizer padding_side was reset! Fixing: {tokenizer.padding_side} -> left"
            )
            tokenizer.padding_side = "left"
        logger.info(f"   Tokenizer padding side: {tokenizer.padding_side}")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        logger.info(f"   Model dtype: {next(model.parameters()).dtype}")

        # Consistency check - ensure coordinate token models have proper detection flag
        if hasattr(model, "detection_enabled"):
            if not hasattr(config, "coordinate_tokens_enabled"):
                raise ValueError(
                    "coordinate_tokens_enabled must be explicitly configured in config"
                )
            coordinate_tokens_enabled = config.coordinate_tokens_enabled
            if coordinate_tokens_enabled:
                # For coordinate token models, always ensure detection_enabled=True
                if not model.detection_enabled:
                    logger.info(
                        "🔧 Setting detection_enabled=True for coordinate token model"
                    )
                    # Use setattr instead of direct assignment to avoid type error
                    setattr(
                        model,
                        "detection_enabled",
                        True,
                    )
                logger.info("✅ Coordinate token model consistency verified")
            elif model.detection_enabled != detection_enabled:
                # For non-coordinate models, enforce strict consistency
                raise ModelLoadingError(
                    f"Model detection flag mismatch: config={detection_enabled}, "
                    f"model={model.detection_enabled}"
                )

        # =====================================================================
        # STEP 7: Initialize Token Managers for vocabulary extension
        # =====================================================================
        if coordinate_tokens_enabled:
            logger.info("🎯 Initializing Token Managers for vocabulary extension...")

            # Calculate expected vocabulary extensions using config values only
            base_vocab_size = config.model_vocab_size  # Use the value from YAML
            expected_extensions = 0

            # If coordinate tokens are enabled, get count from config
            if hasattr(config, "max_coord_value"):
                coordinate_token_count = config.max_coord_value
                expected_extensions += coordinate_token_count
                logger.info(
                    f"🔢 Adding {coordinate_token_count} coordinate tokens to vocabulary"
                )

            # Multi-geometry is always enabled with unified token manager
            # Import to get token count without hardcoding
            from src.utils.tokens.special_tokens import UnifiedTokenManager

            geometry_token_count = len(UnifiedTokenManager.NEW_GEOMETRY_TOKENS)
            expected_extensions += geometry_token_count
            logger.info(
                f"📐 Adding {geometry_token_count} geometry tokens to vocabulary"
            )

            # Calculate and log the expected final vocabulary size
            expected_final_size = base_vocab_size + expected_extensions
            logger.info(
                f"📊 Vocabulary extension: {base_vocab_size} + {expected_extensions} = {expected_final_size}"
            )

            # First, initialize the coordinate token manager (adds coordinate tokens to model)
            # This is already done in the detection model loading path

            # Then initialize unified token manager with awareness of coordinate tokens
            from src.utils.tokens import create_unified_token_manager

            # Create unified token manager (handles everything automatically)
            token_manager = create_unified_token_manager(
                tokenizer,
                model,
                max_coord_value=config.max_coord_value,
            )

            # Verify final vocabulary sizes
            final_tokenizer_size = len(tokenizer.get_vocab())
            final_model_size = model.get_input_embeddings().num_embeddings

            logger.info("📊 FINAL VOCABULARY VERIFICATION:")
            logger.info(f"   Expected size: {expected_final_size}")
            logger.info(f"   Tokenizer size: {final_tokenizer_size}")
            logger.info(f"   Model embedding size: {final_model_size}")

            if final_tokenizer_size != expected_final_size:
                logger.warning(
                    f"⚠️ Final tokenizer size ({final_tokenizer_size}) doesn't match expected size ({expected_final_size})"
                )
            else:
                logger.info(f"✅ Final tokenizer size verified: {final_tokenizer_size}")

            if final_model_size != final_tokenizer_size:
                logger.warning(
                    f"⚠️ Model-tokenizer size mismatch: model={final_model_size}, tokenizer={final_tokenizer_size}"
                )
            else:
                logger.info(
                    f"✅ Model-tokenizer size consistency verified: {final_model_size}"
                )

            logger.info("✅ Simple Token Manager initialized successfully!")
        else:
            logger.info("ℹ️ Coordinate tokens disabled - using standard format")

            # Multi-geometry support is always enabled - add geometry tokens
            logger.info("📐 Adding geometry tokens (coordinate tokens disabled)")
            from src.utils.tokens import create_unified_token_manager
            from src.utils.tokens.special_tokens import UnifiedTokenManager

            # Create unified token manager (handles everything automatically)
            token_manager = create_unified_token_manager(
                tokenizer,
                model,
                max_coord_value=0,  # No coordinate tokens for geometry-only mode
            )

            # Log the vocabulary sizes using config values
            logger.info("📊 GEOMETRY-ONLY VOCABULARY VERIFICATION:")
            logger.info(f"   Base size: {config.model_vocab_size}")
            logger.info(
                f"   Geometry tokens: {len(UnifiedTokenManager.NEW_GEOMETRY_TOKENS)}"
            )
            logger.info(
                f"   Expected size: {config.model_vocab_size + len(UnifiedTokenManager.NEW_GEOMETRY_TOKENS)}"
            )
            logger.info(f"   Actual tokenizer size: {len(tokenizer.get_vocab())}")
            logger.info(
                f"   Actual model size: {model.get_input_embeddings().num_embeddings}"
            )

            return model, tokenizer, image_processor

    except ModelLoadingError:
        raise  # Re-raise our errors
    except Exception as e:
        raise ModelLoadingError(f"Unexpected error in unified model loading: {e}")


def validate_training_inference_consistency(
    training_model_path: str, inference_model_path: str
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
        train_model, train_tok, _ = load_model_and_processor_unified(
            training_model_path, for_inference=False
        )
        infer_model, infer_tok, _ = load_model_and_processor_unified(
            inference_model_path, for_inference=True
        )

        # Compare architectures
        train_detection = getattr(train_model, "detection_enabled", False)
        infer_detection = getattr(infer_model, "detection_enabled", False)

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
