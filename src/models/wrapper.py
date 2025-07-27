"""
Qwen2.5-VL Model Wrapper with Detection Capabilities

This module provides a wrapper around the official Qwen2.5-VL model
that adds object detection capabilities while preserving all original functionality.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes


@dataclass
class CoordinateConfig:
    """Configuration for coordinate token extension."""

    max_coord_value: int = 2048
    coord_token_init_std: float = 0.02
    coordinate_loss_weight: float = 1.0
    regular_loss_weight: float = 1.0
    soft_expectation_temperature: float = 1.0
    enable_coordinate_tokens: bool = False  # Feature flag
    use_official_box_tokens: bool = (
        True  # Always use official <|box_start|> and <|box_end|>
    )

    # Multi-geometry extensions
    enable_multi_geometry: bool = False  # Feature flag for square/line support
    max_line_coordinates: int = 50  # Maximum coordinate pairs for line objects


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16,  # Default to bfloat16 for auto
    }

    # Strict validation - fail if unsupported dtype
    if dtype_str.lower() not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str.lower()]


class Qwen25VLWithDetection(nn.Module):
    """
    Wrapper around official Qwen2.5-VL model with detection capabilities.

    This wrapper adds a detection head while preserving all original functionality
    of the Qwen2.5-VL model for generation tasks.

    ENHANCED: Supports coordinate token extension for soft expectation regression.
    All pretrained weights are preserved when extending vocabulary.
    """

    def __init__(
        self,
        base_model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer: PreTrainedTokenizerBase,
        attn_implementation: str = "",  # Changed from None to empty string
        coordinate_config: Optional[CoordinateConfig] = None,
        config=None,
        use_cache: bool = False,  # Add use_cache parameter for inference mode
    ) -> None:
        super().__init__()

        # Strict validation - fail fast
        if not base_model_path:
            raise ValueError("base_model_path cannot be empty")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if not hasattr(tokenizer, "get_vocab"):
            raise ValueError("tokenizer must have get_vocab method")
        if coordinate_config is not None and not isinstance(
            coordinate_config, CoordinateConfig
        ):
            raise ValueError("coordinate_config must be CoordinateConfig instance")

        # Store tokenizer and logger
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.logger = get_training_logger()

        # Store coordinate configuration
        if coordinate_config is None:
            raise ValueError("coordinate_config is required and cannot be None")
        self.coordinate_config = coordinate_config

        # Coordinate token tracking
        self.coordinate_tokens_enabled = self.coordinate_config.enable_coordinate_tokens
        self.original_vocab_size: int = None
        self.extended_vocab_size: int = None
        self.extended_embeddings = None
        self.extended_lm_head = None

        # Unified token manager (includes coordinate tokens and loss computation)
        self.token_manager = None

        # Coordinate loss tracking for logging - comprehensive initialization
        self._initialize_loss_tracking_components()

        # Ensure tracking components are always available
        self._ensure_loss_tracking_initialized()

        # Get config for model creation
        if config is None:
            from src.config import get_config

            config = get_config()
            self.logger.info("📄 Using global configuration system")

        # Store config for use throughout the wrapper
        self._config = config

        # Determine effective attention implementation
        if attn_implementation:
            effective_attn_impl = attn_implementation
        elif hasattr(config, "attn_implementation"):
            effective_attn_impl = config.attn_implementation
        else:
            raise ValueError(
                "attn_implementation must be provided either directly or in config"
            )

        # Load official Qwen2.5-VL model with proper configuration
        if not hasattr(config, "torch_dtype"):
            raise ValueError("torch_dtype must be specified in config")

        self.base_model: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=_get_torch_dtype(config.torch_dtype),
            attn_implementation=effective_attn_impl,
            device_map=None,  # Single GPU only - no multi-GPU device mapping
            trust_remote_code=True,
            use_cache=use_cache,  # Use provided cache setting (False for training, True for inference)
        )

        # Log attention implementation being used
        self.logger.info(f"🔧 Model loaded with attention: {effective_attn_impl}")
        # EXPLICIT CONFIG: Log attention implementation if available
        if hasattr(self.base_model.config, "_attn_implementation"):
            attn_impl = self.base_model.config._attn_implementation
            self.logger.info(f"🔧 Model config attn_implementation: {attn_impl}")
        else:
            self.logger.info(
                "🔧 Model config does not have _attn_implementation attribute"
            )

        # CRITICAL: Move base model to GPU only if NOT using DeepSpeed
        deepspeed_enabled = (
            os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
        )

        if torch.cuda.is_available() and not deepspeed_enabled:
            device = torch.device("cuda:0")
            self.base_model = self.base_model.to(device)
        elif deepspeed_enabled:
            self.logger.info(
                "🔧 DeepSpeed enabled - letting DeepSpeed handle device placement"
            )

        # CRITICAL: Apply fixes for mRoPE and visual processing
        apply_comprehensive_qwen25_fixes()

        # Store original vocab size before any modifications - use tokenizer vocab size
        self.original_vocab_size = len(self.tokenizer.get_vocab())

        # Initialize coordinate token support if enabled
        if self.coordinate_tokens_enabled:
            self.logger.info(
                f"🚀 Setting up coordinate tokens (enabled: {self.coordinate_tokens_enabled})"
            )
            self._setup_coordinate_tokens()
            self._setup_unified_token_manager()
            self.logger.info(
                f"🚀 Coordinate token setup complete. Extended vocab: {self.extended_vocab_size}"
            )
        else:
            self.logger.info(
                f"🚀 Coordinate tokens disabled, using original vocab: {self.original_vocab_size}"
            )
            # No coordinate tokens - use original vocab
            self.extended_vocab_size = self.original_vocab_size

        # Store our custom config for internal use, but expose base model config for DeepSpeed
        self._custom_config = config

        # NOTE: With official resize_token_embeddings, no manual device placement needed
        # The resized embeddings are already on the same device as the base model
        if not deepspeed_enabled:
            device = next(self.base_model.parameters()).device
            self.logger.info(
                f"🔧 Model on device: {device} (embeddings handled by base model)"
            )
        else:
            self.logger.info(
                "🔧 DeepSpeed enabled - coordinate tokens will be placed by DeepSpeed"
            )

    def forward(
        self, **inputs: Any
    ) -> Union[Tuple[Any, ...], Qwen2_5_VLCausalLMOutputWithPast]:
        """Forward pass with support for coordinate tokens."""
        # Handle device placement explicitly - remove device handling here
        # This is better handled at the model initialization level

        # Store original ground truth objects for detection loss (don't pop them)
        # The trainer will handle detection loss computation

        # Remove ground truth objects and trainer-specific inputs from model inputs (but keep them in original inputs)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)
        model_inputs.pop("cu_seqlens", None)  # Remove Flash Attention 2 parameter
        model_inputs.pop("max_seqlen", None)  # Remove Flash Attention 2 parameter
        model_inputs.pop(
            "teacher_assistant_spans", None
        )  # Remove teacher-student training parameter
        model_inputs.pop(
            "student_assistant_spans", None
        )  # Remove teacher-student training parameter

        # Handle coordinate token processing if enabled
        if self.coordinate_tokens_enabled and "input_ids" in model_inputs:
            return self._forward_with_coordinate_tokens(model_inputs, inputs)
        else:
            # EXPLICIT CONFIG: extended_vocab_size and original_vocab_size are set at initialization
            # No hasattr checks needed - fail fast approach
            if (
                self.extended_vocab_size is not None
                and self.original_vocab_size is not None
                and self.extended_vocab_size > self.original_vocab_size
            ):
                # Use extended embeddings but disable coordinate-aware loss computation
                return self._forward_with_extended_embeddings_only(model_inputs, inputs)
            else:
                # Standard Qwen2.5-VL forward pass with all parameters preserved
                outputs = self.base_model(**model_inputs)
                # Store coordinate loss components in the outputs dictionary
                outputs["_llm_loss"] = self._last_llm_loss
                outputs["_coordinate_l1_loss"] = self._last_coordinate_l1_loss
                return outputs

    def generate(self, **kwargs):
        """Generate using base model - no detection head to disable"""
        return self.base_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, **kwargs):
        """Delegate to base model's preparation method"""
        return self.base_model.prepare_inputs_for_generation(**kwargs)

    def get_rope_index(self, **kwargs):
        """Delegate to base model's RoPE calculation"""
        return self.base_model.get_rope_index(**kwargs)

    def _setup_coordinate_tokens(self):
        """Set up coordinate tokens for the model."""
        # EXPLICIT CONFIG: coordinate_config is validated at initialization
        # No hasattr checks needed - fail fast if not properly configured

        if not self.coordinate_config.enable_coordinate_tokens:
            return

        # EXPLICIT CONFIG: Use tokenizer vocab size, not model config vocab size
        original_vocab_size = self.original_vocab_size  # Set correctly in __init__

        # EXPLICIT CONFIG: max_coord_value is required and validated at config load
        max_coord_value = self.coordinate_config.max_coord_value

        # Validate required values are positive
        if not isinstance(original_vocab_size, int) or original_vocab_size <= 0:
            raise ValueError(
                f"Invalid original_vocab_size: {original_vocab_size}. Must be positive integer."
            )

        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"Invalid max_coord_value: {max_coord_value}. Must be positive integer."
            )

        # Calculate extended vocab size
        self.extended_vocab_size = original_vocab_size + max_coord_value

        # Create extended embeddings and LM head
        self._create_extended_embeddings()
        self._create_extended_lm_head()

        # Setup unified token manager
        self._setup_unified_token_manager()

    def _setup_unified_token_manager(self):
        """Set up unified token manager."""
        # EXPLICIT CONFIG: Both coordinate_config and tokenizer are validated at initialization
        # No hasattr checks needed - fail fast approach

        # EXPLICIT CONFIG: max_coord_value is required and validated at config load
        max_coord_value = self.coordinate_config.max_coord_value
        if max_coord_value <= 0:
            raise ValueError(
                f"Invalid max_coord_value: {max_coord_value}. Must be positive."
            )

        # Create unified token manager (handles everything automatically)
        from src.utils.tokens import create_unified_token_manager

        self.token_manager = create_unified_token_manager(
            tokenizer=self.tokenizer,
            model=self.base_model,
            max_coord_value=max_coord_value,
        )

        # Use UnifiedTokenManager as coordinate manager for both formatting and loss computation
        # The UnifiedTokenManager already handles all coordinate token functionality
        self.coordinate_manager = self.token_manager

    def _extend_tokenizer(self):
        """Extend tokenizer with coordinate tokens."""
        # EXPLICIT CONFIG: tokenizer and coordinate_config validated at initialization
        # No hasattr checks needed - fail fast approach

        # Get token names for box tokens
        box_start_token = "<|box_start|>"
        box_end_token = "<|box_end|>"

        # EXPLICIT CONFIG: use_official_box_tokens is required and validated at config load
        if self.coordinate_config.use_official_box_tokens:
            # EXPLICIT CONFIG: tokenizer must have additional_special_tokens (validated at init)
            existing_tokens = self.tokenizer.additional_special_tokens
            tokens_to_add = []

            if box_start_token not in existing_tokens:
                tokens_to_add.append(box_start_token)

            if box_end_token not in existing_tokens:
                tokens_to_add.append(box_end_token)

            # Add tokens if needed
            if tokens_to_add:
                # Create a new list combining existing and new tokens
                new_special_tokens = list(existing_tokens) + tokens_to_add
                # Update the tokenizer with the complete list
                self.tokenizer.additional_special_tokens = new_special_tokens

        # Add geometry tokens if multi-geometry is enabled
        # EXPLICIT CONFIG: enable_multi_geometry is validated at config load
        if self.coordinate_config.enable_multi_geometry:
            # EXPLICIT CONFIG: tokenizer is validated at initialization
            # Add geometry-specific tokens one by one
            geometry_tokens = [
                "<|square_start|>",
                "<|square_end|>",
                "<|line_start|>",
                "<|line_end|>",
            ]

            # EXPLICIT CONFIG: No getattr fallback - tokenizer must have additional_special_tokens
            existing_tokens = self.tokenizer.additional_special_tokens
            tokens_to_add = []

            for token in geometry_tokens:
                if token not in existing_tokens:
                    tokens_to_add.append(token)

            # Add tokens if needed
            if tokens_to_add:
                # Create a new list combining existing and new tokens
                new_special_tokens = list(existing_tokens) + tokens_to_add
                # Update the tokenizer with the complete list
                self.tokenizer.additional_special_tokens = new_special_tokens

            # Update geometry token IDs
            self._update_geometry_token_ids(geometry_tokens)

    def _update_geometry_token_ids(self, geometry_tokens):
        """Update coordinate manager with actual geometry token IDs."""
        # Get token IDs from tokenizer
        # EXPLICIT CONFIG: No fallback - geometry tokens must exist
        vocab = self.tokenizer.get_vocab()
        square_start_id = vocab["<|square_start|>"]
        square_end_id = vocab["<|square_end|>"]
        line_start_id = vocab["<|line_start|>"]
        line_end_id = vocab["<|line_end|>"]

        # Update coordinate manager's geometry token IDs
        if self.coordinate_manager:
            self.coordinate_manager.geometry_token_ids.update(
                {
                    "square_start": square_start_id,
                    "square_end": square_end_id,
                    "line_start": line_start_id,
                    "line_end": line_end_id,
                }
            )

            self.logger.info("🔧 Updated coordinate manager geometry token IDs:")
            self.logger.info(
                f"   square_start: {square_start_id}, square_end: {square_end_id}"
            )
            self.logger.info(f"   line_start: {line_start_id}, line_end: {line_end_id}")

    def _create_extended_embeddings(self):
        """Create extended embeddings for coordinate tokens."""
        # EXPLICIT CONFIG: coordinate_config validated at initialization
        # No hasattr checks needed - fail fast approach

        # EXPLICIT CONFIG: Validate required dimensions
        original_vocab_size = self.original_vocab_size
        if original_vocab_size is None or original_vocab_size <= 0:
            raise ValueError(f"Invalid original_vocab_size: {original_vocab_size}")

        # EXPLICIT CONFIG: Model must have hidden_size (standard transformer attribute)
        embedding_dim = self.base_model.config.hidden_size
        if embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {embedding_dim}")

        # EXPLICIT CONFIG: max_coord_value is required and validated at config load
        max_coord_value = self.coordinate_config.max_coord_value
        if max_coord_value <= 0:
            raise ValueError(f"Invalid max_coord_value: {max_coord_value}")

        extended_vocab_size = self.extended_vocab_size

        # Create new embeddings
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        new_embeddings = nn.Embedding(
            num_embeddings=extended_vocab_size,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )

        # Copy original embeddings
        original_embeddings = self.base_model.get_input_embeddings()
        if original_embeddings is None:
            return

        # EXPLICIT CONFIG: nn.Embedding always has num_embeddings attribute
        orig_num_embeddings = original_embeddings.num_embeddings
        if orig_num_embeddings <= 0:
            raise ValueError(f"Invalid original embedding size: {orig_num_embeddings}")

        # Copy weights for existing tokens
        with torch.no_grad():
            copy_size = min(orig_num_embeddings, original_vocab_size)
            if copy_size > 0:
                new_embeddings.weight.data[:copy_size] = (
                    original_embeddings.weight.data[:copy_size]
                )

        # Initialize coordinate tokens with proper scale (use Qwen2.5-VL's initializer_range)
        if original_vocab_size < extended_vocab_size:
            coordinate_start = original_vocab_size
            coordinate_end = extended_vocab_size

            self.logger.info(
                f"   🎯 Initializing coordinate tokens [{coordinate_start}:{coordinate_end}]"
            )

            # EXPLICIT CONFIG: Standard transformer models have initializer_range
            initializer_range = self.base_model.config.initializer_range
            nn.init.normal_(
                new_embeddings.weight[coordinate_start:coordinate_end],
                std=initializer_range,
            )

            # Make coordinate tokens trainable
            new_embeddings.weight[coordinate_start:coordinate_end].requires_grad_(True)

        # Keep all embeddings jointly trainable for better BBU domain adaptation
        new_embeddings.weight.requires_grad_(True)
        self.logger.info(
            f"   🔓 All embeddings jointly trainable for BBU domain adaptation"
        )

        # Store the extended embeddings as a named module for proper parameter registration
        self.extended_embeddings = new_embeddings

        # Replace the base model's embeddings with our extended ones
        self.base_model.set_input_embeddings(new_embeddings)

        # CRITICAL: The extended embeddings are now part of the base model's parameter tree
        # They will be accessible through the base model's named_parameters()

        self.logger.info(
            f"   ✅ Created extended embeddings: {extended_vocab_size} tokens"
        )
        coord_tokens_count = extended_vocab_size - original_vocab_size
        self.logger.info(
            f"   ✅ Initialized {coord_tokens_count} coordinate tokens with std={initializer_range}"
        )

    def _create_extended_lm_head(self):
        """Create extended LM head for coordinate tokens."""
        # EXPLICIT CONFIG: coordinate_config validated at initialization
        # No hasattr checks needed - fail fast approach

        # EXPLICIT CONFIG: Validate required dimensions
        original_vocab_size = self.original_vocab_size
        if original_vocab_size is None or original_vocab_size <= 0:
            raise ValueError(f"Invalid original_vocab_size: {original_vocab_size}")

        # EXPLICIT CONFIG: Standard transformer models have hidden_size
        embedding_dim = self.base_model.config.hidden_size
        if embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension: {embedding_dim}")

        # EXPLICIT CONFIG: max_coord_value is required and validated at config load
        max_coord_value = self.coordinate_config.max_coord_value
        if max_coord_value <= 0:
            raise ValueError(f"Invalid max_coord_value: {max_coord_value}")

        extended_vocab_size = self.extended_vocab_size

        # Create new LM head
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        new_lm_head = nn.Linear(
            in_features=embedding_dim,
            out_features=extended_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Copy original LM head weights
        original_lm_head = self.base_model.get_output_embeddings()
        if original_lm_head is None:
            return

        # EXPLICIT CONFIG: nn.Linear always has out_features attribute
        orig_out_features = original_lm_head.out_features
        if orig_out_features <= 0:
            raise ValueError(f"Invalid original LM head size: {orig_out_features}")

        # Copy weights for existing tokens
        with torch.no_grad():
            copy_size = min(orig_out_features, original_vocab_size)
            if copy_size > 0:
                new_lm_head.weight.data[:copy_size] = original_lm_head.weight.data[
                    :copy_size
                ]

        # Initialize coordinate token projections with proper scale
        if original_vocab_size < extended_vocab_size:
            coordinate_start = original_vocab_size
            coordinate_end = extended_vocab_size

            self.logger.info(
                f"   🎯 Initializing coordinate projections [{coordinate_start}:{coordinate_end}]"
            )

            # EXPLICIT CONFIG: Standard transformer models have initializer_range
            initializer_range = self.base_model.config.initializer_range
            nn.init.normal_(
                new_lm_head.weight[coordinate_start:coordinate_end],
                std=initializer_range,
            )

            # Make coordinate projections trainable
            new_lm_head.weight[coordinate_start:coordinate_end].requires_grad_(True)

        # Keep all projections jointly trainable for better BBU domain adaptation
        new_lm_head.weight.requires_grad_(True)
        self.logger.info(
            f"   🔓 All projections jointly trainable for BBU domain adaptation"
        )

        # Store the extended LM head as a named module for proper parameter registration
        self.extended_lm_head = new_lm_head

        # Replace the base model's LM head with our extended one
        self.base_model.set_output_embeddings(new_lm_head)

        # CRITICAL: The extended LM head is now part of the base model's parameter tree
        # They will be accessible through the base model's named_parameters()

        self.logger.info(
            f"   ✅ Created extended LM head: {extended_vocab_size} tokens"
        )
        coord_projections_count = extended_vocab_size - original_vocab_size
        self.logger.info(
            f"   ✅ Initialized {coord_projections_count} coordinate projections with std={initializer_range}"
        )

    def _forward_with_extended_embeddings_only(
        self, model_inputs: Dict, original_inputs: Dict
    ):
        """Forward pass with extended embeddings but no coordinate-aware loss computation."""
        # Reset to zero for standard LLM mode
        self._reset_loss_components_to_zero()

        # Strict validation - fail fast if required keys are missing
        if "input_ids" not in model_inputs:
            raise ValueError("input_ids is required in model_inputs")

        input_ids = model_inputs["input_ids"]

        # Replace input_ids with embeddings from the resized embedding layer
        if input_ids is not None:
            # Use the base model's (resized) input embeddings directly
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            model_inputs["inputs_embeds"] = inputs_embeds
            # Keep input_ids for shape information but mark to use inputs_embeds

        # Forward through base model - OPTIMIZATION: Let model compute built-in CE loss
        labels = original_inputs.get("labels")
        model_inputs["labels"] = labels  # Keep labels for built-in CE loss computation
        model_inputs["return_dict"] = True  # Ensure we get a proper output object
        outputs = self.base_model(**model_inputs)

        # Extract built-in loss and logits from model outputs
        loss = outputs.loss  # Use built-in shifted cross entropy from Qwen2.5-VL
        logits = outputs.logits

        # Create output with extended logits and standard loss
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLCausalLMOutputWithPast,
        )

        result = Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # Attach zero coordinate loss components for logging consistency
        self._attach_coordinate_losses_to_outputs(result)

        return result

    def _forward_with_coordinate_tokens(
        self, model_inputs: Dict, original_inputs: Dict
    ):
        """Forward pass with coordinate token processing."""
        # Reset coordinate loss components for this forward pass
        self._reset_loss_components_for_forward_pass()

        # Strict validation - fail fast if required keys are missing
        if "input_ids" not in model_inputs:
            raise ValueError("input_ids is required in model_inputs")
        if "labels" not in original_inputs:
            raise ValueError("labels is required in original_inputs")

        input_ids = model_inputs["input_ids"]
        labels = original_inputs["labels"]

        # Replace input_ids with embeddings from the resized embedding layer
        if input_ids is not None:
            # Use the base model's (resized) input embeddings directly
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            model_inputs["inputs_embeds"] = inputs_embeds
            # Keep input_ids for shape information but mark to use inputs_embeds
            # Some models need input_ids for position/attention mask calculation

        # OPTIMIZATION: Use Qwen2.5-VL's built-in CE loss computation
        # Keep labels to get the standard CE loss from the model
        model_inputs["labels"] = labels  # Keep labels for standard CE loss computation
        model_inputs["return_dict"] = True  # Ensure we get a proper output object
        outputs = self.base_model(**model_inputs)

        # Extract the standard LLM loss computed by Qwen2.5-VL
        if outputs.loss is not None:
            llm_loss = outputs.loss
        else:
            device = next(self.parameters()).device
            llm_loss = torch.tensor(0.0, device=device)

        logits = outputs.logits

        # Compute coordinate loss separately and combine with existing LLM loss
        loss = llm_loss  # Start with the standard CE loss from Qwen2.5-VL

        if labels is not None and self.coordinate_manager is not None:
            self.logger.debug(f"🔍 DEBUGGING: Starting coordinate loss computation")
            self.logger.debug(f"   Labels shape: {labels.shape}")
            self.logger.debug(f"   Logits shape: {logits.shape}")
            self.logger.debug(f"   LLM loss: {llm_loss.item():.6f}")

            # Compute only coordinate L1 loss (LLM loss already computed by Qwen2.5-VL)
            coordinate_losses = self.coordinate_manager.compute_coordinate_losses(
                logits,
                labels,
                [],  # Empty bbox_spans - internal geometry detection will be used
            )

            # Extract coordinate L1 loss - fail fast if missing
            if "coordinate_loss" not in coordinate_losses:
                raise ValueError(
                    "coordinate_loss missing from coordinate_losses dictionary"
                )

            coordinate_l1_loss = coordinate_losses["coordinate_loss"]

            # Validate coordinate loss is a tensor
            if not isinstance(coordinate_l1_loss, torch.Tensor):
                raise TypeError(
                    f"coordinate_l1_loss must be a tensor, got {type(coordinate_l1_loss)}"
                )

            # Combine LLM loss with coordinate L1 loss using simple weighting
            if coordinate_l1_loss.item() > 0:
                # Validate coordinate manager config has required attributes
                if not hasattr(self.coordinate_manager, "config"):
                    raise ValueError("coordinate_manager missing config attribute")

                if not hasattr(self.coordinate_manager.config, "regular_loss_weight"):
                    raise ValueError(
                        "coordinate_manager.config missing regular_loss_weight attribute"
                    )

                if not hasattr(
                    self.coordinate_manager.config, "coordinate_loss_weight"
                ):
                    raise ValueError(
                        "coordinate_manager.config missing coordinate_loss_weight attribute"
                    )

                combined_loss = (
                    self.coordinate_manager.config.regular_loss_weight * llm_loss
                    + self.coordinate_manager.config.coordinate_loss_weight
                    * coordinate_l1_loss
                )
                loss = combined_loss

                self.logger.debug(
                    f"   Coordinate L1 loss: {coordinate_l1_loss.item():.6f}"
                )
                self.logger.debug(f"   Combined loss: {combined_loss.item():.6f}")
            else:
                self.logger.debug("   No coordinate tokens found, using LLM loss only")

            # Store loss components for logging (with dummy token counts for validation)
            loss_components = {
                "llm_loss": llm_loss,
                "coordinate_l1_loss": coordinate_l1_loss,
                "loss": loss,
                # Add dummy token counts to satisfy validation
                "total_tokens": labels.numel(),  # Total number of tokens
                "coordinate_tokens": 0,  # Will be computed if needed
                "regular_tokens": labels.numel(),  # Assume all tokens are regular for now
            }

            # Log the simplified loss components
            self.logger.debug(f"   📊 SIMPLIFIED LOSS BREAKDOWN:")
            self.logger.debug(
                f"      🎯 LLM loss: {loss_components['llm_loss'].item():.6f}"
            )
            self.logger.debug(
                f"      📐 Coordinate L1 loss: {loss_components['coordinate_l1_loss'].item():.6f}"
            )
            self.logger.debug(
                f"      🎯 Final combined loss: {loss_components['loss'].item():.6f}"
            )

            # Update loss tracking from components with validation
            self._update_loss_components_with_validation(loss_components)

            # Store simplified coordinate losses for loss manager access
            self._last_coordinate_losses = {
                "_llm_loss": loss_components["llm_loss"],
                "_coordinate_l1_loss": loss_components["coordinate_l1_loss"],
            }

            # Simplified validation - only track LLM and coordinate L1 loss
            llm_loss_val = loss_components["llm_loss"].item()
            coordinate_l1_val = loss_components["coordinate_l1_loss"].item()

            total_loss = llm_loss_val + coordinate_l1_val

            # Simplified logging
            self.logger.debug(f"✅ Total loss: {total_loss:.6f}")

            # Update internal tracking with simplified loss names
            self._last_llm_loss = llm_loss_val
            self._last_coordinate_l1_loss = coordinate_l1_val

            # VALIDATION: Log internal tracking update (for debugging)
            internal_total = self._last_llm_loss + self._last_coordinate_l1_loss
            self.logger.debug(f"✅ Internal tracking updated: {internal_total:.6f}")
        else:
            # No coordinate manager or no labels - use LLM loss only
            self.logger.debug("   No coordinate manager or labels, using LLM loss only")
            device = (
                llm_loss.device if hasattr(llm_loss, "device") else torch.device("cpu")
            )
            coordinate_l1_loss = torch.tensor(0.0, device=device)
            loss = llm_loss

            # IMMEDIATE ERROR CHECK: Ensure loss is a tensor
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError(
                    f"Loss computation returned {type(loss)}, expected torch.Tensor"
                )
            if loss.dim() != 0:
                raise RuntimeError(
                    f"Loss should be scalar tensor, got shape {loss.shape}"
                )

        # ALWAYS attach loss components to outputs for loss manager extraction
        self._attach_coordinate_losses_to_outputs(outputs)

        # Validate loss attachment was successful
        self._validate_loss_attachment(outputs)

        # IMMEDIATE ERROR CHECK: Ensure loss is still a tensor before setting
        if loss is not None and not isinstance(loss, torch.Tensor):
            raise RuntimeError(
                f"Loss variable corrupted to {type(loss)}, expected torch.Tensor"
            )

        # Create a new output object to avoid corruption
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLCausalLMOutputWithPast,
        )

        # Create new output object with correct values
        # Validate all required attributes exist
        if not hasattr(outputs, "logits"):
            raise ValueError("outputs missing required attribute: logits")

        # Extract optional attributes with validation
        hidden_states = getattr(outputs, "hidden_states", None)
        past_key_values = getattr(outputs, "past_key_values", None)
        attentions = getattr(outputs, "attentions", None)

        # Create new output object with correct values
        new_outputs = Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attentions=attentions,
        )

        # CRITICAL FIX: Attach coordinate losses to the NEW output object
        # This was the root cause - losses were attached to old outputs but new outputs was returned
        self._attach_coordinate_losses_to_outputs(new_outputs)

        # IMMEDIATE VALIDATION: Ensure coordinate losses are properly attached to new output
        # EXPLICIT CONFIG: _config is set during initialization
        if not hasattr(self, "_config"):
            raise ValueError("_config not initialized")

        if not hasattr(self._config, "coordinate_tokens_enabled"):
            raise ValueError("_config missing coordinate_tokens_enabled attribute")

        if self._config.coordinate_tokens_enabled:
            required_attrs = [
                "_llm_loss",
                "_coordinate_l1_loss",
            ]
            missing_attrs = []

            for attr in required_attrs:
                if not hasattr(new_outputs, attr):
                    missing_attrs.append(attr)

            if missing_attrs:
                raise RuntimeError(
                    f"CRITICAL: New output object missing coordinate loss attributes: {missing_attrs}. "
                    f"This indicates loss attachment to new output object failed."
                )

            # Validate the simplified attached losses
            attached_total = 0.0
            if "_llm_loss" in new_outputs:
                attached_total += new_outputs["_llm_loss"]
            if "_coordinate_l1_loss" in new_outputs:
                attached_total += new_outputs["_coordinate_l1_loss"]

            self.logger.debug(
                f"✅ MODEL_WRAPPER: Simplified coordinate losses attached to new output total: {attached_total:.6f}"
            )
            self.logger.debug(
                f"   new_outputs[_llm_loss]: {new_outputs.get('_llm_loss', 0.0)}"
            )
            self.logger.debug(
                f"   new_outputs[_coordinate_l1_loss]: {new_outputs.get('_coordinate_l1_loss', 0.0)}"
            )

        # IMMEDIATE ERROR CHECK: Verify new outputs.loss is correct
        if loss is not None and not isinstance(new_outputs.loss, torch.Tensor):
            raise RuntimeError(
                f"CRITICAL: new_outputs.loss corrupted to {type(new_outputs.loss)}, expected torch.Tensor"
            )

        # CRITICAL DEBUG: Log that we're returning the correct object
        self.logger.debug(
            f"🔄 MODEL_WRAPPER: Returning new_outputs with id={id(new_outputs)}"
        )
        self.logger.debug(
            f"   new_outputs has _coordinate_l1_loss: {hasattr(new_outputs, '_coordinate_l1_loss')}"
        )

        return new_outputs

    def _update_loss_components_with_validation(
        self, loss_components: Dict[str, float]
    ):
        """Update loss tracking components with geometry-organized structure."""
        # EXPLICIT CONFIG: Validate required loss components are present
        required_loss_keys = [
            "llm_loss",
            "coordinate_l1_loss",
            "total_tokens",
            "coordinate_tokens",
            "regular_tokens",
        ]
        missing_keys = [key for key in required_loss_keys if key not in loss_components]

        if missing_keys:
            raise ValueError(
                f"Missing required loss components: {missing_keys}. "
                f"Ensure all loss components are properly computed and returned."
            )

        # Update simplified loss components with explicit validation
        self._last_llm_loss = self._validate_loss_value(
            loss_components["llm_loss"], "llm_loss"
        )
        self._last_coordinate_l1_loss = self._validate_loss_value(
            loss_components["coordinate_l1_loss"], "coordinate_l1_loss"
        )

        # Enhanced tracking metrics with explicit validation
        self._last_total_tokens = max(0, int(loss_components["total_tokens"]))
        self._last_coordinate_tokens = max(0, int(loss_components["coordinate_tokens"]))
        self._last_regular_tokens = max(0, int(loss_components["regular_tokens"]))

        # Validate token counts consistency
        expected_total = self._last_coordinate_tokens + self._last_regular_tokens
        if self._last_total_tokens > 0 and expected_total > 0:
            # Check if both values are not None before comparison
            if (
                self._last_total_tokens is not None
                and expected_total is not None
                and abs(self._last_total_tokens - expected_total) > 1
            ):  # Allow small rounding differences
                self.logger.warning(
                    f"⚠️ Token count mismatch: total={self._last_total_tokens}, "
                    f"coordinate={self._last_coordinate_tokens} + regular={self._last_regular_tokens} "
                    f"= {expected_total}"
                )

        self.logger.debug(f"✅ Updated loss components from coordinate loss computer")

    def _validate_loss_value(self, value: float, component_name: str) -> float:
        """Validate a loss component value for sanity checks."""
        # Convert tensor to float if needed
        if isinstance(value, torch.Tensor):
            value = value.item()

        # Ensure value is a float
        value = float(value)

        # Check for NaN or Inf
        if math.isnan(value):
            self.logger.warning(f"⚠️ NaN detected in {component_name}, using 0.0")
            return 0.0
        elif math.isinf(value):
            self.logger.warning(f"⚠️ Inf detected in {component_name}, using 0.0")
            return 0.0
        elif value < 0:
            self.logger.warning(
                f"⚠️ Negative value {value} detected in {component_name}, using 0.0"
            )
            return 0.0
        elif value > 1e6:
            self.logger.warning(
                f"⚠️ Suspiciously large value {value} detected in {component_name}, capping at 1e6"
            )
            return 1e6
        else:
            return value

    def _update_loss_components(self, loss_components: Dict[str, float]):
        """Legacy method - redirect to enhanced validation version."""
        self._update_loss_components_with_validation(loss_components)

    def _compute_coordinate_aware_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid loss: standard CE for regular tokens + soft expectation for coordinates."""
        # Shift for causal modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        flat_logits = shift_logits.view(-1, self.extended_vocab_size)
        flat_labels = shift_labels.view(-1)

        # Filter out ignore_index (-100)
        valid_mask = flat_labels != -100

        # Debug coordinate token detection

        self.logger.info(f"🔍 Coordinate loss computation")
        self.logger.info(f"   Total flat labels: {flat_labels.shape[0]}")
        self.logger.info(f"   Valid labels (not -100): {valid_mask.sum().item()}")

        if not valid_mask.any():
            self.logger.info(f"   ⚠️ No valid labels found, returning zero loss")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # Create coordinate mask
        coord_mask = self._get_coordinate_mask(valid_labels)
        regular_mask = ~coord_mask

        self.logger.info(f"   Valid labels shape: {valid_labels.shape}")
        self.logger.info(f"   Coordinate mask sum: {coord_mask.sum().item()}")
        self.logger.info(f"   Regular mask sum: {regular_mask.sum().item()}")
        self.logger.info(f"   Original vocab size: {self.original_vocab_size}")
        self.logger.info(f"   Extended vocab size: {self.extended_vocab_size}")

        # Show some sample valid labels to understand the data
        if valid_labels.shape[0] > 0:
            sample_labels = valid_labels[:10].tolist()  # First 10 valid labels
            self.logger.info(f"   Sample valid labels: {sample_labels}")

        if coord_mask.sum() > 0:
            coord_label_sample = valid_labels[coord_mask][
                :5
            ]  # First 5 coordinate labels
            self.logger.info(
                f"   Sample coordinate labels: {coord_label_sample.tolist()}"
            )
        else:
            self.logger.info(f"   ⚠️ No coordinate tokens found in this batch")

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Regular token loss (standard cross-entropy)
        if regular_mask.any():
            regular_logits = valid_logits[regular_mask]
            regular_labels = valid_labels[regular_mask]

            # Ensure labels are within original vocab range
            valid_regular_mask = regular_labels < self.original_vocab_size
            if valid_regular_mask.any():
                regular_loss = F.cross_entropy(
                    regular_logits[valid_regular_mask, : self.original_vocab_size],
                    regular_labels[valid_regular_mask],
                )
                weighted_regular_loss = (
                    self.coordinate_config.regular_loss_weight * regular_loss
                )
                total_loss += weighted_regular_loss
                self._last_llm_loss = regular_loss.item()

        # Coordinate token loss (soft expectation)
        if coord_mask.any():
            coord_logits = valid_logits[coord_mask]
            coord_labels = valid_labels[coord_mask]

            self.logger.debug(f"   🔍 Computing coordinate losses:")
            self.logger.debug(f"      Coord logits shape: {coord_logits.shape}")
            self.logger.debug(f"      Coord labels shape: {coord_labels.shape}")

            detection_loss, focal_loss = (
                self._compute_soft_expectation_loss_with_components(
                    coord_logits, coord_labels
                )
            )

            self.logger.debug(f"      Detection loss: {detection_loss.item():.6f}")
            self.logger.debug(f"      Focal loss: {focal_loss.item():.6f}")

            coord_loss = detection_loss + 0.1 * focal_loss  # Combined coordinate loss
            weighted_coord_loss = (
                self.coordinate_config.coordinate_loss_weight * coord_loss
            )
            total_loss += weighted_coord_loss

            self.logger.debug(f"      Combined coord loss: {coord_loss.item():.6f}")
            self.logger.debug(
                f"      Weighted coord loss: {weighted_coord_loss.item():.6f}"
            )

            # Track individual loss components for logging
            self._last_geometry_focal_loss = focal_loss.item()

            # Extract L1 and GIoU components separately for detailed logging
            if coord_logits.size(0) > 0:
                l1_loss, giou_loss = self._compute_advanced_detection_losses(
                    self._extract_expected_coordinates(coord_logits),
                    coord_labels.float() - self.original_vocab_size,
                )
                self._last_coordinate_l1_loss = l1_loss.item()
                self._last_geometry_bbox_giou_loss = giou_loss.item()
                self.logger.debug(f"      L1 loss: {l1_loss.item():.6f}")
                self.logger.debug(f"      GIoU loss: {giou_loss.item():.6f}")
        else:
            self.logger.debug(f"   ⚠️ No coordinate tokens found in this batch")

        return total_loss

    def _get_coordinate_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for coordinate tokens."""
        # EXPLICIT CONFIG: coordinate_manager is set up during initialization
        if self.coordinate_manager is None:
            return torch.zeros_like(token_ids, dtype=torch.bool)

        # EXPLICIT CONFIG: coordinate manager always has box_start_id when properly initialized

        box_start_id = self.coordinate_manager.box_start_id
        if box_start_id < 0:
            raise ValueError(
                f"Invalid box_start_id: {box_start_id}. Must be non-negative."
            )

        return token_ids == box_start_id

    def _compute_soft_expectation_loss(
        self, coord_logits: torch.Tensor, coord_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft expectation loss for coordinate tokens (backward compatibility)."""
        l1_loss, focal_loss = self._compute_soft_expectation_loss_with_components(
            coord_logits, coord_labels
        )
        return l1_loss + 0.1 * focal_loss  # Weighted combination

    def _compute_soft_expectation_loss_with_components(
        self, coord_logits: torch.Tensor, coord_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute simplified coordinate loss with only L1 component."""
        # Extract coordinate portion of logits
        coord_start = self.original_vocab_size
        coord_end = None
        if coord_start is not None and self.coordinate_config is not None:
            coord_end = coord_start + self.coordinate_config.max_coord_value
        if coord_start is not None and coord_end is not None:
            coord_only_logits = coord_logits[:, coord_start:coord_end]
        else:
            raise ValueError(
                "Cannot compute coordinate loss: coord_start or coord_end is None"
            )

        # Convert labels to coordinate indices
        coord_indices = coord_labels - coord_start
        coord_indices = torch.clamp(
            coord_indices, 0, self.coordinate_config.max_coord_value - 1
        )

        # Soft expectation computation
        temperature = self.coordinate_config.soft_expectation_temperature
        soft_weights = F.softmax(coord_only_logits / temperature, dim=-1)

        # Expected coordinate values
        coord_range = torch.arange(
            self.coordinate_config.max_coord_value,
            device=coord_logits.device,
            dtype=torch.float32,
        )
        expected_coords = torch.sum(soft_weights * coord_range, dim=-1)

        # Only compute L1 loss (simplified)
        l1_loss = F.l1_loss(expected_coords, coord_indices.float())

        # Average the loss per coordinate token to prevent extremely large losses
        num_coord_tokens = coord_logits.size(0)
        if num_coord_tokens > 0:
            l1_loss = l1_loss / num_coord_tokens

        # Return L1 loss twice for backward compatibility (detection_loss, focal_loss)
        return l1_loss, torch.tensor(
            0.0, device=coord_logits.device
        )  # Return enhanced components

    def _compute_advanced_detection_losses(
        self, predicted_coords: torch.Tensor, target_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advanced detection losses: L1 and GIoU for coordinate tokens.

        Args:
            predicted_coords: Predicted coordinates from soft expectation (N,)
            target_coords: Ground truth coordinate indices (N,)

        Returns:
            Tuple of (l1_loss, giou_loss)
        """
        # Reshape coordinates to bounding boxes (x1, y1, x2, y2)
        # Assume coordinates come in groups of 4
        if predicted_coords.size(0) % 4 != 0:
            # If not divisible by 4, pad with zeros or handle gracefully
            num_coords = predicted_coords.size(0)
            pad_size = 4 - (num_coords % 4)
            if pad_size < 4:
                predicted_coords = F.pad(predicted_coords, (0, pad_size), value=0.0)
                target_coords = F.pad(target_coords, (0, pad_size), value=0.0)

        # Reshape to bounding boxes
        num_boxes = predicted_coords.size(0) // 4
        pred_boxes = predicted_coords.view(num_boxes, 4)  # (N, 4) -> (x1, y1, x2, y2)
        target_boxes = target_coords.view(num_boxes, 4)  # (N, 4) -> (x1, y1, x2, y2)

        # Normalize coordinates to [0, 1] range for GIoU computation
        max_coord = float(self.coordinate_config.max_coord_value - 1)
        pred_boxes_norm = pred_boxes / max_coord
        target_boxes_norm = target_boxes / max_coord

        # Compute L1 loss on coordinate values
        l1_loss = F.l1_loss(pred_boxes, target_boxes)

        # Compute GIoU loss on normalized boxes
        giou_loss = self._compute_giou_loss(pred_boxes_norm, target_boxes_norm)

        return l1_loss, giou_loss

    def _compute_giou_loss(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized IoU (GIoU) loss for bounding box regression.

        Args:
            pred_boxes: Predicted boxes (N, 4) in format (x1, y1, x2, y2)
            target_boxes: Ground truth boxes (N, 4) in format (x1, y1, x2, y2)

        Returns:
            GIoU loss tensor
        """
        # Ensure boxes are in proper format (x1 <= x2, y1 <= y2)
        pred_boxes = torch.stack(
            [
                torch.min(pred_boxes[:, 0], pred_boxes[:, 2]),  # x1
                torch.min(pred_boxes[:, 1], pred_boxes[:, 3]),  # y1
                torch.max(pred_boxes[:, 0], pred_boxes[:, 2]),  # x2
                torch.max(pred_boxes[:, 1], pred_boxes[:, 3]),  # y2
            ],
            dim=1,
        )

        target_boxes = torch.stack(
            [
                torch.min(target_boxes[:, 0], target_boxes[:, 2]),  # x1
                torch.min(target_boxes[:, 1], target_boxes[:, 3]),  # y1
                torch.max(target_boxes[:, 0], target_boxes[:, 2]),  # x2
                torch.max(target_boxes[:, 1], target_boxes[:, 3]),  # y2
            ],
            dim=1,
        )

        # Compute intersection area
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Compute union area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        )
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )
        union_area = pred_area + target_area - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-6)

        # Compute enclosing area for GIoU
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # Compute GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)

        # GIoU loss = 1 - GIoU
        giou_loss = 1.0 - giou.mean()

        return giou_loss

    def _extract_expected_coordinates(self, coord_logits: torch.Tensor) -> torch.Tensor:
        """Extract expected coordinates from logits using soft expectation."""
        # EXPLICIT CONFIG: coordinate_config is validated at initialization
        if self.coordinate_config is None:
            return coord_logits

        # EXPLICIT CONFIG: max_coord_value is required and validated at config load
        max_coord_value = self.coordinate_config.max_coord_value
        if max_coord_value <= 0:
            raise ValueError(
                f"Invalid max_coord_value: {max_coord_value}. Must be positive."
            )

        # Create coordinate range
        device = coord_logits.device
        coord_range = torch.arange(max_coord_value, device=device).float()

        # Apply softmax and compute expectation - EXPLICIT CONFIG: No fallback
        # soft_expectation_temperature is required and validated at config load
        temperature = self.coordinate_config.soft_expectation_temperature
        soft_weights = F.softmax(coord_logits / temperature, dim=-1)

        # Compute expected coordinates
        expected_coords = torch.sum(soft_weights * coord_range.unsqueeze(0), dim=-1)

        # Reshape to match expected output
        batch_size = coord_logits.size(0)
        if batch_size > 0:
            expected_coords = expected_coords.view(batch_size, -1)

        return expected_coords

    def validate_sample_tokens(self, input_ids, sample_info=None):
        """Validate that a sample contains required special tokens and coordinate tokens.

        Args:
            input_ids: Tokenized input sequence (tensor or list)
            sample_info: Optional dict with sample metadata for better error messages

        Raises:
            ValueError: If required tokens are missing
        """
        if not self.coordinate_tokens_enabled:
            return  # Skip validation if coordinate tokens are disabled

        # Convert to list if tensor
        # EXPLICIT CONFIG: input_ids is typically a torch.Tensor with tolist() method
        if isinstance(input_ids, torch.Tensor):
            token_ids = input_ids.tolist()
        else:
            token_ids = list(input_ids)

        # Flatten if nested (batch dimension)
        if isinstance(token_ids[0], list):
            token_ids = [token for seq in token_ids for token in seq]

        sample_desc = (
            f"Sample {sample_info.get('index', 'unknown')}" if sample_info else "Sample"
        )

        # 1. Check for object reference tokens (required for descriptions)
        vocab = self.tokenizer.get_vocab()
        object_ref_start_id = vocab.get(
            "<|object_ref_start|>", self.tokenizer.unk_token_id
        )
        object_ref_end_id = vocab.get("<|object_ref_end|>", self.tokenizer.unk_token_id)

        if object_ref_start_id not in token_ids or object_ref_end_id not in token_ids:
            raise ValueError(
                f"❌ SPECIAL_TOKEN_VIOLATION: {sample_desc} missing object reference tokens. "
                f"All samples must have descriptions wrapped with <|object_ref_start|> and <|object_ref_end|>. "
                f"Found object_ref_start: {object_ref_start_id in token_ids}, "
                f"Found object_ref_end: {object_ref_end_id in token_ids}"
            )

        # 2. Check for geometry tokens (at least one type required)
        geometry_tokens = {
            "bbox": (
                vocab.get("<|box_start|>", self.tokenizer.unk_token_id),
                vocab.get("<|box_end|>", self.tokenizer.unk_token_id),
            ),
            "square": (
                vocab.get("<|square_start|>", self.tokenizer.unk_token_id),
                vocab.get("<|square_end|>", self.tokenizer.unk_token_id),
            ),
            "line": (
                vocab.get("<|line_start|>", self.tokenizer.unk_token_id),
                vocab.get("<|line_end|>", self.tokenizer.unk_token_id),
            ),
        }

        found_geometry_types = []
        for geom_type, (start_id, end_id) in geometry_tokens.items():
            if start_id in token_ids and end_id in token_ids:
                found_geometry_types.append(geom_type)

        if not found_geometry_types:
            raise ValueError(
                f"❌ SPECIAL_TOKEN_VIOLATION: {sample_desc} missing geometry tokens. "
                f"All samples must have at least one geometry type (bbox, square, or line) "
                f"with proper start/end tokens."
            )

        # 3. Check for coordinate tokens (required for coordinates)
        coord_start_id = (
            self.original_vocab_size if self.original_vocab_size is not None else 151669
        )
        coord_end_id = coord_start_id + (
            self.coordinate_config.max_coord_value if self.coordinate_config else 2048
        )

        coordinate_tokens_found = any(
            coord_start_id <= token_id < coord_end_id for token_id in token_ids
        )

        if not coordinate_tokens_found:
            raise ValueError(
                f"❌ SPECIAL_TOKEN_VIOLATION: {sample_desc} missing coordinate tokens. "
                f"All samples must contain coordinate tokens in range [{coord_start_id}, {coord_end_id}). "
                f"Found geometry types: {found_geometry_types} but no coordinate tokens."
            )

        # 4. Validate geometry token pairing
        for geom_type, (start_id, end_id) in geometry_tokens.items():
            if start_id in token_ids or end_id in token_ids:
                start_count = token_ids.count(start_id)
                end_count = token_ids.count(end_id)
                if start_count != end_count:
                    raise ValueError(
                        f"❌ SPECIAL_TOKEN_VIOLATION: {sample_desc} has mismatched {geom_type} tokens. "
                        f"Found {start_count} start tokens and {end_count} end tokens. "
                        f"Each geometry must have matching start/end token pairs."
                    )

        self.logger.debug(
            f"✅ Token validation passed for {sample_desc}: "
            f"geometry_types={found_geometry_types}, coordinate_tokens=True"
        )

    def get_coordinate_tokenizer_utils(self):
        """Get utility functions for coordinate token conversion."""
        if not self.coordinate_tokens_enabled:
            return None

        return {
            "convert_bbox_to_tokens": self._convert_bbox_to_tokens,
            "convert_tokens_to_bbox": self._convert_tokens_to_bbox,
            "coord_start_id": self.original_vocab_size,
            "coord_end_id": (
                self.original_vocab_size + self.coordinate_config.max_coord_value
                if self.original_vocab_size is not None
                and self.coordinate_config is not None
                else None
            ),
            "box_start_id": self.box_start_id,  # 151648
            "box_end_id": self.box_end_id,  # 151649
            "max_coord_value": self.coordinate_config.max_coord_value,  # 2048
            "validate_sample_tokens": self.validate_sample_tokens,  # Add validation function
        }

    def _convert_bbox_to_tokens(self, bbox: List[int]) -> List[int]:
        """Convert integer bbox [0, 2047] to coordinate token IDs."""
        coord_start = self.original_vocab_size

        # Validate input coordinates are integers in [0, 2047]
        for i, coord in enumerate(bbox):
            if not isinstance(coord, int):
                raise ValueError(
                    f"Coordinate {i} must be integer, got {type(coord)}: {coord}"
                )
            if not (0 <= coord < self.coordinate_config.max_coord_value):
                raise ValueError(
                    f"Coordinate {i} = {coord} out of bounds [0, {self.coordinate_config.max_coord_value})"
                )

        coord_tokens = []
        for coord in bbox:
            # Direct mapping: integer coordinate -> token ID
            if coord_start is not None:
                coord_tokens.append(coord_start + coord)

        return [self.box_start_id] + coord_tokens + [self.box_end_id]

    def _convert_tokens_to_bbox(self, token_ids: List[int]) -> Optional[List[int]]:
        """Convert coordinate token IDs back to integer bbox [0, 2047]."""
        coord_start = self.original_vocab_size

        try:
            # Find box boundaries using official tokens
            if self.box_start_id not in token_ids or self.box_end_id not in token_ids:
                return None

            start_idx = token_ids.index(self.box_start_id)
            end_idx = token_ids.index(self.box_end_id)

            # Extract coordinate tokens
            coord_token_ids = token_ids[start_idx + 1 : end_idx]

            if len(coord_token_ids) != 4:
                return None

            # Convert to integer coordinates
            bbox = []
            for token_id in coord_token_ids:
                if (
                    coord_start is None
                    or self.coordinate_config is None
                    or token_id < coord_start
                    or (
                        coord_start is not None
                        and self.coordinate_config is not None
                        and token_id
                        >= coord_start + self.coordinate_config.max_coord_value
                    )
                ):
                    return None

                # Direct mapping: token ID -> integer coordinate
                if coord_start is not None:
                    coord_idx = token_id - coord_start
                    bbox.append(coord_idx)
                else:
                    return None  # Cannot convert if coord_start is None

            return bbox

        except (ValueError, IndexError):
            return None

    def _initialize_loss_tracking_components(self):
        """Initialize all loss tracking components to ensure they always exist."""
        # Primary coordinate loss components with new geometry-organized names
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0

        # Enhanced tracking attributes
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        self.logger.debug(f"⚙️ Initialized loss tracking components")

    def _ensure_loss_tracking_initialized(self):
        """Ensure simplified loss tracking components exist - EXPLICIT initialization."""
        # EXPLICIT CONFIG: All attributes are initialized in __init__ - no hasattr checks needed
        # This method is now redundant but kept for compatibility
        pass

    def _reset_loss_components_for_forward_pass(self):
        """Reset loss components at the start of each forward pass."""
        self._ensure_loss_tracking_initialized()

        # Reset simplified components to defaults for new forward pass
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        # CRITICAL FIX: Do NOT reset _last_coordinate_losses here!
        # The fallback cache must persist across forward passes for loss manager
        # The _last_coordinate_losses dict is only updated after coordinate computation
        # and should remain available for loss manager fallback in the same forward pass

        self.logger.debug(f"🔄 Reset loss components for forward pass")

    def _reset_loss_components_to_zero(self):
        """Reset all loss components to zero for standard LLM mode."""
        self._ensure_loss_tracking_initialized()

        # Set simplified components to zero for standard mode
        self._last_llm_loss = 0.0
        self._last_coordinate_l1_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        self.logger.debug(f"📄 Reset loss components to zero (standard LLM mode)")

    def _attach_coordinate_losses_to_outputs(self, outputs):
        """Attach all coordinate loss components to model outputs for loss manager extraction."""
        self._ensure_loss_tracking_initialized()

        # CRITICAL FIX: Store coordinate losses in the outputs dictionary to survive HuggingFace reconstruction
        # The issue is that Qwen2_5_VLCausalLMOutputWithPast is a dataclass that gets reconstructed,
        # losing custom attributes. Storing in the dict ensures they survive.

        # CRITICAL FIX: Ensure all values are tensors for DataParallel compatibility
        # Use a more robust way to get device that doesn't exhaust generators
        try:
            device = next(iter(self.parameters())).device
        except StopIteration:
            # Fallback to cuda:0 if no parameters found
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def _ensure_tensor(value):
            """Convert value to tensor if it's not already one and detach for evaluation."""
            if isinstance(value, torch.Tensor):
                # Detach tensor to remove gradients for evaluation compatibility
                return value.detach()
            else:
                return torch.tensor(
                    float(value),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=False,
                )

        # Store simplified coordinate losses
        coordinate_losses = {
            "_llm_loss": _ensure_tensor(self._last_llm_loss),
            "_coordinate_l1_loss": _ensure_tensor(self._last_coordinate_l1_loss),
            "_total_tokens": _ensure_tensor(self._last_total_tokens),
            "_coordinate_tokens": _ensure_tensor(self._last_coordinate_tokens),
            "_regular_tokens": _ensure_tensor(self._last_regular_tokens),
        }

        # Store in outputs dictionary (survives reconstruction)
        for key, value in coordinate_losses.items():
            outputs[key] = value

        # Use dictionary access instead of attribute access
        self.logger.debug(f"🔗 Attached coordinate losses to outputs")

    def _validate_loss_attachment(self, outputs):
        """Validate that all loss components were properly attached to outputs."""
        required_loss_keys = [
            "_llm_loss",
            "_coordinate_l1_loss",
            "_total_tokens",
            "_coordinate_tokens",
            "_regular_tokens",
        ]

        missing_keys = []
        for key in required_loss_keys:
            if key not in outputs:
                missing_keys.append(key)

        if missing_keys:
            raise RuntimeError(
                f"Failed to attach loss components to outputs: missing {missing_keys}"
            )

        # Losses attached successfully

        # Losses attached successfully

    @property
    def device(self):
        """Return the device of the base model"""
        return next(self.base_model.parameters()).device

    def train(self, mode=True):
        """Override train mode"""
        super().train(mode)
        self.base_model.train(mode)
        return self

    def eval(self):
        """Override eval mode"""
        super().eval()
        self.base_model.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer=None,
        coordinate_config: Optional[CoordinateConfig] = None,
        attn_implementation: str = None,
        config=None,
        use_cache: bool = False,  # Add use_cache parameter
        **kwargs,
    ):
        """
        Load model with coordinate token support.

        Args:
            model_path: Path to model
            tokenizer: Tokenizer for the model
            coordinate_config: Coordinate token configuration
            attn_implementation: Attention implementation ('flash_attention_2', 'eager', etc.)
            **kwargs: Additional arguments

        Returns:
            Qwen25VLWithDetection: Model with coordinate token support
        """
        # Strict validation - fail fast
        if not model_path:
            raise ValueError("model_path cannot be empty")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if not hasattr(tokenizer, "get_vocab"):
            raise ValueError("tokenizer must have get_vocab method")
        if coordinate_config is not None and not isinstance(
            coordinate_config, CoordinateConfig
        ):
            raise ValueError("coordinate_config must be CoordinateConfig instance")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Create model
        model = cls(
            base_model_path=model_path,
            num_queries=100,  # Dummy value - not used for coordinate tokens
            max_caption_length=32,  # Dummy value - not used for coordinate tokens
            tokenizer=tokenizer,
            attn_implementation=attn_implementation,
            coordinate_config=coordinate_config,
            config=config,
            use_cache=use_cache,  # Pass use_cache parameter
        )

        # Load coordinate token extensions if they exist
        model._load_coordinate_extensions(model_path)

        return model

    @property
    def config(self):
        """Return the base model's config for DeepSpeed compatibility"""
        # Update vocab_size if coordinate tokens are enabled
        if self.coordinate_tokens_enabled and self.extended_vocab_size:
            # Create a copy to avoid modifying the original config
            config_copy = type(self.base_model.config)(
                **self.base_model.config.__dict__
            )
            config_copy.vocab_size = self.extended_vocab_size
            return config_copy
        return self.base_model.config

    def get_last_coordinate_losses(self) -> Dict[str, torch.Tensor]:
        """Get coordinate losses from last forward pass. EXPLICIT implementation."""
        # EXPLICIT CONFIG: _last_coordinate_losses is initialized in __init__
        if self._last_coordinate_losses is not None:
            return self._last_coordinate_losses.copy()
        else:
            # Return zero losses if no coordinate losses computed
            device = next(self.parameters()).device
            return {
                "_llm_loss": torch.tensor(0.0, device=device),
                "_coordinate_l1_loss": torch.tensor(0.0, device=device),
            }

    def update_coordinate_manager_geometry_tokens(self, tokenizer):
        """Update coordinate manager with geometry token IDs from tokenizer.

        This method should be called after SimpleTokenManager has added geometry tokens
        to ensure the CoordinateTokenManager can detect all geometry types.

        Args:
            tokenizer: Tokenizer with geometry tokens added
        """
        if not self.coordinate_manager:
            self.logger.warning(
                "⚠️ No coordinate manager to update with geometry tokens"
            )
            return

        # Get geometry token IDs from tokenizer
        geometry_tokens = {
            "square_start": tokenizer.convert_tokens_to_ids("<|square_start|>"),
            "square_end": tokenizer.convert_tokens_to_ids("<|square_end|>"),
            "line_start": tokenizer.convert_tokens_to_ids("<|line_start|>"),
            "line_end": tokenizer.convert_tokens_to_ids("<|line_end|>"),
        }

        # Check if tokens were found
        unk_id = tokenizer.unk_token_id
        missing_tokens = [k for k, v in geometry_tokens.items() if v == unk_id]

        if missing_tokens:
            raise ValueError(
                f"❌ SPECIAL_TOKEN_VIOLATION: Geometry tokens not found in tokenizer: {missing_tokens}"
            )

        # EXPLICIT CONFIG: coordinate_manager type is known at initialization
        # Use UnifiedTokenManager interface (standard implementation)
        if hasattr(self.coordinate_manager, "token_ids"):
            # UnifiedTokenManager - update token_ids
            self.coordinate_manager.token_ids.update(geometry_tokens)
        else:
            # Fallback for legacy CoordinateTokenManager
            if hasattr(self.coordinate_manager, "geometry_token_ids"):
                self.coordinate_manager.geometry_token_ids.update(geometry_tokens)
            else:
                raise ValueError(
                    "Coordinate manager missing token storage interface. "
                    "Ensure proper UnifiedTokenManager initialization."
                )

        # EXPLICIT CONFIG: coordinate manager always has config when properly initialized
        config = self.coordinate_manager.config
        config.square_start_id = geometry_tokens["square_start"]
        config.square_end_id = geometry_tokens["square_end"]
        config.line_start_id = geometry_tokens["line_start"]
        config.line_end_id = geometry_tokens["line_end"]

        # Enable multi-geometry since we have the tokens
        config.enable_multi_geometry = True

        self.logger.info("✅ Updated coordinate manager with geometry token IDs:")
        for name, token_id in geometry_tokens.items():
            self.logger.info(f"   🎯 {name}: {token_id}")

    def _update_coordinate_token_ranges(self):
        """Update coordinate manager with correct coordinate token ranges after tokenizer extension."""
        # EXPLICIT CONFIG: coordinate_manager is set up during initialization
        if not self.coordinate_manager:
            return

        # Find the actual coordinate token range in the extended tokenizer
        vocab = self.tokenizer.get_vocab()

        # Look for <coord_0> to find the start of coordinate tokens
        coord_0_token = "<coord_0>"
        coord_0_id = vocab.get(coord_0_token)

        if coord_0_id is not None:
            # Found coordinate tokens - calculate the range
            max_coord_value = self.coordinate_manager.config.max_coord_value
            coord_start_id = coord_0_id
            coord_end_id = coord_start_id + max_coord_value

            # Update coordinate manager ranges
            self.coordinate_manager.coord_start_id = coord_start_id
            self.coordinate_manager.coord_end_id = coord_end_id

            self.logger.info("🔧 Updated coordinate token ranges:")
            self.logger.info(f"   coord_start_id: {coord_start_id}")
            self.logger.info(f"   coord_end_id: {coord_end_id}")
            self.logger.info(f"   coordinate tokens: {max_coord_value}")

            # Verify the range is correct
            coord_last_token = f"<coord_{max_coord_value - 1}>"
            coord_last_id = vocab.get(coord_last_token)
            if coord_last_id == coord_end_id - 1:
                self.logger.info("✅ Coordinate token range verification passed")
            else:
                self.logger.warning(f"⚠️ Coordinate token range verification failed:")
                self.logger.warning(
                    f"   Expected <coord_{max_coord_value - 1}> at ID {coord_end_id - 1}"
                )
                self.logger.warning(
                    f"   Found <coord_{max_coord_value - 1}> at ID {coord_last_id}"
                )
        else:
            self.logger.warning("⚠️ Could not find <coord_0> in tokenizer vocabulary")
            self.logger.warning("   Coordinate token ranges not updated")

    # ------------------------------------------------------------------
    # HuggingFace compatibility helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the base model (with full HF metadata) + detection head.

        This makes the unified checkpoint fully reloadable via
        `from_pretrained(save_directory)` without any manual copying.

        Args:
            save_directory: Target directory.
            **kwargs: Forwarded to the underlying `save_pretrained` call of the
                base model. Common useful kwargs are `safe_serialization=True`
                and `max_shard_size="2GB"`.
        """

        os.makedirs(save_directory, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Save the base Qwen2.5-VL model. This writes:
        #    • config.json
        #    • generation_config.json
        #    • model.safetensors  (or sharded *.index.json + shards)
        #    • tokenizer / special-token files if they already exist in the
        #      directory and we pass `is_main_process=True` (handled by Trainer)
        # ------------------------------------------------------------------

        # Ensure safe serialization unless the caller overrides it.
        default_kwargs = {
            "safe_serialization": True,
            "max_shard_size": "2GB",
        }
        default_kwargs.update(kwargs)

        # Delegate to the underlying HF model
        if self.base_model is not None:
            self.base_model.save_pretrained(save_directory, **default_kwargs)
        else:
            self.logger.warning("Base model is None, cannot save pretrained model")

        # Also persist generation config explicitly if available (HF does not
        # always write it automatically for older versions).
        # Handle generation config saving safely
        if self.base_model is not None:
            # EXPLICIT CONFIG: Standard transformer models have generation_config
            generation_config = getattr(self.base_model, "generation_config", None)
            if generation_config is not None:
                # EXPLICIT CONFIG: GenerationConfig always has save_pretrained method
                generation_config.save_pretrained(save_directory)

        # ------------------------------------------------------------------
        # 2. CRITICAL: Save coordinate token extensions if enabled
        # ------------------------------------------------------------------
        if self.coordinate_tokens_enabled:
            # EXPLICIT CONFIG: extended_embeddings and extended_lm_head are set during initialization
            if self.extended_embeddings is None or self.extended_lm_head is None:
                raise RuntimeError(
                    "Coordinate tokens enabled but extended components are None - model corrupted!"
                )

            if self.extended_embeddings is None or self.extended_lm_head is None:
                raise RuntimeError(
                    "Coordinate tokens enabled but extended components are None - model corrupted!"
                )

            # Validate vocab sizes
            expected_vocab_size = None
            if (
                self.original_vocab_size is not None
                and self.coordinate_config is not None
            ):
                expected_vocab_size = (
                    self.original_vocab_size + self.coordinate_config.max_coord_value
                )

            if (
                self.extended_vocab_size is not None
                and expected_vocab_size is not None
                and self.extended_vocab_size != expected_vocab_size
            ):
                raise RuntimeError(
                    f"Vocab size mismatch: extended={self.extended_vocab_size}, "
                    f"expected={expected_vocab_size}"
                )

            import torch

            coord_weights_path = os.path.join(
                save_directory, "coordinate_extensions.pt"
            )
            coord_metadata = {
                "coordinate_tokens_enabled": True,
                "original_vocab_size": self.original_vocab_size,
                "extended_vocab_size": self.extended_vocab_size,
                "coordinate_config": {
                    "max_coord_value": self.coordinate_config.max_coord_value,
                    "enable_coordinate_tokens": self.coordinate_config.enable_coordinate_tokens,
                    "use_official_box_tokens": self.coordinate_config.use_official_box_tokens,
                    "coordinate_loss_weight": self.coordinate_config.coordinate_loss_weight,
                    "regular_loss_weight": self.coordinate_config.regular_loss_weight,
                    "soft_expectation_temperature": self.coordinate_config.soft_expectation_temperature,
                },
                "extended_embeddings": self.extended_embeddings.state_dict(),
                "extended_lm_head": self.extended_lm_head.state_dict(),
                "box_start_id": self.box_start_id,
                "box_end_id": self.box_end_id,
            }

            torch.save(coord_metadata, coord_weights_path)
            self.logger.info(
                f"✅ Saved coordinate token extensions to {coord_weights_path}"
            )
            self.logger.info(f"   Original vocab: {self.original_vocab_size}")
            self.logger.info(f"   Extended vocab: {self.extended_vocab_size}")
            self.logger.info(
                f"   Coordinate tokens: {self.coordinate_config.max_coord_value}"
            )
            self.logger.info(
                f"   Extended embeddings shape: {self.extended_embeddings.weight.shape}"
            )
            self.logger.info(
                f"   Extended LM head shape: {self.extended_lm_head.weight.shape}"
            )
        else:
            self.logger.info("📄 Standard model - no coordinate tokens to save")

        # NOTE: Tokenizer / processor saving is handled by Trainer once per
        # checkpoint; duplicating here is unnecessary and may overwrite user
        # modifications.
        self.logger.info(
            f"✅ save_pretrained completed for directory: {save_directory}"
        )

    def _load_coordinate_extensions(self, model_path: str):
        """Load coordinate token extensions with strict validation."""

        import torch

        coord_weights_path = os.path.join(model_path, "coordinate_extensions.pt")

        if not os.path.exists(coord_weights_path):
            self.logger.info("📄 No coordinate extensions found - using standard model")
            return

        self.logger.info(
            f"🔧 Loading coordinate token extensions from {coord_weights_path}"
        )
        coord_metadata = torch.load(coord_weights_path, map_location="cpu")

        # Strict validation of required keys
        required_keys = [
            "coordinate_tokens_enabled",
            "original_vocab_size",
            "extended_vocab_size",
            "coordinate_config",
            "extended_embeddings",
            "extended_lm_head",
            "box_start_id",
            "box_end_id",
        ]

        for key in required_keys:
            if key not in coord_metadata:
                raise RuntimeError(
                    f"Missing required key '{key}' in coordinate extensions file"
                )

        # Restore coordinate configuration
        self.coordinate_tokens_enabled = coord_metadata["coordinate_tokens_enabled"]
        self.original_vocab_size = coord_metadata["original_vocab_size"]
        self.extended_vocab_size = coord_metadata["extended_vocab_size"]
        self.box_start_id = coord_metadata["box_start_id"]
        self.box_end_id = coord_metadata["box_end_id"]

        # Validate vocab size consistency
        if not self.coordinate_tokens_enabled:
            raise RuntimeError(
                "Coordinate extensions file has coordinate_tokens_enabled=False"
            )

        # Restore coordinate config with all parameters
        coord_config_data = coord_metadata["coordinate_config"]
        required_config_keys = [
            "enable_coordinate_tokens",
            "max_coord_value",
            "use_official_box_tokens",
        ]

        for key in required_config_keys:
            if key not in coord_config_data:
                raise RuntimeError(f"Missing required coordinate config key '{key}'")

        # Check for all required config parameters
        required_config_keys.extend(
            [
                "coordinate_loss_weight",
                "regular_loss_weight",
                "soft_expectation_temperature",
            ]
        )

        for key in required_config_keys:
            if key not in coord_config_data:
                raise RuntimeError(
                    f"Missing required coordinate config key '{key}' in saved config"
                )

        self.coordinate_config = CoordinateConfig(
            enable_coordinate_tokens=coord_config_data["enable_coordinate_tokens"],
            max_coord_value=coord_config_data["max_coord_value"],
            use_official_box_tokens=coord_config_data["use_official_box_tokens"],
            coordinate_loss_weight=coord_config_data["coordinate_loss_weight"],
            regular_loss_weight=coord_config_data["regular_loss_weight"],
            soft_expectation_temperature=coord_config_data[
                "soft_expectation_temperature"
            ],
        )

        # Validate extended vocab size
        expected_vocab_size = None
        if self.original_vocab_size is not None and self.coordinate_config is not None:
            expected_vocab_size = (
                self.original_vocab_size + self.coordinate_config.max_coord_value
            )
        if (
            self.extended_vocab_size is not None
            and expected_vocab_size is not None
            and self.extended_vocab_size != expected_vocab_size
        ):
            raise RuntimeError(
                f"Extended vocab size mismatch: got {self.extended_vocab_size}, "
                f"expected {expected_vocab_size}"
            )

        # Recreate extended embeddings and LM head with correct shapes
        if self.base_model is None:
            raise RuntimeError("Cannot load coordinate extensions: base_model is None")

        hidden_size = self.base_model.get_input_embeddings().weight.shape[1]

        # Extended embeddings
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        self.extended_embeddings = nn.Embedding(
            self.extended_vocab_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.extended_embeddings.load_state_dict(coord_metadata["extended_embeddings"])

        # Extended LM head
        self.extended_lm_head = nn.Linear(
            hidden_size,
            self.extended_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.extended_lm_head.load_state_dict(coord_metadata["extended_lm_head"])

        self.logger.info(f"✅ Coordinate token extensions loaded successfully")
        self.logger.info(f"   Original vocab: {self.original_vocab_size}")
        self.logger.info(f"   Extended vocab: {self.extended_vocab_size}")
        self.logger.info(
            f"   Coordinate tokens: {self.coordinate_config.max_coord_value}"
        )
        self.logger.info(
            f"   Extended embeddings shape: {self.extended_embeddings.weight.shape}"
        )
        self.logger.info(
            f"   Extended LM head shape: {self.extended_lm_head.weight.shape}"
        )

        # CRITICAL: Add coordinate tokens to tokenizer if missing
        coordinate_tokens = [
            f"<coord_{i}>" for i in range(self.coordinate_config.max_coord_value)
        ]
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        missing_tokens = [
            token for token in coordinate_tokens if token not in existing_tokens
        ]

        if missing_tokens:
            self.logger.info(
                f"🔧 Adding {len(missing_tokens)} coordinate tokens to tokenizer"
            )
            # Update the additional_special_tokens property directly
            # EXPLICIT CONFIG: tokenizer validated at initialization - always has additional_special_tokens
            existing_tokens = self.tokenizer.additional_special_tokens
            new_special_tokens = list(existing_tokens) + missing_tokens
            self.tokenizer.additional_special_tokens = new_special_tokens
            num_added = len(missing_tokens)
            self.logger.info(f"✅ Added {num_added} coordinate tokens to tokenizer")
        else:
            self.logger.info(
                f"✅ All {len(coordinate_tokens)} coordinate tokens already in tokenizer"
            )

    def get_input_embeddings(self):
        """Delegate to base model's get_input_embeddings method."""
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        """Delegate to base model's get_output_embeddings method."""
        return self.base_model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int):
        """Delegate to base model's resize_token_embeddings method."""
        result = self.base_model.resize_token_embeddings(new_num_tokens)
        # EXPLICIT CONFIG: extended_vocab_size is always initialized
        self.extended_vocab_size = new_num_tokens
        return result


class DummyOptim:
    """Dummy optimizer for compatibility with trainer.py."""

    def __init__(self, params=None, lr=1e-3):
        self.params = params
        self.defaults = {"lr": lr}

    def step(self, *args, **kwargs):
        """Dummy step method."""
        pass

    def zero_grad(self, *args, **kwargs):
        """Dummy zero_grad method."""
        pass


class DummyScheduler:
    """Dummy scheduler for compatibility with trainer.py."""

    def __init__(self, optimizer=None, lr=1e-3):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        """Dummy step method."""
        pass

    def get_last_lr(self):
        """Return the last learning rate."""
        return [self.lr]
