"""
Qwen2.5-VL Model Wrapper with Detection Capabilities

This module provides a wrapper around the official Qwen2.5-VL model
that adds object detection capabilities while preserving all original functionality.
"""

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

from src.config import get_config
from src.logger_utils import get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes
from src.utils.coordinate_token_manager import (
    CoordinateTokenManager,
    create_coordinate_token_manager,
)


@dataclass
class CoordinateConfig:
    """Configuration for coordinate token extension."""

    max_coord_value: int = 2048
    coord_token_init_std: float = 0.01
    coordinate_loss_weight: float = 1.0
    regular_loss_weight: float = 1.0
    soft_expectation_temperature: float = 1.0
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    enable_coordinate_tokens: bool = False  # Feature flag
    use_official_box_tokens: bool = (
        True  # Always use official <|box_start|> and <|box_end|>
    )


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
        attn_implementation: str = None,
        coordinate_config: Optional[CoordinateConfig] = None,
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
        self.coordinate_config = coordinate_config or CoordinateConfig()

        # Coordinate token tracking
        self.coordinate_tokens_enabled = self.coordinate_config.enable_coordinate_tokens
        self.original_vocab_size = None
        self.extended_vocab_size = None
        self.extended_embeddings = None
        self.extended_lm_head = None

        # Unified coordinate token manager (includes loss computation)
        self.coordinate_manager: Optional[CoordinateTokenManager] = None

        # Coordinate loss tracking for logging - comprehensive initialization
        self._initialize_loss_tracking_components()

        # Ensure tracking components are always available
        self._ensure_loss_tracking_initialized()

        # Get config for model creation
        config = get_config()

        # Determine effective attention implementation
        effective_attn_impl = (
            attn_implementation
            if attn_implementation is not None
            else config.attn_implementation
        )

        # Load official Qwen2.5-VL model with proper configuration
        self.base_model: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=_get_torch_dtype(config.torch_dtype),
                attn_implementation=effective_attn_impl,
                device_map=None,  # Single GPU only - no multi-GPU device mapping
                trust_remote_code=True,
                use_cache=True,  # Enable KV cache for generation
            )
        )

        # CRITICAL: Move base model to GPU only if NOT using DeepSpeed
        import os
        deepspeed_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"

        if torch.cuda.is_available() and not deepspeed_enabled:
            self.base_model = self.base_model.to("cuda:0")
        elif deepspeed_enabled:
            self.logger.info("🔧 DeepSpeed enabled - letting DeepSpeed handle device placement")

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
            self._setup_coordinate_manager()
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

        # Move coordinate token components to same device as base model (only if not using DeepSpeed)
        if not deepspeed_enabled:
            device = next(self.base_model.parameters()).device
            # Move extended components to device if they exist
            if self.extended_embeddings is not None:
                self.extended_embeddings = self.extended_embeddings.to(device=device)
            if self.extended_lm_head is not None:
                self.extended_lm_head = self.extended_lm_head.to(device=device)
        else:
            self.logger.info("🔧 DeepSpeed enabled - coordinate tokens will be placed by DeepSpeed")

    def forward(
        self, **inputs: Any
    ) -> Union[Tuple[Any, ...], Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Forward pass that preserves all functionality and supports coordinate tokens.
        """
        # Store original ground truth objects for detection loss (don't pop them)
        # The trainer will handle detection loss computation

        # Remove ground truth objects and trainer-specific inputs from model inputs (but keep them in original inputs)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)
        model_inputs.pop("cu_seqlens", None)  # Remove Flash Attention 2 parameter
        model_inputs.pop("max_seqlen", None)  # Remove Flash Attention 2 parameter
        model_inputs.pop("teacher_assistant_spans", None)  # Remove teacher-student training parameter
        model_inputs.pop("student_assistant_spans", None)  # Remove teacher-student training parameter

        # Handle coordinate token processing if enabled
        if self.coordinate_tokens_enabled and "input_ids" in model_inputs:
            return self._forward_with_coordinate_tokens(model_inputs, inputs)
        else:
            # CRITICAL FIX: If coordinate tokens exist in vocab, we must use extended embeddings
            # even when coordinate token processing is disabled
            if (
                hasattr(self, "extended_embeddings")
                and self.extended_embeddings is not None
            ):
                # Use extended embeddings but disable coordinate-aware loss computation
                return self._forward_with_extended_embeddings_only(model_inputs, inputs)
            else:
                # Standard Qwen2.5-VL forward pass with all parameters preserved
                outputs = self.base_model(**model_inputs)
                # Still attach coordinate loss components for logging consistency
                outputs._focal_loss = self._last_focal_loss
                outputs._regular_loss = self._last_regular_loss
                outputs._l1_loss = self._last_l1_loss
                outputs._giou_loss = self._last_giou_loss
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

    def resize_token_embeddings(self, new_num_tokens):
        """Delegate to base model for token embedding resizing"""
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def _setup_coordinate_tokens(self):
        """Setup coordinate token support while preserving pretrained weights."""
        self.logger.info("🚀 Setting up coordinate token support...")

        # Calculate new vocabulary size (only coordinate tokens, reuse existing box tokens)
        num_new_tokens = (
            self.coordinate_config.max_coord_value
        )  # Only coordinate tokens
        self.extended_vocab_size = self.original_vocab_size + num_new_tokens

        # Store official box token IDs
        self.box_start_id = 151648  # <|box_start|>
        self.box_end_id = 151649  # <|box_end|>

        # Extend tokenizer with coordinate tokens
        self._extend_tokenizer()

        # Create extended embeddings and LM head
        self._create_extended_embeddings()
        self._create_extended_lm_head()

        self.logger.info(
            f"✅ Extended vocab from {self.original_vocab_size} to {self.extended_vocab_size}"
        )
        self.logger.info(
            f"✅ Added {num_new_tokens} coordinate tokens (reusing official box tokens)"
        )
        self.logger.info("✅ All pretrained weights preserved")

    def _setup_coordinate_manager(self):
        """Setup coordinate token manager and loss computer."""
        # Create coordinate token manager
        coordinate_config_dict = {
            "enable_coordinate_tokens": self.coordinate_tokens_enabled,
            "max_coord_value": self.coordinate_config.max_coord_value,
            "box_start_id": self.box_start_id,  # CRITICAL: Pass official box token IDs
            "box_end_id": self.box_end_id,      # CRITICAL: Pass official box token IDs
            "coordinate_loss_weight": self.coordinate_config.coordinate_loss_weight,
            "regular_loss_weight": self.coordinate_config.regular_loss_weight,
            "soft_expectation_temperature": self.coordinate_config.soft_expectation_temperature,
            "focal_loss_alpha": self.coordinate_config.focal_loss_alpha,
            "focal_loss_gamma": self.coordinate_config.focal_loss_gamma,
        }

        self.coordinate_manager = create_coordinate_token_manager(
            tokenizer=self.tokenizer,
            original_vocab_size=self.original_vocab_size,
            coordinate_config=coordinate_config_dict,
        )

        self.logger.info("✅ Unified coordinate token manager initialized")
        self.logger.info(f"   🎯 Box token IDs: start={self.box_start_id}, end={self.box_end_id}")
        self.logger.info(f"   🎯 Manager box token IDs: start={self.coordinate_manager.config.box_start_id}, end={self.coordinate_manager.config.box_end_id}")

    def _extend_tokenizer(self):
        """Add coordinate tokens to tokenizer (reuse existing box tokens)."""
        # Only add coordinate tokens - box tokens already exist
        coordinate_tokens = [
            f"<coord_{i}>" for i in range(self.coordinate_config.max_coord_value)
        ]

        # Get existing additional special tokens to preserve them
        existing_tokens = self.tokenizer.additional_special_tokens or []

        # Add coordinate tokens to the end
        all_additional_tokens = existing_tokens + coordinate_tokens

        _ = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": all_additional_tokens}
        )

        # Verify coordinate tokens are in the expected range
        first_coord_id = self.tokenizer.convert_tokens_to_ids("<coord_0>")
        expected_coord_start = self.original_vocab_size

        if first_coord_id != expected_coord_start:
            self.logger.warning(
                f"⚠️ Coordinate token ID mismatch: <coord_0> -> {first_coord_id}, expected {expected_coord_start}"
            )
            self.logger.warning("   This may cause index out of bounds errors")

        self.logger.info(
            f"✅ Added {len(coordinate_tokens)} coordinate tokens to tokenizer"
        )
        self.logger.info(f"   First coordinate token <coord_0> -> ID {first_coord_id}")
        self.logger.info(
            f"   Expected range: [{expected_coord_start}, {expected_coord_start + self.coordinate_config.max_coord_value})"
        )
        self.logger.info(
            f"✅ Reusing official box tokens: <|box_start|> ({self.box_start_id}), <|box_end|> ({self.box_end_id})"
        )

    def _create_extended_embeddings(self):
        """Create extended embeddings while preserving pretrained weights."""
        original_embeddings = self.base_model.get_input_embeddings()
        original_vocab_size = original_embeddings.weight.shape[0]
        hidden_size = original_embeddings.weight.shape[1]

        self.logger.info(f"🔧 Creating extended embeddings:")
        self.logger.info(f"   Original model vocab size: {original_vocab_size}")
        self.logger.info(f"   Tokenizer vocab size: {self.original_vocab_size}")
        self.logger.info(f"   Extended vocab size: {self.extended_vocab_size}")

        # Create new embedding layer with extended size
        self.extended_embeddings = nn.Embedding(
            self.extended_vocab_size,
            hidden_size,
            device=original_embeddings.weight.device,
            dtype=original_embeddings.weight.dtype,
        )

        # SEAMLESS LOADING: Copy ALL pretrained weights that exist
        # This ensures we preserve all pretrained embeddings regardless of tokenizer size
        with torch.no_grad():
            # Copy all pretrained weights
            pretrained_size = min(original_vocab_size, self.extended_vocab_size)
            self.extended_embeddings.weight[:pretrained_size].copy_(
                original_embeddings.weight[:pretrained_size]
            )

            # Initialize coordinate tokens only (starting from tokenizer vocab size)
            if self.original_vocab_size < self.extended_vocab_size:
                coordinate_start = self.original_vocab_size
                coordinate_end = self.extended_vocab_size

                self.logger.info(
                    f"   🎯 Initializing coordinate tokens [{coordinate_start}:{coordinate_end}]"
                )
                nn.init.normal_(
                    self.extended_embeddings.weight[coordinate_start:coordinate_end],
                    std=self.coordinate_config.coord_token_init_std,
                )

                # Make coordinate tokens trainable
                self.extended_embeddings.weight[
                    coordinate_start:coordinate_end
                ].requires_grad_(True)

        # Keep all embeddings jointly trainable for better BBU domain adaptation
        freeze_size = min(self.original_vocab_size, pretrained_size)
        self.extended_embeddings.weight[:freeze_size].requires_grad_(True)
        self.logger.info(
            f"   🔓 All embeddings jointly trainable for BBU domain adaptation"
        )

        self.logger.info(f"   ✅ Preserved {pretrained_size} pretrained embeddings")
        self.logger.info(
            f"   ✅ Initialized {self.extended_vocab_size - self.original_vocab_size} coordinate tokens"
        )

    def _create_extended_lm_head(self):
        """Create extended LM head while preserving pretrained weights."""
        original_lm_head = self.base_model.get_output_embeddings()
        original_vocab_size = original_lm_head.weight.shape[0]
        hidden_size = original_lm_head.weight.shape[1]

        self.logger.info(f"🔧 Creating extended LM head:")
        self.logger.info(f"   Original model vocab size: {original_vocab_size}")
        self.logger.info(f"   Tokenizer vocab size: {self.original_vocab_size}")
        self.logger.info(f"   Extended vocab size: {self.extended_vocab_size}")

        # Create new LM head with extended size
        self.extended_lm_head = nn.Linear(
            hidden_size,
            self.extended_vocab_size,
            bias=False,
            device=original_lm_head.weight.device,
            dtype=original_lm_head.weight.dtype,
        )

        # SEAMLESS LOADING: Copy ALL pretrained weights that exist
        # This ensures we preserve all pretrained projections regardless of tokenizer size
        with torch.no_grad():
            # Copy all pretrained weights
            pretrained_size = min(original_vocab_size, self.extended_vocab_size)
            self.extended_lm_head.weight[:pretrained_size].copy_(
                original_lm_head.weight[:pretrained_size]
            )

            # Initialize coordinate token projections only (starting from tokenizer vocab size)
            if self.original_vocab_size < self.extended_vocab_size:
                coordinate_start = self.original_vocab_size
                coordinate_end = self.extended_vocab_size

                self.logger.info(
                    f"   🎯 Initializing coordinate projections [{coordinate_start}:{coordinate_end}]"
                )
                nn.init.normal_(
                    self.extended_lm_head.weight[coordinate_start:coordinate_end],
                    std=self.coordinate_config.coord_token_init_std,
                )

                # Make coordinate projections trainable
                self.extended_lm_head.weight[
                    coordinate_start:coordinate_end
                ].requires_grad_(True)

        # Keep all LM head projections jointly trainable for better BBU domain adaptation
        freeze_size = min(self.original_vocab_size, pretrained_size)
        self.extended_lm_head.weight[:freeze_size].requires_grad_(True)
        self.logger.info(
            f"   🔓 All LM head projections jointly trainable for BBU domain adaptation"
        )

        self.logger.info(f"   ✅ Preserved {pretrained_size} pretrained projections")
        self.logger.info(
            f"   ✅ Initialized {self.extended_vocab_size - self.original_vocab_size} coordinate projections"
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

        # Replace input_ids with extended embeddings
        if input_ids is not None:
            inputs_embeds = self.extended_embeddings(input_ids)
            model_inputs["inputs_embeds"] = inputs_embeds
            # Keep input_ids for shape information but mark to use inputs_embeds

        # Forward through base model
        model_inputs["labels"] = None  # Remove labels to compute loss ourselves
        model_inputs["output_hidden_states"] = True  # Ensure we get hidden states
        model_inputs["return_dict"] = True  # Ensure we get a proper output object
        outputs = self.base_model(**model_inputs)

        # Use extended LM head - get hidden states from the last layer
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                f"Base model outputs missing hidden_states: {type(outputs)}"
            )
        if not isinstance(outputs.hidden_states, (list, tuple)):
            raise RuntimeError(
                f"Base model hidden_states should be list/tuple, got {type(outputs.hidden_states)}"
            )
        if len(outputs.hidden_states) == 0:
            raise RuntimeError("Base model hidden_states is empty")

        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        logits = self.extended_lm_head(hidden_states)

        # Compute standard cross entropy loss if labels provided
        loss = None
        labels = original_inputs.get("labels")
        if labels is not None:
            # Standard cross entropy loss (no coordinate-aware splitting)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

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

        # Replace input_ids with extended embeddings
        if input_ids is not None:
            inputs_embeds = self.extended_embeddings(input_ids)
            model_inputs["inputs_embeds"] = inputs_embeds
            # Keep input_ids for shape information but mark to use inputs_embeds
            # Some models need input_ids for position/attention mask calculation

        # Forward through base model (exclude output_hidden_states if not needed)
        model_inputs["labels"] = None  # Remove labels to compute loss ourselves
        model_inputs["output_hidden_states"] = True  # Ensure we get hidden states
        model_inputs["return_dict"] = True  # Ensure we get a proper output object
        outputs = self.base_model(**model_inputs)

        # Coordinate token processing enabled - compute coordinate-aware loss

        # Use extended LM head - get hidden states from the last layer
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                f"Base model outputs missing hidden_states: {type(outputs)}"
            )
        if not isinstance(outputs.hidden_states, (list, tuple)):
            raise RuntimeError(
                f"Base model hidden_states should be list/tuple, got {type(outputs.hidden_states)}"
            )
        if len(outputs.hidden_states) == 0:
            raise RuntimeError("Base model hidden_states is empty")

        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        logits = self.extended_lm_head(hidden_states)

        # Compute coordinate-aware loss if labels provided
        loss = None
        if labels is not None:
            self.logger.debug(f"🔍 DEBUGGING: Starting coordinate loss computation")
            self.logger.debug(f"   Labels shape: {labels.shape}")
            self.logger.debug(f"   Logits shape: {logits.shape}")

            if self.coordinate_manager is not None:
                self.logger.debug(f"   Using unified coordinate manager")
                # Detect bbox spans for coordinate loss computation
                bbox_spans = self.coordinate_manager.detect_bbox_spans(input_ids)
                coordinate_losses = self.coordinate_manager.compute_coordinate_losses(
                    logits, labels, bbox_spans
                )

                # Compute regular LLM loss for non-coordinate tokens
                regular_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )

                # Combine losses
                if coordinate_losses:
                    coordinate_loss = coordinate_losses.get("coordinate_loss", torch.tensor(0.0))
                    focal_loss = coordinate_losses.get("focal_loss", torch.tensor(0.0))
                    total_loss = (
                        self.coordinate_manager.config.coordinate_loss_weight * coordinate_loss +
                        self.coordinate_manager.config.regular_loss_weight * regular_loss +
                        focal_loss
                    )

                    # Extract all losses from coordinate manager
                    giou_loss = coordinate_losses.get("giou_loss", torch.tensor(0.0, device=logits.device))

                    loss_components = {
                        "coordinate_loss": coordinate_loss,
                        "focal_loss": focal_loss,
                        "regular_loss": regular_loss,
                        "total_loss": total_loss,
                        # Map coordinate_loss to l1_loss for internal tracking compatibility
                        "l1_loss": coordinate_loss,
                        "giou_loss": giou_loss,  # Now computed properly
                    }
                else:
                    loss_components = {
                        "regular_loss": regular_loss,
                        "total_loss": regular_loss,
                        # Default values for missing coordinate losses
                        "focal_loss": torch.tensor(0.0),
                        "l1_loss": torch.tensor(0.0),
                        "giou_loss": torch.tensor(0.0),
                    }

                loss = loss_components["total_loss"]

                # Log the computed loss components
                self.logger.debug(f"   Computed loss components: {loss_components}")
                self.logger.debug(f"   Total loss: {loss}")

                # Update loss tracking from components with validation
                self._update_loss_components_with_validation(loss_components)

                # IMMEDIATE VALIDATION: Ensure coordinate losses are non-zero when they should be
                focal_loss_val = loss_components.get("focal_loss", torch.tensor(0.0))
                l1_loss_val = loss_components.get("l1_loss", torch.tensor(0.0))
                giou_loss_val = loss_components.get("giou_loss", torch.tensor(0.0))

                # Convert tensors to float for comparison
                focal_val = focal_loss_val.item() if hasattr(focal_loss_val, 'item') else focal_loss_val
                l1_val = l1_loss_val.item() if hasattr(l1_loss_val, 'item') else l1_loss_val
                giou_val = giou_loss_val.item() if hasattr(giou_loss_val, 'item') else giou_loss_val

                total_coord_loss = focal_val + l1_val + giou_val

                if total_coord_loss == 0.0:
                    self.logger.error("❌ MODEL_WRAPPER: Coordinate loss computer returned zero losses!")
                    self.logger.error(f"   loss_components: {loss_components}")
                    self.logger.error(f"   This indicates coordinate loss computation failed in coordinate_loss_computer")
                    raise RuntimeError(
                        "Coordinate loss computer returned zero losses. "
                        "This indicates coordinate token processing failed at the computation level."
                    )
                else:
                    self.logger.debug(f"✅ MODEL_WRAPPER: Coordinate loss computer returned non-zero losses: {total_coord_loss}")

                # IMMEDIATE VALIDATION: Ensure internal tracking is updated correctly
                internal_total = (self._last_coordinate_loss + self._last_focal_loss +
                                self._last_l1_loss + self._last_giou_loss)

                if internal_total == 0.0:
                    self.logger.error("❌ MODEL_WRAPPER: Internal loss tracking failed!")
                    self.logger.error(f"   _last_coordinate_loss: {self._last_coordinate_loss}")
                    self.logger.error(f"   _last_focal_loss: {self._last_focal_loss}")
                    self.logger.error(f"   _last_l1_loss: {self._last_l1_loss}")
                    self.logger.error(f"   _last_giou_loss: {self._last_giou_loss}")
                    raise RuntimeError(
                        "Internal loss tracking failed - losses were computed but not stored correctly."
                    )
                else:
                    self.logger.debug(f"✅ MODEL_WRAPPER: Internal loss tracking updated correctly: {internal_total}")
            else:
                # Fallback to legacy loss computation
                self.logger.warning(f"⚠️ Coordinate loss computer not available, using legacy computation")
                loss = self._compute_coordinate_aware_loss(logits, labels)

            # IMMEDIATE ERROR CHECK: Ensure loss is a tensor
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError(
                    f"Loss computation returned {type(loss)}, expected torch.Tensor"
                )
            if loss.dim() != 0:
                raise RuntimeError(
                    f"Loss should be scalar tensor, got shape {loss.shape}"
                )
        else:
            self.logger.warning(f"⚠️ No labels provided for coordinate loss computation")

        # ALWAYS attach loss components to outputs for loss manager extraction
        self._attach_coordinate_losses_to_outputs(outputs)

        # Validate loss attachment was successful
        self._validate_loss_attachment(outputs)

        # IMMEDIATE ERROR CHECK: Ensure loss is still a tensor before setting
        if loss is not None and not isinstance(loss, torch.Tensor):
            raise RuntimeError(
                f"Loss variable corrupted to {type(loss)}, expected torch.Tensor"
            )

        # IMMEDIATE ERROR CHECK: Ensure outputs object is proper type
        if not hasattr(outputs, "loss"):
            raise RuntimeError(f"Outputs object {type(outputs)} missing loss attribute")

        # CRITICAL FIX: Create a new output object to avoid corruption
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLCausalLMOutputWithPast,
        )

        # Create new output object with correct values
        new_outputs = Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            past_key_values=getattr(outputs, "past_key_values", None),
            attentions=getattr(outputs, "attentions", None),
        )

        # CRITICAL FIX: Attach coordinate losses to the NEW output object
        # This was the root cause - losses were attached to old outputs but new outputs was returned
        self._attach_coordinate_losses_to_outputs(new_outputs)

        # IMMEDIATE VALIDATION: Ensure coordinate losses are properly attached to new output
        config = get_config()
        if hasattr(config, "coordinate_tokens_enabled") and config.coordinate_tokens_enabled:
            required_attrs = ["_coordinate_loss", "_focal_loss", "_l1_loss", "_giou_loss"]
            missing_attrs = []

            for attr in required_attrs:
                if not hasattr(new_outputs, attr):
                    missing_attrs.append(attr)

            if missing_attrs:
                raise RuntimeError(
                    f"CRITICAL: New output object missing coordinate loss attributes: {missing_attrs}. "
                    f"This indicates loss attachment to new output object failed."
                )

            # Validate the attached losses are non-zero
            attached_total = (new_outputs._coordinate_loss + new_outputs._focal_loss +
                            new_outputs._l1_loss + new_outputs._giou_loss)

            if attached_total == 0.0:
                self.logger.error("❌ MODEL_WRAPPER: Coordinate losses attached to new output are zero!")
                self.logger.error(f"   new_outputs._coordinate_loss: {new_outputs._coordinate_loss}")
                self.logger.error(f"   new_outputs._focal_loss: {new_outputs._focal_loss}")
                self.logger.error(f"   new_outputs._l1_loss: {new_outputs._l1_loss}")
                self.logger.error(f"   new_outputs._giou_loss: {new_outputs._giou_loss}")
                raise RuntimeError(
                    "Coordinate losses attached to new output object are zero. "
                    "This indicates loss attachment failed."
                )
            else:
                self.logger.debug(f"✅ MODEL_WRAPPER: Coordinate losses attached to new output: {attached_total}")

        # IMMEDIATE ERROR CHECK: Verify new outputs.loss is correct
        if loss is not None and not isinstance(new_outputs.loss, torch.Tensor):
            raise RuntimeError(
                f"CRITICAL: new_outputs.loss corrupted to {type(new_outputs.loss)}, expected torch.Tensor"
            )

        return new_outputs

    def _update_loss_components_with_validation(self, loss_components: Dict[str, float]):
        """Update loss tracking components with enhanced validation."""
        self._ensure_loss_tracking_initialized()

        # Validate that all required loss components are present
        required_components = [
            "coordinate_loss", "focal_loss", "regular_loss", "l1_loss", "giou_loss"
        ]

        missing_components = []
        for component in required_components:
            if component not in loss_components:
                missing_components.append(component)

        if missing_components:
            self.logger.warning(
                f"⚠️ Missing loss components: {missing_components}. Using defaults."
            )

        # Update components with validation and defaults
        self._last_coordinate_loss = self._validate_loss_value(
            loss_components.get("coordinate_loss", 0.0), "coordinate_loss"
        )
        self._last_focal_loss = self._validate_loss_value(
            loss_components.get("focal_loss", 0.0), "focal_loss"
        )
        self._last_regular_loss = self._validate_loss_value(
            loss_components.get("regular_loss", 0.0), "regular_loss"
        )
        self._last_l1_loss = self._validate_loss_value(
            loss_components.get("l1_loss", 0.0), "l1_loss"
        )
        self._last_giou_loss = self._validate_loss_value(
            loss_components.get("giou_loss", 0.0), "giou_loss"
        )

        # Enhanced tracking metrics with validation
        self._last_detection_loss = self._validate_loss_value(
            loss_components.get("detection_loss", 0.0), "detection_loss"
        )
        self._last_total_tokens = max(0, int(loss_components.get("total_tokens", 0)))
        self._last_coordinate_tokens = max(0, int(loss_components.get("coordinate_tokens", 0)))
        self._last_regular_tokens = max(0, int(loss_components.get("regular_tokens", 0)))

        # Validate token counts consistency
        expected_total = self._last_coordinate_tokens + self._last_regular_tokens
        if self._last_total_tokens > 0 and expected_total > 0:
            if abs(self._last_total_tokens - expected_total) > 1:  # Allow small rounding differences
                self.logger.debug(
                    f"⚠️ Token count mismatch: total={self._last_total_tokens}, "
                    f"coord+regular={expected_total}"
                )

        self.logger.debug(f"✅ Updated loss components from coordinate loss computer")

    def _validate_loss_value(self, value: float, component_name: str) -> float:
        """Validate and sanitize a loss value - handles tensors and scalars."""
        # Handle PyTorch tensors by extracting scalar value
        if hasattr(value, 'item'):
            try:
                scalar_value = value.item()
            except (ValueError, RuntimeError):
                self.logger.warning(f"⚠️ Cannot extract scalar from {component_name} tensor: {value}, using 0.0")
                return 0.0
        elif isinstance(value, (int, float)):
            scalar_value = float(value)
        else:
            self.logger.warning(f"⚠️ Invalid {component_name} type: {type(value)}, using 0.0")
            return 0.0

        # Check for NaN or infinite values
        if torch.isnan(torch.tensor(scalar_value)) or torch.isinf(torch.tensor(scalar_value)):
            self.logger.warning(f"⚠️ Invalid {component_name} value: {scalar_value}, using 0.0")
            return 0.0

        # Clamp to reasonable bounds to prevent extreme values
        if abs(scalar_value) > 1000.0:
            self.logger.warning(f"⚠️ Extreme {component_name} value: {scalar_value}, clamping")
            return max(-1000.0, min(1000.0, scalar_value))

        return scalar_value

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
                self._last_regular_loss = regular_loss.item()

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
            self._last_focal_loss = focal_loss.item()

            # Extract L1 and GIoU components separately for detailed logging
            if coord_logits.size(0) > 0:
                l1_loss, giou_loss = self._compute_advanced_detection_losses(
                    self._extract_expected_coordinates(coord_logits),
                    coord_labels.float() - self.original_vocab_size,
                )
                self._last_l1_loss = l1_loss.item()
                self._last_giou_loss = giou_loss.item()
                self.logger.debug(f"      L1 loss: {l1_loss.item():.6f}")
                self.logger.debug(f"      GIoU loss: {giou_loss.item():.6f}")
        else:
            self.logger.debug(f"   ⚠️ No coordinate tokens found in this batch")

        return total_loss

    def _get_coordinate_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for coordinate tokens."""
        coord_start = self.original_vocab_size  # Start immediately after original vocab
        coord_end = coord_start + self.coordinate_config.max_coord_value
        return (token_ids >= coord_start) & (token_ids < coord_end)

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
        """Compute enhanced coordinate loss with L1 and GIoU components."""
        # Extract coordinate portion of logits
        coord_start = self.original_vocab_size
        coord_end = coord_start + self.coordinate_config.max_coord_value
        coord_only_logits = coord_logits[:, coord_start:coord_end]

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

        # ENHANCED: Compute advanced detection losses
        l1_loss, giou_loss = self._compute_advanced_detection_losses(
            expected_coords, coord_indices.float()
        )

        # Focal loss on distribution for sharpness
        focal_loss = self._compute_focal_loss_on_distribution(
            soft_weights, coord_indices
        )

        # Combine L1 and GIoU for better bbox regression
        combined_detection_loss = l1_loss + 0.5 * giou_loss

        # CRITICAL: Average the losses per coordinate token to prevent extremely large losses
        # The current implementation sums losses across all coordinate tokens, making them huge
        num_coord_tokens = coord_logits.size(0)
        if num_coord_tokens > 0:
            combined_detection_loss = combined_detection_loss / num_coord_tokens
            focal_loss = focal_loss / num_coord_tokens

        return combined_detection_loss, focal_loss  # Return enhanced components

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
        """Extract expected coordinates from coordinate logits for loss computation."""
        # Extract coordinate portion of logits
        coord_start = self.original_vocab_size
        coord_end = coord_start + self.coordinate_config.max_coord_value
        coord_only_logits = coord_logits[:, coord_start:coord_end]

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

        return expected_coords

    def _compute_focal_loss_on_distribution(
        self, soft_weights: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss to encourage sharp distributions."""
        # Focal loss computation using cross-entropy (not binary cross-entropy)
        alpha = self.coordinate_config.focal_loss_alpha
        gamma = self.coordinate_config.focal_loss_gamma

        # Get target probabilities from soft distribution
        target_probs = soft_weights.gather(1, targets.unsqueeze(-1)).squeeze(-1)

        # Compute focal weight: alpha * (1 - p_t)^gamma
        focal_weight = alpha * (1 - target_probs) ** gamma

        # Compute cross-entropy loss: -log(p_t)
        ce_loss = -torch.log(target_probs + 1e-8)  # Add epsilon for numerical stability

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()

    def get_coordinate_tokenizer_utils(self):
        """Get utility functions for coordinate token conversion."""
        if not self.coordinate_tokens_enabled:
            return None

        return {
            "convert_bbox_to_tokens": self._convert_bbox_to_tokens,
            "convert_tokens_to_bbox": self._convert_tokens_to_bbox,
            "coord_start_id": self.original_vocab_size,
            "coord_end_id": self.original_vocab_size
            + self.coordinate_config.max_coord_value,
            "box_start_id": self.box_start_id,  # 151648
            "box_end_id": self.box_end_id,  # 151649
            "max_coord_value": self.coordinate_config.max_coord_value,  # 2048
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
                    token_id < coord_start
                    or token_id >= coord_start + self.coordinate_config.max_coord_value
                ):
                    return None

                # Direct mapping: token ID -> integer coordinate
                coord_idx = token_id - coord_start
                bbox.append(coord_idx)

            return bbox

        except (ValueError, IndexError):
            return None

    def _initialize_loss_tracking_components(self):
        """Initialize all loss tracking components to ensure they always exist."""
        # Primary coordinate loss components
        self._last_coordinate_loss = 0.0
        self._last_focal_loss = 0.0
        self._last_regular_loss = 0.0
        self._last_l1_loss = 0.0
        self._last_giou_loss = 0.0

        # Enhanced tracking attributes
        self._last_detection_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        self.logger.debug(f"⚙️ Initialized loss tracking components")

    def _ensure_loss_tracking_initialized(self):
        """Ensure all loss tracking components exist with defensive programming."""
        # Defensive initialization - ensure all attributes exist
        if not hasattr(self, '_last_coordinate_loss'):
            self._last_coordinate_loss = 0.0
        if not hasattr(self, '_last_focal_loss'):
            self._last_focal_loss = 0.0
        if not hasattr(self, '_last_regular_loss'):
            self._last_regular_loss = 0.0
        if not hasattr(self, '_last_l1_loss'):
            self._last_l1_loss = 0.0
        if not hasattr(self, '_last_giou_loss'):
            self._last_giou_loss = 0.0
        if not hasattr(self, '_last_detection_loss'):
            self._last_detection_loss = 0.0
        if not hasattr(self, '_last_total_tokens'):
            self._last_total_tokens = 0
        if not hasattr(self, '_last_coordinate_tokens'):
            self._last_coordinate_tokens = 0
        if not hasattr(self, '_last_regular_tokens'):
            self._last_regular_tokens = 0

    def _reset_loss_components_for_forward_pass(self):
        """Reset loss components at the start of each forward pass."""
        self._ensure_loss_tracking_initialized()

        # Reset to defaults for new forward pass
        self._last_coordinate_loss = 0.0
        self._last_focal_loss = 0.0
        self._last_regular_loss = 0.0
        self._last_l1_loss = 0.0
        self._last_giou_loss = 0.0
        self._last_detection_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        self.logger.debug(f"🔄 Reset loss components for forward pass")

    def _reset_loss_components_to_zero(self):
        """Reset all loss components to zero for standard LLM mode."""
        self._ensure_loss_tracking_initialized()

        # Set all to zero for standard mode
        self._last_coordinate_loss = 0.0
        self._last_focal_loss = 0.0
        self._last_regular_loss = 0.0
        self._last_l1_loss = 0.0
        self._last_giou_loss = 0.0
        self._last_detection_loss = 0.0
        self._last_total_tokens = 0
        self._last_coordinate_tokens = 0
        self._last_regular_tokens = 0

        self.logger.debug(f"📄 Reset loss components to zero (standard LLM mode)")

    def _attach_coordinate_losses_to_outputs(self, outputs):
        """Attach all coordinate loss components to model outputs for loss manager extraction."""
        self._ensure_loss_tracking_initialized()

        # Attach individual loss components
        outputs._coordinate_loss = self._last_coordinate_loss
        outputs._focal_loss = self._last_focal_loss
        outputs._regular_loss = self._last_regular_loss
        outputs._l1_loss = self._last_l1_loss
        outputs._giou_loss = self._last_giou_loss

        # Enhanced metrics
        outputs._detection_loss = self._last_detection_loss
        outputs._total_tokens = self._last_total_tokens
        outputs._coordinate_tokens = self._last_coordinate_tokens
        outputs._regular_tokens = self._last_regular_tokens

        self.logger.debug(f"🔗 Attached coordinate losses to outputs")

    def _validate_loss_attachment(self, outputs):
        """Validate that all loss components were properly attached to outputs."""
        required_loss_attrs = [
            '_focal_loss', '_regular_loss',
            '_l1_loss', '_giou_loss', '_detection_loss',
            '_total_tokens', '_coordinate_tokens', '_regular_tokens'
        ]

        missing_attrs = []
        for attr in required_loss_attrs:
            if not hasattr(outputs, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            raise RuntimeError(
                f"Failed to attach loss components to outputs: missing {missing_attrs}"
            )

        # Log validation summary
        self.logger.debug(f"✅ Loss attachment validated: {len(required_loss_attrs)} components attached")
        self.logger.debug(f"   focal_loss: {self._last_focal_loss:.6f}")
        self.logger.debug(f"   regular_loss: {self._last_regular_loss:.6f}")
        self.logger.debug(f"   l1_loss: {self._last_l1_loss:.6f}")
        self.logger.debug(f"   giou_loss: {self._last_giou_loss:.6f}")
        self.logger.debug(f"   token counts: total={self._last_total_tokens}, coord={self._last_coordinate_tokens}, regular={self._last_regular_tokens}")

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

        import os

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

        import os

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
        self.base_model.save_pretrained(save_directory, **default_kwargs)

        # Also persist generation config explicitly if available (HF does not
        # always write it automatically for older versions).
        if getattr(self.base_model, "generation_config", None) is not None:
            self.base_model.generation_config.save_pretrained(save_directory)

        # ------------------------------------------------------------------
        # 2. CRITICAL: Save coordinate token extensions if enabled
        # ------------------------------------------------------------------
        if self.coordinate_tokens_enabled:
            if not hasattr(self, "extended_embeddings") or not hasattr(
                self, "extended_lm_head"
            ):
                raise RuntimeError(
                    "Coordinate tokens enabled but extended embeddings/lm_head not found - model corrupted!"
                )

            if self.extended_embeddings is None or self.extended_lm_head is None:
                raise RuntimeError(
                    "Coordinate tokens enabled but extended components are None - model corrupted!"
                )

            # Validate vocab sizes
            if (
                self.extended_vocab_size
                != self.original_vocab_size + self.coordinate_config.max_coord_value
            ):
                raise RuntimeError(
                    f"Vocab size mismatch: extended={self.extended_vocab_size}, "
                    f"expected={self.original_vocab_size + self.coordinate_config.max_coord_value}"
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
        import os

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
        expected_vocab_size = (
            self.original_vocab_size + self.coordinate_config.max_coord_value
        )
        if self.extended_vocab_size != expected_vocab_size:
            raise RuntimeError(
                f"Extended vocab size mismatch: got {self.extended_vocab_size}, "
                f"expected {expected_vocab_size}"
            )

        # Recreate extended embeddings and LM head with correct shapes
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
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": missing_tokens}
            )
            self.logger.info(f"✅ Added {num_added} coordinate tokens to tokenizer")
        else:
            self.logger.info(
                f"✅ All {len(coordinate_tokens)} coordinate tokens already in tokenizer"
            )
