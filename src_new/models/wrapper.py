"""
Detection Model Wrapper for New Qwen2.5-VL Architecture

This module implements a composition-based model wrapper that integrates coordinate token
support with the base Qwen2.5-VL model. It focuses on clean separation of concerns and
compatibility with HuggingFace Trainer.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


if TYPE_CHECKING:
    from src_new.config.config import Config

from .loss_manager import LossComponents, LossManager, ModelOutput
from .patches import apply_comprehensive_qwen25_fixes


class CoordinateProcessor:
    """
    Coordinate token processing utilities.

    Handles coordinate token masking, vocabulary extension, and coordinate-specific
    operations for the detection model.
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize coordinate processor.

        Args:
            config: Configuration object with coordinate settings
        """
        self.config = config
        self.coordinate_tokens_enabled = config.coordinate_tokens_enabled
        self.max_coord_value = config.max_coord_value
        self.original_vocab_size = None  # Will be set when tokenizer is available

        # Initialize coordinate token range as None - will be set when tokenizer is available
        self.coordinate_token_range = None
        self._tokenizer = None

    def set_tokenizer(self, tokenizer) -> None:
        """
        Set tokenizer and update coordinate token range.

        Args:
            tokenizer: Tokenizer to use for coordinate token detection
        """
        self._tokenizer = tokenizer
        if tokenizer is not None:
            vocab = tokenizer.get_vocab()

            # Determine original vocab size
            if self.coordinate_tokens_enabled:
                # Look for coordinate value tokens (not geometry tokens)
                # Coordinate tokens are <|coord_0|> through <|coord_2048|>
                coord_value_tokens = [
                    token for token in vocab.keys() if token.startswith("<|coord_")
                ]

                if coord_value_tokens:
                    # Original vocab size is the minimum coordinate token ID
                    coord_ids = [vocab[token] for token in coord_value_tokens]
                    self.original_vocab_size = min(coord_ids)
                    self.coordinate_token_range = (min(coord_ids), max(coord_ids) + 1)
                    print(f"🎯 Found {len(coord_value_tokens)} coordinate value tokens")
                    print(f"🎯 Original vocab size: {self.original_vocab_size}")
                    print(f"🎯 Coordinate token range: {self.coordinate_token_range}")
                    print(
                        f"🎯 Sample tokens: {sorted(coord_value_tokens)[:3]} ... {sorted(coord_value_tokens)[-3:]}"
                    )
                else:
                    # No coordinate tokens found - check if they were expected
                    if self.coordinate_tokens_enabled:
                        raise ValueError(
                            "❌ CRITICAL: Coordinate tokens are enabled in configuration but no coordinate tokens "
                            f"(e.g., '<|coord_0|>', '<|coord_1|>', etc.) were found in the tokenizer vocabulary. "
                            f"This indicates that the tokenizer vocabulary was not properly extended with coordinate tokens. "
                            f"Please ensure that coordinate tokens are added to the tokenizer before model initialization."
                        )

                    # Coordinate tokens not expected - use full vocab size as original
                    self.original_vocab_size = len(vocab)
                    print(
                        "ℹ️ No coordinate tokens found in vocabulary (coordinate processing disabled in config)"
                    )
                    self.coordinate_tokens_enabled = False
                    self.coordinate_token_range = (0, 0)
            else:
                # Coordinate tokens disabled - use full vocab size as original
                self.original_vocab_size = len(vocab)
                self.coordinate_token_range = (0, 0)
        else:
            self.coordinate_token_range = (0, 0)

    def mask_coordinate_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        coordinate_token_range: tuple[int, int],
    ) -> torch.Tensor:
        """
        Mask coordinate-specific logits for non-coordinate positions.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]
            coordinate_token_range: Tuple of (start_idx, end_idx) for coordinate tokens

        Returns:
            Masked logits tensor
        """
        if not self.coordinate_tokens_enabled or self.coordinate_token_range is None:
            return logits

        # Use the actual coordinate token range from tokenizer
        start_idx, end_idx = self.coordinate_token_range

        # Check if the range is valid
        if start_idx >= end_idx or end_idx > logits.size(-1):
            # Invalid range - return original logits
            return logits

        # Create coordinate mask
        coord_mask = self.get_coordinate_mask(input_ids)

        # Mask coordinate token logits for non-coordinate positions
        masked_logits = logits.clone()

        # Set coordinate token logits to very negative values where not expected
        non_coord_positions = ~coord_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Work on the vocabulary slice separately to keep indexing simple and
        # avoid type-checker complaints about complex tuple indices.
        vocab_slice = masked_logits[..., start_idx:end_idx]

        vocab_slice = torch.where(
            non_coord_positions.expand_as(vocab_slice),
            torch.full_like(vocab_slice, -1e9),
            vocab_slice,
        )

        # Write the processed slice back into the logits tensor
        masked_logits[..., start_idx:end_idx] = vocab_slice

        return masked_logits

    def get_coordinate_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get coordinate token mask from input IDs.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Boolean mask for coordinate tokens [batch_size, seq_len]
        """
        if not self.coordinate_tokens_enabled or self.coordinate_token_range is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)

        # Create mask for coordinate value tokens based on the actual token range
        start_idx, end_idx = self.coordinate_token_range
        coord_mask = (input_ids >= start_idx) & (input_ids < end_idx)

        return coord_mask


class DetectionModel(nn.Module):
    """
    Detection model wrapper for Qwen2.5-VL.

    This wrapper adds coordinate token support to the base model using composition
    rather than inheritance. This approach provides better separation of concerns
    and more maintainable code.
    """

    # Declare tied weights to handle shared memory during checkpoint saving
    _tied_weights_keys = ["base_model.lm_head.weight"]

    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        config: "Config",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        """
        Initialize detection model.

        Args:
            base_model: Base Qwen2.5-VL model
            config: Configuration object with model settings
            tokenizer: Tokenizer for text processing (optional)
        """
        super().__init__()
        self.base_model = base_model
        # Store custom config separately to avoid HuggingFace trainer conflicts
        self.training_config = config
        # Ensure model.config points to the HuggingFace config (required by trainer)
        self.config = base_model.config
        self.tokenizer = tokenizer

        # Initialize coordinate mode first
        self._coordinate_mode = config.coordinate_tokens_enabled

        # Initialize coordinate processor
        self.coordinate_processor = CoordinateProcessor(config)

        # Initialize token processor for vocabulary extension
        from src_new.processing.token_processor import TokenConfig, TokenProcessor

        token_config = TokenConfig(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
        )
        self.token_processor = TokenProcessor(token_config)

        # Store tokenizer reference for coordinate processing
        self._tokenizer = tokenizer

        # Extend tokenizer vocabulary and model embeddings FIRST
        if tokenizer is not None and self._coordinate_mode:
            from src_new.config.config import logger

            logger.info("🔧 Extending tokenizer and model for coordinate tokens...")
            tokenizer = self.token_processor.extend_tokenizer_vocabulary(tokenizer)
            self.token_processor.extend_model_embeddings(base_model, tokenizer)
            logger.info("✅ Tokenizer and model extension completed")

        # Set tokenizer for coordinate processor AFTER extension
        if tokenizer is not None:
            self.coordinate_processor.set_tokenizer(tokenizer)

        # Initialize loss manager with token processor for correct coordinate token IDs
        self.loss_manager = LossManager(
            config, token_processor=self.token_processor, tokenizer=tokenizer
        )

        # Extend model embeddings if coordinate tokens are enabled and tokenizer provided
        if self._coordinate_mode and tokenizer is not None:
            original_vocab_size = base_model.config.vocab_size
            extended_vocab_size = len(tokenizer.get_vocab())

            if extended_vocab_size > original_vocab_size:
                print(
                    f"🔧 Extending model embeddings from {original_vocab_size} to {extended_vocab_size}"
                )
                self.resize_token_embeddings(extended_vocab_size)
                print(f"✅ Model embeddings extended successfully")

        # Disable cache during training for better performance
        self.base_model.config.use_cache = config.use_cache

        # Apply any necessary patches
        apply_comprehensive_qwen25_fixes()

        # Ensure tied weights are properly set up
        self.tie_weights()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: "Config",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ) -> "DetectionModel":
        """
        Create detection model from pretrained model.

        Args:
            model_path: Path to pretrained model
            config: Configuration object with model settings
            tokenizer: Tokenizer for text processing (optional)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            DetectionModel instance
        """
        # Load base model with specified dtype
        torch_dtype = getattr(torch, config.torch_dtype)
        attn_implementation = config.attn_implementation

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=False,
            **kwargs,
        )

        # Create detection model
        return cls(
            base_model=base_model,
            config=config,
            tokenizer=tokenizer,
        )

    def forward(self, **kwargs) -> Union["ModelOutput", Any]:
        """
        Forward pass with coordinate token support.

        Args:
            **kwargs: Arguments for base model forward pass

        Returns:
            Model output with loss components
        """
        # Extract inputs
        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels")

        # CRITICAL FIX: Ensure image_grid_thw has correct shape before passing to base model
        if "image_grid_thw" in kwargs:
            from src_new.config.config import logger

            image_grid_thw = kwargs["image_grid_thw"]
            if image_grid_thw is not None:
                logger.debug(
                    f"🔍 BEFORE FIX: image_grid_thw shape: {image_grid_thw.shape}, values: {image_grid_thw}"
                )

                # Fix tensor shape if needed: [batch_size, 1, 3] -> [batch_size, 3]
                if image_grid_thw.dim() == 3 and image_grid_thw.shape[1] == 1:
                    image_grid_thw = image_grid_thw.squeeze(
                        1
                    )  # [batch_size, 1, 3] -> [batch_size, 3]
                    kwargs["image_grid_thw"] = image_grid_thw
                    logger.debug(
                        f"✅ FIXED image_grid_thw shape: {image_grid_thw.shape}"
                    )

                # Also handle case where it's [batch_size, 2] instead of [batch_size, 3]
                if image_grid_thw.dim() == 2 and image_grid_thw.shape[1] == 2:
                    logger.error(
                        f"❌ CRITICAL: image_grid_thw has only 2 values per batch item: {image_grid_thw.shape}"
                    )
                    logger.error(f"   Values: {image_grid_thw}")
                    raise ValueError(
                        f"image_grid_thw has shape {image_grid_thw.shape} with only 2 values per item, but model expects 3 (t, h, w). "
                        f"This indicates a fundamental issue in the data pipeline where grid_thw is missing one dimension."
                    )

                # Validate final shape
                if image_grid_thw.dim() == 2 and image_grid_thw.shape[1] == 3:
                    logger.debug(
                        f"✅ image_grid_thw shape validated: {image_grid_thw.shape}"
                    )
                else:
                    logger.error(
                        f"❌ Invalid image_grid_thw shape: {image_grid_thw.shape}"
                    )
                    raise ValueError(
                        f"image_grid_thw must have shape [batch_size, 3], got {image_grid_thw.shape}. "
                        f"This indicates a tensor shape issue in the data pipeline."
                    )

        # Filter out HuggingFace Trainer-specific arguments that base model doesn't accept
        excluded_args = [
            "num_items_in_batch",
            "teacher_assistant_spans",
            "student_assistant_spans",
        ]
        base_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_args}

        # Forward pass through base model
        base_outputs = self.base_model(**base_kwargs)

        # Process outputs based on mode
        if self._coordinate_mode and input_ids is not None and labels is not None:
            # Debug logging for coordinate mode
            from src_new.config.config import logger

            logger.debug(
                f"🎯 Using coordinate mode: coordinate_tokens_enabled={self.coordinate_processor.coordinate_tokens_enabled}"
            )
            logger.debug(
                f"🎯 Coordinate token range: {self.coordinate_processor.coordinate_token_range}"
            )

            # Apply coordinate token processing
            return self._forward_with_coordinate_loss(
                base_outputs, input_ids, labels, kwargs
            )
        else:
            # Debug logging for standard mode
            from src_new.config.config import logger

            logger.debug(
                f"🔍 Using standard mode: coordinate_mode={self._coordinate_mode}, input_ids={input_ids is not None}, labels={labels is not None}"
            )

            # Standard forward pass - create detailed loss components for better logging
            if (
                hasattr(base_outputs, "loss")
                and base_outputs.loss is not None
                and input_ids is not None
                and labels is not None
            ):
                # Extract teacher and student spans from kwargs
                teacher_spans = kwargs.get("teacher_assistant_spans", None)
                student_spans = kwargs.get("student_assistant_spans", None)

                # Use the loss manager to compute detailed components even without coordinate tokens
                loss_components = self.loss_manager.compute_loss_components(
                    logits=base_outputs.logits,
                    labels=labels,
                    coord_mask=None,  # No coordinate mask in standard mode
                    teacher_spans=teacher_spans,
                    student_spans=student_spans,
                )
                # Store for callback access
                self.loss_manager.last_loss_components = loss_components
            elif hasattr(base_outputs, "loss") and base_outputs.loss is not None:
                # Fallback for cases without input_ids/labels
                loss_components = LossComponents(
                    loss=base_outputs.loss,
                    llm_loss=base_outputs.loss,
                )
                # Store for callback access
                self.loss_manager.last_loss_components = loss_components
            else:
                # No loss available
                loss_components = LossComponents(loss=None)
                self.loss_manager.last_loss_components = loss_components

            # Return simple dict-like object for DataParallel compatibility
            # The trainer expects either a dict with 'loss' key or an object with loss attribute
            class SimpleOutput:
                def __init__(self, loss, logits, loss_components, hidden_states=None):
                    self.loss = loss
                    self.logits = logits
                    self.loss_components = loss_components
                    self.hidden_states = hidden_states

                def __getitem__(self, key):
                    if key == "loss":
                        return self.loss
                    elif key == "logits":
                        return self.logits
                    elif key == 0:  # tuple access
                        return self.loss
                    return getattr(self, key, None)

                def __contains__(self, key):
                    return hasattr(self, key)

            return SimpleOutput(
                loss=base_outputs.loss if hasattr(base_outputs, "loss") else None,
                logits=base_outputs.logits if hasattr(base_outputs, "logits") else None,
                loss_components=loss_components,
                hidden_states=base_outputs.hidden_states
                if hasattr(base_outputs, "hidden_states")
                else None,
            )

    def _forward_with_coordinate_loss(
        self,
        base_outputs: Any,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        kwargs: Dict[str, Any],
    ) -> "ModelOutput":
        """
        Forward pass with coordinate token loss.

        Args:
            base_outputs: Outputs from base model
            input_ids: Input token IDs
            labels: Labels for loss computation
            kwargs: Additional arguments

        Returns:
            Model output with coordinate loss components
        """
        # Apply coordinate token masking to logits
        masked_logits = self.coordinate_processor.mask_coordinate_logits(
            base_outputs.logits,
            input_ids,
            self.coordinate_processor.coordinate_token_range,
        )

        # Get coordinate mask
        coord_mask = self.coordinate_processor.get_coordinate_mask(input_ids)

        # Debug logging for coordinate mask
        from src_new.config.config import logger

        coord_count = coord_mask.sum().item() if coord_mask is not None else 0
        logger.debug(
            f"🎯 Coordinate mask: {coord_count} coordinate tokens found in batch"
        )

        # Extract teacher and student spans from kwargs
        teacher_spans = kwargs.get("teacher_assistant_spans", None)
        student_spans = kwargs.get("student_assistant_spans", None)

        # Compute loss components with teacher-student spans
        loss_components = self.loss_manager.compute_loss_components(
            logits=masked_logits,
            labels=labels,
            coord_mask=coord_mask,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )

        # Create simple output for DataParallel compatibility
        class SimpleOutput:
            def __init__(self, loss, logits, loss_components, hidden_states=None):
                self.loss = loss
                self.logits = logits
                self.loss_components = loss_components
                self.hidden_states = hidden_states

            def __getitem__(self, key):
                if key == "loss":
                    return self.loss
                elif key == "logits":
                    return self.logits
                elif key == 0:  # tuple access
                    return self.loss
                return getattr(self, key, None)

            def __contains__(self, key):
                return hasattr(self, key)

        return SimpleOutput(
            loss=loss_components.loss,
            logits=masked_logits,
            loss_components=loss_components,
            hidden_states=base_outputs.hidden_states
            if hasattr(base_outputs, "hidden_states")
            else None,
        )

    def generate(self, **kwargs) -> torch.Tensor:
        """
        Generate text with coordinate token support.

        Args:
            **kwargs: Arguments for base model generate

        Returns:
            Generated token IDs
        """
        # Enable cache for generation
        self.base_model.config.use_cache = True

        # Generate with base model
        return self.base_model.generate(**kwargs)

    def get_loss_components(self) -> Optional[LossComponents]:
        """
        Get loss components from the last forward pass.

        Returns:
            Loss components or None if not available
        """
        return self.loss_manager.last_loss_components

    def get_last_loss_components(self) -> Optional[LossComponents]:
        """
        Get loss components from the last forward pass.

        This method provides compatibility with DistributedLossTrainer
        which expects this specific method name.

        Returns:
            Loss components or None if not available
        """
        return self.get_loss_components()

    def enable_coordinate_mode(self) -> None:
        """Enable coordinate token mode."""
        self._coordinate_mode = True

    def disable_coordinate_mode(self) -> None:
        """Disable coordinate token mode."""
        self._coordinate_mode = False

    @property
    def coordinate_mode_enabled(self) -> bool:
        """Check if coordinate token mode is enabled."""
        return self._coordinate_mode

    def train(self, mode: bool = True) -> "DetectionModel":
        """Set model to train mode."""
        self.base_model.train(mode)
        return super().train(mode)

    def eval(self) -> "DetectionModel":
        """Set model to eval mode."""
        self.base_model.eval()
        return super().eval()

    @property
    def device(self) -> torch.device:
        """Get device of base model."""
        return self.base_model.device

    @property
    def config(self) -> Any:
        """Get configuration object."""
        return self._config

    @config.setter
    def config(self, config: "Config") -> None:
        """Set configuration object."""
        self._config = config

    @property
    def model_config(self) -> Any:
        """Get model configuration."""
        return self.base_model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable gradient checkpointing in base model."""
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing in base model."""
        self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings from base model."""
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings from base model."""
        return self.base_model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Module:
        """
        Resize token embeddings in base model.

        Args:
            new_num_tokens: New number of tokens

        Returns:
            Updated embedding module
        """
        # Resize embeddings in base model
        self.base_model.resize_token_embeddings(new_num_tokens)

        # Update coordinate token range
        if hasattr(self, "coordinate_processor"):
            self.coordinate_processor.original_vocab_size = new_num_tokens
            self.coordinate_processor.coordinate_token_range = (
                new_num_tokens,
                new_num_tokens + self.coordinate_processor.max_coord_value,
            )

        return self.base_model.get_input_embeddings()

    def prepare_inputs_for_generation(self, **kwargs) -> Dict[str, Any]:
        """Prepare inputs for generation."""
        return self.base_model.prepare_inputs_for_generation(**kwargs)

    def get_rope_index(self, **kwargs) -> Any:
        """Get RoPE index from base model."""
        if hasattr(self.base_model, "get_rope_index"):
            return self.base_model.get_rope_index(**kwargs)
        return None

    def tie_weights(self) -> None:
        """
        Tie weights between input and output embeddings.

        This method ensures that the base model's tied weights are properly handled
        and prevents issues during checkpoint saving with shared tensors.
        """
        # Delegate to base model's tie_weights method
        if hasattr(self.base_model, "tie_weights"):
            self.base_model.tie_weights()

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending on configuration."""
        # Delegate to base model's implementation
        if hasattr(self.base_model, "_tie_or_clone_weights"):
            return self.base_model._tie_or_clone_weights(
                output_embeddings, input_embeddings
            )
        else:
            # Fallback implementation
            if getattr(self.base_model.config, "torchscript", False):
                output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
            else:
                output_embeddings.weight = input_embeddings.weight

    def save_pretrained(
        self, save_directory: str, safe_serialization: bool = True, **kwargs
    ) -> None:
        """
        Save the model with proper handling of tied weights.

        Args:
            save_directory: Directory to save the model
            safe_serialization: Whether to use safetensors format
            **kwargs: Additional arguments passed to base model's save_pretrained
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        try:
            # Try normal save first
            self.base_model.save_pretrained(
                save_directory, safe_serialization=safe_serialization, **kwargs
            )
        except RuntimeError as e:
            if "shared tensors" in str(e):
                print(
                    "⚠️ Shared tensor error detected, falling back to non-safe serialization"
                )
                # Fallback to non-safe serialization
                self.base_model.save_pretrained(
                    save_directory, safe_serialization=False, **kwargs
                )
            else:
                raise e

        # Save additional configuration
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(save_directory, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)

        # Log save location
        print(f"Model saved to {save_directory}")

        # Save coordinate configuration
        if self._coordinate_mode:
            coordinate_config = {
                "coordinate_tokens_enabled": True,
                "max_coord_value": self.coordinate_processor.max_coord_value,
                "original_vocab_size": self.coordinate_processor.original_vocab_size,
            }

            # Save as JSON
            import json

            with open(os.path.join(save_directory, "coordinate_config.json"), "w") as f:
                json.dump(coordinate_config, f, indent=2)
