"""
Model Adapter Module

This module handles core model extensions including embeddings, LM head modifications,
and tokenizer extensions for the Qwen2.5-VL model with coordinate token support.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger


class ModelAdapter:
    """
    Handles model adaptations and extensions for coordinate token support.

    Responsibilities:
    - Extended embeddings creation and management
    - Extended LM head creation and initialization
    - Tokenizer extension logic
    - Model extension validation
    """

    def __init__(
        self,
        base_model: Any,
        coordinate_config: Any,
        tokenizer: PreTrainedTokenizerBase,
        original_vocab_size: int,
        extended_vocab_size: int,
        logger: Optional[Any] = None,
    ):
        """Initialize model adapter."""
        self.base_model = base_model
        self.coordinate_config = coordinate_config
        self.tokenizer = tokenizer
        self.original_vocab_size = original_vocab_size
        self.extended_vocab_size = extended_vocab_size
        self.logger = logger or get_training_logger()

        # Initialize extension components
        self.extended_embeddings = None
        self.extended_lm_head = None

    def create_extended_embeddings(self):
        """Create extended embeddings with coordinate tokens."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        self.logger.info("🔧 Creating extended embeddings...")

        # Get original embeddings
        original_embeddings = self.base_model.get_input_embeddings()
        if original_embeddings is None:
            raise ValueError("Base model has no input embeddings")

        original_vocab_size = self.original_vocab_size
        extended_vocab_size = self.extended_vocab_size

        # Create new embedding layer with extended vocabulary
        self.extended_embeddings = nn.Embedding(
            num_embeddings=extended_vocab_size,
            embedding_dim=original_embeddings.embedding_dim,
            padding_idx=original_embeddings.padding_idx,
            device=original_embeddings.weight.device,
            dtype=original_embeddings.weight.dtype,
        )

        # Copy original embeddings
        with torch.no_grad():
            copy_size = min(original_vocab_size, extended_vocab_size)
            self.extended_embeddings.weight.data[:copy_size] = (
                original_embeddings.weight.data[:copy_size]
            )

        # Initialize coordinate token embeddings if extended
        if original_vocab_size < extended_vocab_size:
            coordinate_start = original_vocab_size
            coordinate_end = extended_vocab_size

            self.logger.info(
                f"   Initializing coordinate embeddings [{coordinate_start}:{coordinate_end}]"
            )

            # Try improved initialization if available
            try:
                from src.models.improved_coordinate_init import (
                    ImprovedCoordinateInitializer,
                )

                # Use tokenizer for improved initialization
                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    self.logger.info(
                        "🔧 Applying improved coordinate initialization..."
                    )
                    initializer = ImprovedCoordinateInitializer(
                        self.tokenizer,
                        self.base_model,
                        max_coord_value=coordinate_end - coordinate_start,
                    )
                    initializer.apply_improved_initialization(
                        coordinate_start, coordinate_end
                    )
                else:
                    self.logger.info(
                        "🔧 Applying simplified improved initialization..."
                    )
                    self._apply_simplified_improved_init(
                        coordinate_start, coordinate_end
                    )

            except ImportError:
                self.logger.warning(
                    "⚠️ Improved initialization not available, using standard initialization"
                )
                # Fallback to original initialization
                std = self.coordinate_config.coord_token_init_std
                nn.init.normal_(
                    self.extended_embeddings.weight[coordinate_start:coordinate_end],
                    mean=0.0,
                    std=std,
                )

            # Ensure coordinate embeddings require gradients
            self.extended_embeddings.weight[
                coordinate_start:coordinate_end
            ].requires_grad_(True)

        # Set requires_grad for all embeddings
        self.extended_embeddings.weight.requires_grad_(True)

        # Replace base model embeddings
        self.base_model.set_input_embeddings(self.extended_embeddings)

        coord_tokens_count = extended_vocab_size - original_vocab_size
        self.logger.info(
            f"✅ Extended embeddings created: +{coord_tokens_count} coordinate tokens"
        )

    def create_extended_lm_head(self):
        """Create extended LM head with coordinate token projections."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        self.logger.info("🔧 Creating extended LM head...")

        original_vocab_size = self.original_vocab_size
        extended_vocab_size = self.extended_vocab_size

        # Get original LM head
        original_lm_head = self.base_model.get_output_embeddings()
        if original_lm_head is None:
            raise ValueError("Base model has no output embeddings (LM head)")

        # Create new LM head with extended vocabulary
        new_lm_head = nn.Linear(
            in_features=original_lm_head.in_features,
            out_features=extended_vocab_size,
            bias=original_lm_head.bias is not None,
            device=original_lm_head.weight.device,
            dtype=original_lm_head.weight.dtype,
        )

        # Copy original weights
        with torch.no_grad():
            orig_out_features = original_lm_head.out_features
            copy_size = min(orig_out_features, extended_vocab_size)

            if copy_size > 0:
                new_lm_head.weight.data[:copy_size] = original_lm_head.weight.data[
                    :copy_size
                ]

            if original_lm_head.bias is not None and new_lm_head.bias is not None:
                new_lm_head.bias.data[:copy_size] = original_lm_head.bias.data[
                    :copy_size
                ]

        # Initialize coordinate token projections if extended
        if original_vocab_size < extended_vocab_size:
            coordinate_start = original_vocab_size
            coordinate_end = extended_vocab_size

            self.logger.info(
                f"   Initializing coordinate projections [{coordinate_start}:{coordinate_end}]"
            )

            # Initialize with small random values
            std = self.coordinate_config.coord_token_init_std
            nn.init.normal_(
                new_lm_head.weight[coordinate_start:coordinate_end], mean=0.0, std=std
            )

            # Ensure coordinate projections require gradients
            new_lm_head.weight[coordinate_start:coordinate_end].requires_grad_(True)

        # Set requires_grad for all weights
        new_lm_head.weight.requires_grad_(True)
        if new_lm_head.bias is not None:
            new_lm_head.bias.requires_grad_(True)

        # Store and set extended LM head
        self.extended_lm_head = new_lm_head
        self.base_model.set_output_embeddings(new_lm_head)

        coord_projections_count = extended_vocab_size - original_vocab_size
        self.logger.info(
            f"✅ Extended LM head created: +{coord_projections_count} coordinate projections"
        )

    def extend_tokenizer(self):
        """Extend tokenizer with coordinate tokens."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        self.logger.info("🔧 Extending tokenizer with coordinate tokens...")

        # Create coordinate tokens
        max_coord_value = self.coordinate_config.max_coord_value
        coordinate_tokens = [f"<coord_{i}>" for i in range(max_coord_value)]

        # Add tokens to tokenizer
        num_added_tokens = self.tokenizer.add_tokens(coordinate_tokens)

        if num_added_tokens != max_coord_value:
            self.logger.warning(
                f"Expected to add {max_coord_value} tokens, but added {num_added_tokens}"
            )

        self.logger.info(f"✅ Added {num_added_tokens} coordinate tokens to tokenizer")

    def validate_extensions(self):
        """Validate that all model extensions are properly configured."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        self.logger.info("🔍 Validating model extensions...")

        # Validate extended embeddings
        if self.extended_embeddings is None:
            raise ValueError("Extended embeddings not created")

        expected_embedding_shape = (
            self.extended_vocab_size,
            self.extended_embeddings.embedding_dim,
        )
        actual_embedding_shape = self.extended_embeddings.weight.shape

        if actual_embedding_shape != expected_embedding_shape:
            raise ValueError(
                f"Extended embeddings shape mismatch: "
                f"expected {expected_embedding_shape}, got {actual_embedding_shape}"
            )

        # Validate extended LM head
        if self.extended_lm_head is None:
            raise ValueError("Extended LM head not created")

        expected_lm_head_shape = (
            self.extended_vocab_size,
            self.extended_lm_head.in_features,
        )
        actual_lm_head_shape = self.extended_lm_head.weight.shape

        if actual_lm_head_shape != expected_lm_head_shape:
            raise ValueError(
                f"Extended LM head shape mismatch: "
                f"expected {expected_lm_head_shape}, got {actual_lm_head_shape}"
            )

        # Validate vocabulary size consistency
        coordinate_token_count = self.extended_vocab_size - self.original_vocab_size
        expected_extended_size = self.original_vocab_size + coordinate_token_count

        if self.extended_vocab_size != expected_extended_size:
            raise ValueError(
                f"Vocabulary size mismatch: "
                f"expected {expected_extended_size} "
                f"(original {self.original_vocab_size} + coordinate {coordinate_token_count}), "
                f"got {self.extended_vocab_size}"
            )

        # Validate coordinate token embeddings
        self._validate_coordinate_embeddings()

        self.logger.info("✅ Model extensions validation passed")

    def _validate_coordinate_embeddings(self):
        """Validate coordinate token embeddings are properly initialized."""
        coordinate_start_idx = self.original_vocab_size
        coordinate_end_idx = self.extended_vocab_size

        if coordinate_start_idx >= coordinate_end_idx:
            return  # No coordinate tokens

        # Check coordinate embeddings are not zero
        coordinate_embeddings = self.extended_embeddings.weight[
            coordinate_start_idx:coordinate_end_idx
        ]

        if torch.allclose(
            coordinate_embeddings, torch.zeros_like(coordinate_embeddings)
        ):
            raise ValueError(
                "Coordinate embeddings are all zeros - initialization failed"
            )

        # Check coordinate LM head weights are not zero
        coordinate_lm_weights = self.extended_lm_head.weight[
            coordinate_start_idx:coordinate_end_idx
        ]

        if torch.allclose(
            coordinate_lm_weights, torch.zeros_like(coordinate_lm_weights)
        ):
            raise ValueError(
                "Coordinate LM head weights are all zeros - initialization failed"
            )

        # Validate device and dtype consistency
        base_device = next(self.base_model.parameters()).device
        base_dtype = next(self.base_model.parameters()).dtype

        if self.extended_embeddings.weight.device != base_device:
            raise ValueError(
                f"Device mismatch: "
                f"base model on {base_device}, extended embeddings on {self.extended_embeddings.weight.device}"
            )

        if self.extended_embeddings.weight.dtype != base_dtype:
            raise ValueError(
                f"Dtype mismatch: "
                f"base model uses {base_dtype}, extended embeddings use {self.extended_embeddings.weight.dtype}"
            )

        if self.extended_lm_head.weight.device != base_device:
            raise ValueError(
                f"Device mismatch: "
                f"base model on {base_device}, extended LM head on {self.extended_lm_head.weight.device}"
            )

        if self.extended_lm_head.weight.dtype != base_dtype:
            raise ValueError(
                f"Dtype mismatch: "
                f"base model uses {base_dtype}, extended LM head uses {self.extended_lm_head.weight.dtype}"
            )

    def get_extension_metadata(self) -> Dict[str, Any]:
        """Get metadata about model extensions for saving/loading."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return {}

        return {
            "extended_vocab_size": self.extended_vocab_size,
            "original_vocab_size": self.original_vocab_size,
            "coordinate_token_count": self.extended_vocab_size
            - self.original_vocab_size,
            "coord_token_init_std": self.coordinate_config.coord_token_init_std,
            "coordinate_loss_weight": self.coordinate_config.coordinate_loss_weight,
            "regular_loss_weight": self.coordinate_config.regular_loss_weight,
        }

    def get_extension_state_dict(self) -> Dict[str, Any]:
        """Get state dict for model extensions."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return {}

        if self.extended_embeddings is None or self.extended_lm_head is None:
            raise ValueError("Model extensions not properly initialized")

        return {
            "extended_embeddings": self.extended_embeddings.state_dict(),
            "extended_lm_head": self.extended_lm_head.state_dict(),
        }

    def load_extension_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for model extensions."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        if (
            "extended_embeddings" not in state_dict
            or "extended_lm_head" not in state_dict
        ):
            raise ValueError("Missing extension state dict components")

        # Load extended embeddings
        if self.extended_embeddings is None:
            raise ValueError("Extended embeddings not initialized")

        self.extended_embeddings.load_state_dict(state_dict["extended_embeddings"])

        # Load extended LM head
        if self.extended_lm_head is None:
            # Create LM head if not exists
            self.extended_lm_head = nn.Linear(
                in_features=self.base_model.get_output_embeddings().in_features,
                out_features=self.extended_vocab_size,
                bias=self.base_model.get_output_embeddings().bias is not None,
            )

        self.extended_lm_head.load_state_dict(state_dict["extended_lm_head"])

        # Set as base model components
        self.base_model.set_input_embeddings(self.extended_embeddings)
        self.base_model.set_output_embeddings(self.extended_lm_head)

    def _apply_simplified_improved_init(
        self, coordinate_start: int, coordinate_end: int
    ):
        """Apply simplified improved initialization when full initialization is not available."""
        with torch.no_grad():
            weights = self.extended_embeddings.weight
            embedding_dim = weights.shape[1]

            # Get coordinate embeddings
            coord_embeddings = weights[coordinate_start:coordinate_end]
            num_coords = coordinate_end - coordinate_start

            # Reduce variance from default initialization
            coord_embeddings *= 0.5

            # Add position-dependent variation to distinguish coordinates
            for i in range(num_coords):
                # Small position bias that scales with coordinate value
                position_factor = i / num_coords
                position_bias = torch.randn(embedding_dim) * position_factor * 0.01
                coord_embeddings[i] += position_bias

            self.logger.info(
                f"✅ Applied simplified improved initialization to {num_coords} coordinate tokens"
            )

            # Log statistics
            norms = torch.norm(coord_embeddings, dim=1)
            self.logger.info(
                f"   📊 Coordinate embedding norms: mean={norms.mean():.6f}, std={norms.std():.6f}"
            )
