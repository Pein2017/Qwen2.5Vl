"""
Coordinate Token Handler

This module handles all coordinate token operations including setup, conversion,
validation, and coordinate-related computations for the Qwen2.5-VL model.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger


class CoordinateHandler:
    """
    Handles coordinate token operations for the Qwen2.5-VL model.

    Responsibilities:
    - Coordinate token setup and initialization
    - Token-to-coordinate conversion logic
    - Coordinate mask computation
    - Coordinate token validation and ranges
    """

    def __init__(
        self,
        coordinate_config: Any,
        tokenizer: PreTrainedTokenizerBase,
        original_vocab_size: int,
        logger: Optional[Any] = None,
    ):
        """Initialize coordinate handler."""
        self.coordinate_config = coordinate_config
        self.tokenizer = tokenizer
        self.original_vocab_size = original_vocab_size
        self.logger = logger or get_training_logger()

        # Initialize coordinate-related attributes
        self.extended_vocab_size = original_vocab_size
        self.coordinate_manager = None
        self.box_start_id = None
        self.box_end_id = None

        # Initialize coordinate token ranges
        self._coordinate_token_ranges = {}

    def setup_coordinate_tokens(self, base_model=None):
        """Set up coordinate tokens for the model."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        # Store base model for token manager setup
        self.base_model = base_model

        # Validate required values
        max_coord_value = self.coordinate_config.max_coord_value
        if (
            not isinstance(self.original_vocab_size, int)
            or self.original_vocab_size <= 0
        ):
            raise ValueError(
                f"Invalid original_vocab_size: {self.original_vocab_size}. Must be positive integer."
            )

        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"Invalid max_coord_value: {max_coord_value}. Must be positive integer."
            )

        # Calculate extended vocab size
        self.extended_vocab_size = self.original_vocab_size + max_coord_value

        # Setup unified token manager
        self._setup_unified_token_manager()

    def _setup_unified_token_manager(self):
        """Set up unified token manager."""
        max_coord_value = self.coordinate_config.max_coord_value
        if max_coord_value <= 0:
            raise ValueError(
                f"Invalid max_coord_value: {max_coord_value}. Must be positive."
            )

        # Create unified token manager
        from src.utils.tokens import create_unified_token_manager

        self.coordinate_manager = create_unified_token_manager(
            tokenizer=self.tokenizer,
            model=self.base_model,
            max_coord_value=self.coordinate_config.max_coord_value,
            coordinate_tokens_enabled=self.coordinate_config.coordinate_tokens_enabled,
        )

        # Update geometry token IDs if multi-geometry is enabled
        if self.coordinate_config.enable_multi_geometry:
            geometry_tokens = ["<square>", "<line>"]
            self._update_geometry_token_ids(geometry_tokens)

        # Update coordinate token ranges
        self._update_coordinate_token_ranges()

    def _update_geometry_token_ids(self, geometry_tokens: List[str]):
        """Update geometry token IDs in coordinate manager."""
        if not self.coordinate_manager:
            return

        geometry_token_ids = {}
        for token in geometry_tokens:
            convert_fn = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            if token in self.tokenizer.get_vocab() and callable(convert_fn):
                geometry_token_ids[token] = convert_fn(token)

        if hasattr(self.coordinate_manager, "set_geometry_token_ids"):
            self.coordinate_manager.set_geometry_token_ids(geometry_token_ids)

    def _update_coordinate_token_ranges(self):
        """Update coordinate token ranges for different geometries."""
        if not self.coordinate_manager:
            return

        coord_start = self.original_vocab_size

        # Standard bbox coordinates (4 values)
        self._coordinate_token_ranges["bbox"] = {
            "start": coord_start,
            "end": coord_start + 4,
            "count": 4,
        }

        # Multi-geometry support
        if self.coordinate_config.enable_multi_geometry:
            # Line coordinates (variable length up to max_line_coordinates)
            max_line_coords = self.coordinate_config.max_line_coordinates
            self._coordinate_token_ranges["line"] = {
                "start": coord_start + 4,
                "end": coord_start + 4 + max_line_coords,
                "count": max_line_coords,
            }

    def get_coordinate_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for coordinate tokens in the input."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return torch.zeros_like(token_ids, dtype=torch.bool)

        coord_start = self.original_vocab_size
        coord_end = self.extended_vocab_size

        # Create mask for coordinate tokens
        coord_mask = (token_ids >= coord_start) & (token_ids < coord_end)
        return coord_mask

    def convert_bbox_to_tokens(self, bbox: List[int]) -> List[int]:
        """Convert integer bbox [0, 2047] to coordinate token IDs."""
        coord_start = self.original_vocab_size

        # Validate input coordinates
        for i, coord in enumerate(bbox):
            if not isinstance(coord, int):
                raise ValueError(
                    f"Coordinate {i} must be integer, got {type(coord)}: {coord}"
                )
            if not (0 <= coord < self.coordinate_config.max_coord_value):
                raise ValueError(
                    f"Coordinate {i} = {coord} out of bounds [0, {self.coordinate_config.max_coord_value})"
                )

        # Convert to token IDs
        coord_tokens = []
        for coord in bbox:
            if coord_start is not None:
                coord_tokens.append(coord_start + coord)

        return [self.box_start_id] + coord_tokens + [self.box_end_id]

    def convert_tokens_to_bbox(self, token_ids: List[int]) -> Optional[List[int]]:
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
                    or token_id >= coord_start + self.coordinate_config.max_coord_value
                ):
                    return None

                # Direct mapping: token ID -> integer coordinate
                coord_value = token_id - coord_start
                bbox.append(coord_value)

            return bbox

        except (ValueError, IndexError):
            return None

    def extract_expected_coordinates(self, coord_logits: torch.Tensor) -> torch.Tensor:
        """Extract expected coordinates from coordinate logits using soft expectation."""
        if coord_logits.size(-1) != self.coordinate_config.max_coord_value:
            raise ValueError(
                f"Coordinate logits size {coord_logits.size(-1)} != max_coord_value {self.coordinate_config.max_coord_value}"
            )

        # Apply temperature scaling if configured
        temperature = getattr(
            self.coordinate_config, "soft_expectation_temperature", 1.0
        )
        if temperature != 1.0:
            coord_logits = coord_logits / temperature

        # Softmax over coordinate values
        coord_probs = F.softmax(coord_logits, dim=-1)

        # Create coordinate indices [0, 1, 2, ..., max_coord_value-1]
        coord_indices = torch.arange(
            self.coordinate_config.max_coord_value,
            device=coord_logits.device,
            dtype=coord_logits.dtype,
        )

        # Compute expected coordinates
        expected_coords = torch.sum(coord_probs * coord_indices, dim=-1)

        return expected_coords

    def validate_coordinate_tokens(
        self, input_ids: torch.Tensor, sample_info: Optional[Dict] = None
    ):
        """Validate coordinate tokens in the input."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        # Get coordinate mask
        coord_mask = self.get_coordinate_mask(input_ids)

        if coord_mask.any():
            coord_tokens = input_ids[coord_mask]
            coord_start = self.original_vocab_size
            coord_end = self.extended_vocab_size

            # Validate all coordinate tokens are in valid range
            invalid_tokens = coord_tokens[
                (coord_tokens < coord_start) | (coord_tokens >= coord_end)
            ]

            if len(invalid_tokens) > 0:
                raise ValueError(
                    f"Invalid coordinate tokens found: {invalid_tokens.tolist()}, "
                    f"expected range [{coord_start}, {coord_end})"
                )

        self.logger.debug(f"✅ Coordinate token validation passed")

    def get_coordinate_tokenizer_utils(self) -> Dict[str, Any]:
        """Get coordinate tokenizer utility functions."""
        return {
            "convert_bbox_to_tokens": self.convert_bbox_to_tokens,
            "convert_tokens_to_bbox": self.convert_tokens_to_bbox,
            "get_coordinate_mask": self.get_coordinate_mask,
            "validate_coordinate_tokens": self.validate_coordinate_tokens,
        }

    def update_coordinate_manager_geometry_tokens(
        self, tokenizer: PreTrainedTokenizerBase
    ):
        """Update coordinate manager with geometry tokens from tokenizer."""
        if not self.coordinate_manager:
            return

        # Update official box tokens
        convert_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_fn):
            self.box_start_id = convert_fn("<|box_start|>")
            self.box_end_id = convert_fn("<|box_end|>")

        if hasattr(self.coordinate_manager, "update_geometry_tokens"):
            self.coordinate_manager.update_geometry_tokens(tokenizer)

    def ensure_coordinate_tokens_in_vocabulary(self):
        """Ensure coordinate tokens are properly added to vocabulary."""
        if not self.coordinate_config.enable_coordinate_tokens:
            return

        # Check if coordinate tokens are properly set up
        if self.extended_vocab_size <= self.original_vocab_size:
            raise ValueError(
                f"Extended vocab size {self.extended_vocab_size} must be > original size {self.original_vocab_size}"
            )

        # Validate coordinate token ranges
        coord_count = self.extended_vocab_size - self.original_vocab_size
        if coord_count != self.coordinate_config.max_coord_value:
            raise ValueError(
                f"Coordinate token count {coord_count} != max_coord_value {self.coordinate_config.max_coord_value}"
            )

        self.logger.debug(
            f"✅ Coordinate tokens properly configured: {coord_count} tokens"
        )
