"""
Unified Coordinate Token Manager for Production-Robust Coordinate Processing

This module provides a centralized, production-ready system for handling coordinate tokens
in the Qwen2.5-VL fine-tuning pipeline. It replaces the scattered coordinate token logic
with a unified, well-tested, and optimized implementation.

Key Features:
- Unified coordinate token management
- Proper bbox span detection and masking
- Efficient coordinate token ID management
- Comprehensive validation and error handling
- Integer coordinate handling [0, 2047]
- Production-ready with proper logging and monitoring
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_logger


@dataclass
class CoordinateTokenConfig:
    """Configuration for coordinate token system."""

    # Core settings
    enable_coordinate_tokens: bool
    max_coord_value: int

    # Token IDs (official Qwen2.5-VL tokens)
    box_start_id: int
    box_end_id: int

    # Loss computation settings
    coordinate_loss_weight: float
    regular_loss_weight: float
    soft_expectation_temperature: float
    focal_loss_alpha: float
    focal_loss_gamma: float

    # Performance settings
    enable_validation: bool
    enable_caching: bool
    batch_processing: bool


class CoordinateTokenManager:
    """
    Unified manager for all coordinate token operations.

    This class centralizes all coordinate token functionality including:
    - Tokenizer integration and vocabulary extension
    - Format conversion (JSON ↔ coordinate tokens)
    - Bbox span detection and masking
    - Loss computation with proper coordinate-aware splitting
    - Validation and error handling
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: CoordinateTokenConfig,
        original_vocab_size: int,
    ):
        """
        Initialize coordinate token manager.

        Args:
            tokenizer: Model tokenizer
            config: Coordinate token configuration
            original_vocab_size: Original vocabulary size before extension
        """
        self.tokenizer = tokenizer
        self.config = config
        self.original_vocab_size = original_vocab_size
        self.logger = get_logger("coordinate_token_manager")

        # Calculate extended vocabulary size
        self.extended_vocab_size = (
            original_vocab_size + config.max_coord_value
            if config.enable_coordinate_tokens
            else original_vocab_size
        )

        # Token ID ranges - get actual IDs from tokenizer after tokens are added
        # Note: These will be updated after tokens are added to tokenizer
        self.coord_start_id = original_vocab_size  # Initial estimate
        self.coord_end_id = (
            original_vocab_size + config.max_coord_value
        )  # Initial estimate

        # Validation cache
        self._validation_cache = {} if config.enable_caching else None

        # Performance metrics
        self._metrics = {
            "conversions_performed": 0,
            "bbox_spans_detected": 0,
            "coordinate_tokens_processed": 0,
            "validation_errors": 0,
        }

        # NOTE: Do NOT add tokens to tokenizer here - wrapper handles this
        # to prevent double addition and vocabulary size mismatches
        if config.enable_coordinate_tokens:
            self._update_coordinate_token_ranges()
            self._validate_coordinate_tokens()

        self.logger.info(f"✅ CoordinateTokenManager initialized:")
        self.logger.info(f"   Enabled: {config.enable_coordinate_tokens}")
        self.logger.info(f"   Original vocab: {original_vocab_size}")
        self.logger.info(f"   Extended vocab: {self.extended_vocab_size}")
        self.logger.info(
            f"   Coordinate range: [{self.coord_start_id}, {self.coord_end_id})"
        )

    def _update_coordinate_token_ranges(self):
        """Update coordinate token ID ranges based on actual tokenizer state."""
        # Get the actual token ID for the first coordinate token
        first_coord_id = self.tokenizer.convert_tokens_to_ids("<coord_0>")

        if first_coord_id != self.tokenizer.unk_token_id:
            # Update ranges based on actual token IDs
            self.coord_start_id = first_coord_id
            self.coord_end_id = first_coord_id + self.config.max_coord_value

            self.logger.info(f"🔧 Updated coordinate token ranges:")
            self.logger.info(f"   Actual coord_start_id: {self.coord_start_id}")
            self.logger.info(f"   Actual coord_end_id: {self.coord_end_id}")
        else:
            self.logger.warning("⚠️ Could not find <coord_0> token in tokenizer")

    def _validate_coordinate_tokens(self):
        """Validate that coordinate tokens exist in tokenizer (added by wrapper)."""
        # Verify token IDs are in expected range
        missing_tokens = []
        for i in range(min(10, self.config.max_coord_value)):  # Check first 10 tokens
            token_text = f"<coord_{i}>"
            token_id = self.tokenizer.convert_tokens_to_ids(token_text)
            expected_id = self.coord_start_id + i

            if token_id != expected_id:
                missing_tokens.append((token_text, token_id, expected_id))

        if missing_tokens:
            self.logger.warning(f"⚠️ Coordinate token validation failed:")
            for token_text, actual_id, expected_id in missing_tokens:
                self.logger.warning(
                    f"   {token_text} -> {actual_id}, expected {expected_id}"
                )
            self.logger.warning(
                "   Make sure wrapper._extend_tokenizer() was called first"
            )
        else:
            self.logger.info(f"✅ Coordinate tokens validated (checked first 10)")

    def _setup_coordinate_tokens(self):
        """DEPRECATED: Token addition now handled by wrapper to prevent double-addition."""
        self.logger.warning(
            "⚠️ _setup_coordinate_tokens() is deprecated - use wrapper._extend_tokenizer() instead"
        )
        self._validate_coordinate_tokens()

    def convert_json_to_coordinate_format(self, json_response: str) -> str:
        """
        Convert JSON bbox response to coordinate token format.

        Args:
            json_response: JSON string with bbox_2d and label fields

        Returns:
            Coordinate token formatted string
        """
        if not self.config.enable_coordinate_tokens:
            return json_response

        try:
            objects = json.loads(json_response)
            if not objects:
                return "[]"

            coordinate_responses = []
            for obj in objects:
                if "bbox_2d" not in obj:
                    raise ValueError(
                        f"Missing required 'bbox_2d' field in object: {obj}"
                    )
                if "label" not in obj:
                    raise ValueError(f"Missing required 'label' field in object: {obj}")

                bbox = obj["bbox_2d"]
                label = obj["label"]

                # Convert to integer coordinates
                int_bbox = self._normalize_bbox_to_integers(bbox)

                # Create coordinate token sequence
                coord_tokens = self._create_coordinate_token_sequence(int_bbox)

                # Format with label
                coord_response = f"{label}: {coord_tokens}"
                coordinate_responses.append(coord_response)

            result = "\n".join(coordinate_responses)
            self._metrics["conversions_performed"] += 1
            return result

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to convert JSON to coordinate format: {e}")
            self._metrics["validation_errors"] += 1
            return json_response

    def convert_coordinate_to_json_format(self, coordinate_response: str) -> str:
        """
        Convert coordinate token response back to JSON format.

        Args:
            coordinate_response: Coordinate token formatted string

        Returns:
            JSON string with bbox_2d and label fields
        """
        if not self.config.enable_coordinate_tokens:
            return coordinate_response

        try:
            # Pattern to match coordinate sequences
            pattern = r"(.*?):\s*<\|box_start\|>(<coord_\d+>){4}<\|box_end\|>"
            matches = re.findall(pattern, coordinate_response)

            if not matches:
                # Try parsing as JSON fallback
                try:
                    json.loads(coordinate_response)
                    return coordinate_response
                except Exception:
                    return "[]"

            json_objects = []
            for match in matches:
                label = match[0].strip()

                # Extract coordinate tokens
                coord_match = re.search(
                    rf"{re.escape(label)}:\s*<\|box_start\|>(<coord_\d+>){{4}}<\|box_end\|>",
                    coordinate_response,
                )

                if coord_match:
                    coord_tokens = re.findall(r"<coord_(\d+)>", coord_match.group(0))

                    if len(coord_tokens) == 4:
                        # Convert back to normalized coordinates
                        int_coords = [int(token) for token in coord_tokens]
                        bbox = self._denormalize_integers_to_bbox(int_coords)
                        json_objects.append({"bbox_2d": bbox, "label": label})

            result = json.dumps(
                json_objects, ensure_ascii=False, separators=(",", ": ")
            )
            self._metrics["conversions_performed"] += 1
            return result

        except Exception as e:
            self.logger.warning(f"Failed to convert coordinate format to JSON: {e}")
            self._metrics["validation_errors"] += 1
            return "[]"

    def detect_bbox_spans(self, token_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect bbox spans in token sequence.

        Args:
            token_ids: Token ID tensor (batch_size, seq_len)

        Returns:
            List of (start_idx, end_idx) tuples for bbox spans
        """
        bbox_spans = []

        # Handle both batched and unbatched input
        if token_ids.dim() == 2:
            # Batched input - process each sequence
            for batch_idx in range(token_ids.size(0)):
                batch_spans = self._detect_bbox_spans_single(token_ids[batch_idx])
                bbox_spans.extend(
                    [(batch_idx, start, end) for start, end in batch_spans]
                )
        else:
            # Single sequence
            bbox_spans = self._detect_bbox_spans_single(token_ids)

        self._metrics["bbox_spans_detected"] += len(bbox_spans)
        return bbox_spans

    def _detect_bbox_spans_single(
        self, token_ids: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Detect bbox spans in single token sequence."""
        spans = []
        i = 0

        while i < len(token_ids):
            # Look for box_start token
            if token_ids[i] == self.config.box_start_id:
                start_idx = i
                i += 1

                # Look for box_end token
                while i < len(token_ids) and token_ids[i] != self.config.box_end_id:
                    i += 1

                if i < len(token_ids):  # Found box_end
                    end_idx = i + 1  # Include box_end token
                    spans.append((start_idx, end_idx))
                    i += 1
                else:
                    # No matching box_end found
                    break
            else:
                i += 1

        return spans

    def create_coordinate_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for coordinate tokens within bbox spans.

        Args:
            token_ids: Token ID tensor

        Returns:
            Boolean mask tensor (same shape as token_ids)
        """
        mask = torch.zeros_like(token_ids, dtype=torch.bool)

        # Handle both batched and unbatched input
        if token_ids.dim() == 2:
            for batch_idx in range(token_ids.size(0)):
                batch_mask = self._create_coordinate_mask_single(token_ids[batch_idx])
                mask[batch_idx] = batch_mask
        else:
            mask = self._create_coordinate_mask_single(token_ids)

        return mask

    def _create_coordinate_mask_single(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create coordinate mask for single sequence."""
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        bbox_spans = self._detect_bbox_spans_single(token_ids)

        for start_idx, end_idx in bbox_spans:
            # Only mark actual coordinate tokens (not box_start/box_end)
            for i in range(start_idx + 1, end_idx - 1):
                if self.coord_start_id <= token_ids[i] < self.coord_end_id:
                    mask[i] = True
                    self._metrics["coordinate_tokens_processed"] += 1

        return mask

    def _normalize_bbox_to_integers(self, bbox: List[Union[int, float]]) -> List[int]:
        """
        Convert bbox coordinates to integers in [0, 2047] range.

        Args:
            bbox: Bbox coordinates (normalized [0,1] or absolute)

        Returns:
            Integer coordinates in [0, 2047] range
        """
        if self.config.enable_validation:
            if len(bbox) != 4:
                raise ValueError(f"Bbox must have 4 coordinates, got {len(bbox)}")

        int_bbox = []
        for coord in bbox:
            # Convert to float first to handle both int and float input
            float_coord = float(coord)

            # If coordinate is in [0, 1] range, assume it's normalized
            if 0 <= float_coord <= 1:
                int_coord = int(float_coord * (self.config.max_coord_value - 1))
            else:
                # Assume it's already in absolute coordinates
                int_coord = int(float_coord)

            # Clamp to valid range
            int_coord = max(0, min(int_coord, self.config.max_coord_value - 1))
            int_bbox.append(int_coord)

        return int_bbox

    def _denormalize_integers_to_bbox(self, int_coords: List[int]) -> List[float]:
        """
        Convert integer coordinates back to normalized bbox.

        Args:
            int_coords: Integer coordinates in [0, 2047] range

        Returns:
            Normalized bbox coordinates in [0, 1] range
        """
        bbox = []
        for coord in int_coords:
            # Clamp to valid range
            coord = max(0, min(coord, self.config.max_coord_value - 1))
            # Convert to normalized [0, 1] range
            normalized = coord / (self.config.max_coord_value - 1)
            bbox.append(normalized)

        return bbox

    def _create_coordinate_token_sequence(self, int_bbox: List[int]) -> str:
        """
        Create coordinate token sequence from integer bbox.

        Args:
            int_bbox: Integer bbox coordinates [x1, y1, x2, y2]

        Returns:
            Coordinate token sequence string
        """
        if self.config.enable_validation:
            for i, coord in enumerate(int_bbox):
                if not (0 <= coord < self.config.max_coord_value):
                    raise ValueError(
                        f"Coordinate {i} = {coord} out of bounds [0, {self.config.max_coord_value})"
                    )

        # Create coordinate tokens
        coord_tokens = [f"<coord_{coord}>" for coord in int_bbox]

        # Combine with box tokens
        return f"<|box_start|>{''.join(coord_tokens)}<|box_end|>"

    def validate_coordinate_format(self, text: str) -> bool:
        """
        Validate coordinate token format.

        Args:
            text: Text to validate

        Returns:
            True if format is valid
        """
        if not self.config.enable_coordinate_tokens:
            return True

        # Check cache first
        if self._validation_cache is not None and text in self._validation_cache:
            return self._validation_cache[text]

        # Validate coordinate token pattern
        pattern = r"<\|box_start\|>(<coord_\d+>){4}<\|box_end\|>"

        is_valid = True
        for match in re.finditer(pattern, text):
            coord_tokens = re.findall(r"<coord_(\d+)>", match.group(0))
            for coord_str in coord_tokens:
                coord_val = int(coord_str)
                if coord_val >= self.config.max_coord_value:
                    is_valid = False
                    break
            if not is_valid:
                break

        # Cache result
        if self._validation_cache is not None:
            self._validation_cache[text] = is_valid

        if not is_valid:
            self._metrics["validation_errors"] += 1

        return is_valid

    def get_metrics(self) -> Dict[str, int]:
        """Get performance metrics."""
        return self._metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = {
            "conversions_performed": 0,
            "bbox_spans_detected": 0,
            "coordinate_tokens_processed": 0,
            "validation_errors": 0,
        }

    def is_coordinate_token(self, token_id: int) -> bool:
        """Check if token ID is a coordinate token."""
        result = self.coord_start_id <= token_id < self.coord_end_id
        # Debug first few calls
        if hasattr(self, "_debug_count"):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 10:
            self.logger.debug(
                f"   🔍 is_coordinate_token({token_id}): range=[{self.coord_start_id}, {self.coord_end_id}), result={result}"
            )

        return result

    def get_coordinate_index(self, token_id: int) -> Optional[int]:
        """Get coordinate index from token ID."""
        if self.is_coordinate_token(token_id):
            return token_id - self.coord_start_id
        return None

    def get_coordinate_token_id(self, coord_index: int) -> Optional[int]:
        """Get token ID from coordinate index."""
        if 0 <= coord_index < self.config.max_coord_value:
            return self.coord_start_id + coord_index
        return None


def create_coordinate_token_manager(
    tokenizer: PreTrainedTokenizerBase,
    original_vocab_size: int,
    coordinate_config: dict,
) -> CoordinateTokenManager:
    """
    Factory function to create coordinate token manager.

    Args:
        tokenizer: Model tokenizer
        original_vocab_size: Original vocabulary size
        coordinate_config: Configuration dictionary from model config

    Returns:
        Configured CoordinateTokenManager
    """
    config = CoordinateTokenConfig(
        enable_coordinate_tokens=coordinate_config["enable_coordinate_tokens"],
        max_coord_value=coordinate_config["max_coord_value"],
        box_start_id=coordinate_config.get("box_start_id", 151648),
        box_end_id=coordinate_config.get("box_end_id", 151649),
        coordinate_loss_weight=coordinate_config["coordinate_loss_weight"],
        regular_loss_weight=coordinate_config["regular_loss_weight"],
        soft_expectation_temperature=coordinate_config["soft_expectation_temperature"],
        focal_loss_alpha=coordinate_config["focal_loss_alpha"],
        focal_loss_gamma=coordinate_config["focal_loss_gamma"],
        enable_validation=coordinate_config.get("enable_validation", True),
        enable_caching=coordinate_config.get("enable_caching", True),
        batch_processing=coordinate_config.get("batch_processing", True),
    )

    return CoordinateTokenManager(
        tokenizer=tokenizer, config=config, original_vocab_size=original_vocab_size
    )
