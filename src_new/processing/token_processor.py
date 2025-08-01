#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token processor for Qwen2.5-VL coordinate token handling.

Implements coordinate token conversion, tokenizer vocabulary extension,
and special token wrapping for multi-geometry annotations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer, Qwen2VLForConditionalGeneration


def get_token_logger() -> logging.Logger:
    """Get logger for token processing."""
    from src_new.config.config import _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL

    logger = logging.getLogger("token_processor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_GLOBAL_LOG_LEVEL)
        _CONFIGURED_LOGGERS.add("token_processor")
    return logger


logger = get_token_logger()


@dataclass
class TokenConfig:
    """Configuration for token processing."""

    max_coord_value: int
    coordinate_tokens_enabled: bool = False
    new_geometry_tokens: Optional[List[str]] = None

    def __post_init__(self):
        if self.new_geometry_tokens is None:
            # Only add line tokens - quad tokens already exist in Qwen2.5-VL
            self.new_geometry_tokens = [
                "<|line_start|>",
                "<|line_end|>",
            ]


class TokenProcessor:
    """
    Handles coordinate token conversion and tokenizer vocabulary extension.

    Supports both standard mode (coordinates as integers) and coordinate token mode
    (coordinates as special tokens). Manages tokenizer vocabulary extension for
    new geometry types.
    """

    def __init__(self, config: TokenConfig) -> None:
        """
        Initialize token processor.

        Args:
            config: Token processing configuration
        """
        self.config = config
        self.coordinate_token_map: Dict[int, str] = {}
        self.reverse_coordinate_map: Dict[str, int] = {}

        if config.coordinate_tokens_enabled:
            self._build_coordinate_token_maps()

    def _build_coordinate_token_maps(self) -> None:
        """Build coordinate token mapping dictionaries."""
        for coord in range(self.config.max_coord_value + 1):
            token = f"<|coord_{coord}|>"
            self.coordinate_token_map[coord] = token
            self.reverse_coordinate_map[token] = coord

        logger.info(f"Built coordinate token maps for 0-{self.config.max_coord_value}")

    def extend_tokenizer_vocabulary(
        self, tokenizer: PreTrainedTokenizer
    ) -> PreTrainedTokenizer:
        """
        Extend tokenizer vocabulary with new geometry tokens and coordinate tokens.

        Args:
            tokenizer: Original tokenizer to extend

        Returns:
            Extended tokenizer with new vocabulary
        """
        new_tokens = []

        # Add new geometry tokens (square, line)
        if self.config.new_geometry_tokens:
            for token in self.config.new_geometry_tokens:
                if token not in tokenizer.get_vocab():
                    new_tokens.append(token)

        # Add coordinate tokens if enabled
        if self.config.coordinate_tokens_enabled:
            for coord_token in self.coordinate_token_map.values():
                if coord_token not in tokenizer.get_vocab():
                    new_tokens.append(coord_token)

        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} new tokens to tokenizer vocabulary")
            tokenizer.add_tokens(new_tokens)
            logger.info(f"New vocabulary size: {len(tokenizer)}")
        else:
            logger.info(
                "No new tokens to add - vocabulary already contains required tokens"
            )

        return tokenizer

    def extend_model_embeddings(
        self, model: Qwen2VLForConditionalGeneration, tokenizer: PreTrainedTokenizer
    ) -> Qwen2VLForConditionalGeneration:
        """
        Extend model embeddings to accommodate new tokens.

        Args:
            model: Model to extend
            tokenizer: Extended tokenizer

        Returns:
            Model with extended embeddings
        """
        original_vocab_size = model.config.vocab_size
        new_vocab_size = len(tokenizer.get_vocab())

        if new_vocab_size > original_vocab_size:
            logger.info(
                f"🔧 Extending model embeddings from {original_vocab_size} to {new_vocab_size}"
            )

            # Resize token embeddings
            model.resize_token_embeddings(new_vocab_size)

            # Initialize new embeddings
            self._initialize_new_embeddings(
                model, original_vocab_size, new_vocab_size, tokenizer
            )

            logger.info(f"✅ Model embeddings extended successfully")
        else:
            logger.info("ℹ️ No embedding extension needed")

        return model

    def _initialize_new_embeddings(
        self,
        model: Qwen2VLForConditionalGeneration,
        original_vocab_size: int,
        new_vocab_size: int,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """
        Initialize embeddings for new tokens.

        Args:
            model: Model with extended embeddings
            original_vocab_size: Original vocabulary size
            new_vocab_size: New vocabulary size
            tokenizer: Extended tokenizer
        """
        vocab = tokenizer.get_vocab()

        # Get embedding layers
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        with torch.no_grad():
            # Initialize geometry tokens using existing geometry token embeddings
            self._initialize_geometry_tokens(
                input_embeddings, output_embeddings, vocab, tokenizer
            )

            # Initialize coordinate tokens using positional encoding approach
            if self.config.coordinate_tokens_enabled:
                self._initialize_coordinate_tokens(
                    input_embeddings, output_embeddings, vocab, original_vocab_size
                )

    def _initialize_geometry_tokens(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """Initialize geometry token embeddings using existing geometry tokens."""
        # Map new line tokens to appropriate reference tokens
        # Line tokens should be initialized from quad tokens (more semantically similar)
        # Rationale: Quadrilaterals are flexible geometric shapes like lines,
        # whereas boxes are constrained rectangles with less geometric flexibility
        reference_mapping = {
            "<|line_start|>": "<|quad_start|>",
            "<|line_end|>": "<|quad_end|>",
        }

        for new_token, ref_token in reference_mapping.items():
            if new_token in vocab and ref_token in vocab:
                new_id = vocab[new_token]
                ref_id = vocab[ref_token]

                # Copy embeddings from reference token
                input_embeddings.weight[new_id] = input_embeddings.weight[
                    ref_id
                ].clone()
                if output_embeddings.weight.shape[1] > new_id:
                    output_embeddings.weight[new_id] = output_embeddings.weight[
                        ref_id
                    ].clone()

                logger.debug(
                    f"🔧 Initialized {new_token} (ID: {new_id}) from {ref_token} (ID: {ref_id})"
                )

    def _initialize_coordinate_tokens(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        original_vocab_size: int,
    ) -> None:
        """Initialize coordinate token embeddings using vectorized positional encoding."""
        embedding_dim = input_embeddings.embedding_dim

        # Collect all coordinate token IDs that need initialization
        coord_token_ids = []
        coord_positions = []

        for i in range(self.config.max_coord_value + 1):
            coord_token = f"<|coord_{i}|>"
            if coord_token in vocab:
                coord_token_ids.append(vocab[coord_token])
                coord_positions.append(float(i))

        if not coord_token_ids:
            logger.info("No coordinate tokens found in vocabulary")
            return

        logger.info(
            f"🔧 Vectorized initialization of {len(coord_token_ids)} coordinate tokens..."
        )

        # Vectorized positional encoding computation
        # Create position tensor: [num_coords, 1]
        positions = torch.tensor(coord_positions, dtype=torch.float32).unsqueeze(1)

        # Create dimension indices: [1, embedding_dim//2]
        dim_indices = torch.arange(0, embedding_dim, 2, dtype=torch.float32).unsqueeze(
            0
        )

        # Compute div_term vectorized: [1, embedding_dim//2]
        div_term = torch.exp(
            dim_indices * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        # Compute positional encodings: [num_coords, embedding_dim//2]
        angle = (
            positions * div_term
        )  # Broadcasting: [num_coords, 1] * [1, embedding_dim//2]

        # Create full positional encoding tensor: [num_coords, embedding_dim]
        pos_encodings = torch.zeros(len(coord_token_ids), embedding_dim)
        pos_encodings[:, 0::2] = torch.sin(angle)  # Even indices
        if embedding_dim % 2 == 0:
            pos_encodings[:, 1::2] = torch.cos(angle)  # Odd indices
        else:
            pos_encodings[:, 1::2] = torch.cos(
                angle[:, :-1]
            )  # Handle odd embedding_dim

        # Scale and add randomness
        pos_encodings = pos_encodings * 0.1 + torch.randn_like(pos_encodings) * 0.02

        # Batch assign embeddings
        with torch.no_grad():
            for idx, coord_id in enumerate(coord_token_ids):
                input_embeddings.weight[coord_id] = pos_encodings[idx]
                if output_embeddings.weight.shape[1] > coord_id:
                    output_embeddings.weight[coord_id] = pos_encodings[idx]

        logger.info(
            f"✅ Initialized {len(coord_token_ids)} coordinate token embeddings"
        )

    def coordinates_to_tokens(self, coordinates: List[int]) -> List[str]:
        """
        Convert coordinate integers to coordinate tokens.

        Args:
            coordinates: List of coordinate integers

        Returns:
            List of coordinate token strings
        """
        if not self.config.coordinate_tokens_enabled:
            return [str(coord) for coord in coordinates]

        tokens = []
        for coord in coordinates:
            if coord > self.config.max_coord_value:
                logger.warning(
                    f"Coordinate {coord} exceeds max value {self.config.max_coord_value}, clipping"
                )
                coord = self.config.max_coord_value
            elif coord < 0:
                logger.warning(f"Negative coordinate {coord} found, setting to 0")
                coord = 0

            tokens.append(self.coordinate_token_map[coord])

        return tokens

    def tokens_to_coordinates(self, tokens: List[str]) -> List[int]:
        """
        Convert coordinate tokens back to integers.

        Args:
            tokens: List of coordinate token strings

        Returns:
            List of coordinate integers
        """
        if not self.config.coordinate_tokens_enabled:
            # Try to parse as regular integers
            coordinates = []
            for token in tokens:
                try:
                    coordinates.append(int(token))
                except ValueError:
                    logger.warning(
                        f"Cannot parse token '{token}' as coordinate integer"
                    )
                    coordinates.append(0)
            return coordinates

        coordinates = []
        for token in tokens:
            if token in self.reverse_coordinate_map:
                coordinates.append(self.reverse_coordinate_map[token])
            else:
                logger.warning(f"Unknown coordinate token: {token}")
                coordinates.append(0)

        return coordinates

    def wrap_object_with_tokens(self, obj: Dict[str, Any]) -> str:
        """
        Wrap object with appropriate special tokens based on geometry type.

        Args:
            obj: Object dictionary with geometry and description

        Returns:
            Formatted string with special tokens
        """
        desc = obj.get("desc", "")

        # Handle different geometry types
        if "bbox_2d" in obj:
            coords = obj["bbox_2d"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|bbox_start|>[{coord_str}]<|bbox_end|>"
            else:
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|bbox_start|>{coords}<|bbox_end|>"

        elif "quad" in obj:
            coords = obj["quad"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|quad_start|>[{coord_str}]<|quad_end|>"
            else:
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|quad_start|>{coords}<|quad_end|>"

        elif "line" in obj:
            coords = obj["line"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|line_start|>[{coord_str}]<|line_end|>"
            else:
                return f"<|obj_ref_start|>{desc}<|obj_ref_end|><|line_start|>{coords}<|line_end|>"

        else:
            # Raise error for unsupported geometry types
            available_keys = [k for k in obj.keys() if k not in ["desc"]]
            raise ValueError(
                f"Object contains unsupported geometry type. "
                f"Expected one of: bbox_2d, quad, line. "
                f"Found geometry keys: {available_keys}. "
                f"Full object: {obj}"
            )

    def extract_coordinates_from_tokens(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Extract coordinate sequences from tokenized input.

        Args:
            input_ids: Tokenized input tensor
            tokenizer: Tokenizer used for encoding

        Returns:
            List of (start_idx, end_idx, coordinates) tuples
        """
        coordinate_sequences = []

        if not self.config.coordinate_tokens_enabled:
            return coordinate_sequences

        # Convert to list for easier processing
        tokens = (
            input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        )

        # Find coordinate token sequences
        i = 0
        while i < len(tokens):
            token_str = tokenizer.decode([tokens[i]])

            if token_str in self.reverse_coordinate_map:
                # Found start of coordinate sequence
                start_idx = i
                coordinates = []

                # Collect consecutive coordinate tokens
                while i < len(tokens):
                    current_token = tokenizer.decode([tokens[i]])
                    if current_token in self.reverse_coordinate_map:
                        coordinates.append(self.reverse_coordinate_map[current_token])
                        i += 1
                    else:
                        break

                end_idx = i - 1
                coordinate_sequences.append((start_idx, end_idx, coordinates))
            else:
                i += 1

        return coordinate_sequences

    def create_coordinate_mask(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer
    ) -> torch.Tensor:
        """
        Create mask indicating coordinate token positions.

        Args:
            input_ids: Tokenized input tensor [seq_len]
            tokenizer: Tokenizer used for encoding

        Returns:
            Boolean mask tensor [seq_len] where True indicates coordinate tokens
        """
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        if not self.config.coordinate_tokens_enabled:
            return mask

        coordinate_sequences = self.extract_coordinates_from_tokens(
            input_ids, tokenizer
        )

        for start_idx, end_idx, _ in coordinate_sequences:
            mask[start_idx : end_idx + 1] = True

        return mask

    def get_geometry_token_ids(self, tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
        """
        Get token IDs for geometry start/end tokens.

        Args:
            tokenizer: Extended tokenizer

        Returns:
            Dictionary mapping token names to IDs
        """
        token_ids = {}

        # Standard geometry tokens (should already exist)
        standard_tokens = [
            "<|obj_ref_start|>",
            "<|obj_ref_end|>",
            "<|bbox_start|>",
            "<|bbox_end|>",
        ]

        # New geometry tokens
        all_tokens = standard_tokens + self.config.new_geometry_tokens

        for token in all_tokens:
            if token in tokenizer.get_vocab():
                token_ids[token] = tokenizer.get_vocab()[token]
            else:
                logger.warning(
                    f"Geometry token '{token}' not found in tokenizer vocabulary"
                )

        return token_ids

    def validate_coordinate_range(self, coordinates: List[int]) -> List[int]:
        """
        Validate and clip coordinates to valid range.

        Args:
            coordinates: List of coordinate values

        Returns:
            Validated coordinate list with values clipped to valid range
        """
        validated = []
        for coord in coordinates:
            if coord < 0:
                logger.warning(f"Negative coordinate {coord} found, clipping to 0")
                validated.append(0)
            elif coord > self.config.max_coord_value:
                logger.warning(
                    f"Coordinate {coord} exceeds max {self.config.max_coord_value}, clipping"
                )
                validated.append(self.config.max_coord_value)
            else:
                validated.append(coord)

        return validated

    def get_coordinate_token_range(
        self, tokenizer: PreTrainedTokenizer
    ) -> Tuple[int, int]:
        """
        Get the token ID range for coordinate tokens.

        Args:
            tokenizer: Extended tokenizer

        Returns:
            (min_coord_token_id, max_coord_token_id) tuple
        """
        if not self.config.coordinate_tokens_enabled:
            return (0, 0)

        coord_token_ids = []
        for coord_token in self.coordinate_token_map.values():
            if coord_token in tokenizer.get_vocab():
                coord_token_ids.append(tokenizer.get_vocab()[coord_token])

        if not coord_token_ids:
            logger.warning("No coordinate tokens found in tokenizer vocabulary")
            return (0, 0)

        return (min(coord_token_ids), max(coord_token_ids))
