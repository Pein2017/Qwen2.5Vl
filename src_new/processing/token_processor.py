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
    """Get rank-aware logger for token processing (strict, no fallback)."""
    from ..utils.rank_aware_logging import get_rank_aware_logger

    return get_rank_aware_logger("token_processor")


logger = get_token_logger()

# Global cache for tokenizer extensions to avoid repeated processing in multi-GPU setups
_TOKENIZER_EXTENSION_CACHE = {}


def clear_tokenizer_cache():
    """Clear the tokenizer extension cache."""
    global _TOKENIZER_EXTENSION_CACHE
    _TOKENIZER_EXTENSION_CACHE.clear()
    logger.info("🧹 Tokenizer extension cache cleared")


@dataclass
class TokenConfig:
    """Configuration for token processing."""

    max_coord_value: int
    coordinate_tokens_enabled: bool = False
    new_geometry_tokens: Optional[List[str]] = None
    # Make lazy initialization behavior explicit to avoid hasattr checks elsewhere
    lazy_coordinate_init: bool = True

    def __post_init__(self):
        if self.new_geometry_tokens is None:
            # Only add line tokens - quad and box tokens already exist in Qwen2.5-VL
            self.new_geometry_tokens = [
                "<|line_start|>",
                "<|line_end|>",
            ]


@dataclass
class OptimizedTokenConfig(TokenConfig):
    """
    Enhanced token configuration with ms-swift inspired optimizations.
    """

    # ms-swift optimization: vocabulary padding
    pad_vocab_to_multiple_of: int = 128

    # ms-swift optimization: smart initialization
    use_smart_initialization: bool = True

    # ms-swift optimization: use add_special_tokens
    use_special_tokens_api: bool = True

    # Performance optimization: initialization standard deviation
    init_std: Optional[float] = None  # Auto-detect from model if None

    # Memory optimization: lazy initialization
    lazy_coordinate_init: bool = True


class TokenProcessor:
    """
    Handles coordinate token conversion and tokenizer vocabulary extension.

    Supports both standard mode (coordinates as integers) and coordinate token mode
    (coordinates as special tokens). Manages tokenizer vocabulary extension for
    new geometry types.
    """

    def __init__(self, config: TokenConfig) -> None:
        """
        Initialize token processor with lazy loading optimization.

        Args:
            config: Token processing configuration
        """
        self.config = config
        self.coordinate_token_map: Dict[int, str] = {}
        self.reverse_coordinate_map: Dict[str, int] = {}
        self._maps_built = False

        # OPTIMIZATION: Lazy initialization - defer expensive operations
        # Build coordinate token maps only when actually needed
        if config.coordinate_tokens_enabled and not config.lazy_coordinate_init:
            self._build_coordinate_token_maps()

    def _build_coordinate_token_maps(self) -> None:
        """Build coordinate token mapping dictionaries with lazy loading."""
        if self._maps_built:
            return  # Already built

        for coord in range(self.config.max_coord_value + 1):
            token = f"<|coord_{coord}|>"
            self.coordinate_token_map[coord] = token
            self.reverse_coordinate_map[token] = coord

        self._maps_built = True
        logger.info(f"Built coordinate token maps for 0-{self.config.max_coord_value}")

    def _ensure_maps_built(self) -> None:
        """Ensure coordinate token maps are built (lazy initialization)."""
        if not self._maps_built and self.config.coordinate_tokens_enabled:
            self._build_coordinate_token_maps()

    def extend_tokenizer_vocabulary(
        self, tokenizer: PreTrainedTokenizer
    ) -> PreTrainedTokenizer:
        """
        Extend tokenizer vocabulary with ms-swift inspired optimizations and caching.

        Args:
            tokenizer: Original tokenizer to extend

        Returns:
            Extended tokenizer with new vocabulary
        """
        # Create cache key based on tokenizer state and config
        vocab_size = len(tokenizer.get_vocab())
        cache_key = f"{vocab_size}_{self.config.coordinate_tokens_enabled}_{self.config.max_coord_value}"

        # Disable cache for strict deterministic behavior

        new_tokens = []

        # Add exactly 2 line tokens first, in fixed order
        vocab = tokenizer.get_vocab()
        for t in ("<|line_start|>", "<|line_end|>"):
            if t not in vocab:
                new_tokens.append(t)

        # Add exactly 1025 coordinate tokens in fixed order: <|coord_0|>.. <|coord_1024|>
        if self.config.coordinate_tokens_enabled:
            for coord in range(self.config.max_coord_value + 1):  # 0..1024 inclusive
                coord_token = f"<|coord_{coord}|>"
                if coord_token not in vocab:
                    new_tokens.append(coord_token)

        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} new tokens to tokenizer vocabulary")
            logger.info(
                f"Sample tokens: {new_tokens[:5]}..."
            )  # Show first 5 tokens for debugging

            # MS-SWIFT OPTIMIZATION: Use add_special_tokens for better integration
            # This ensures proper handling by HuggingFace tokenizers
            if len(new_tokens) > 0:
                # For coordinate tokens, use add_special_tokens for better performance
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )
                logger.info(
                    f"✅ Added {num_added} special tokens using ms-swift approach"
                )

            logger.info(f"New vocabulary size: {len(tokenizer.get_vocab())}")

        # Strict post-conditions
        final_vocab = tokenizer.get_vocab()
        base_len_known = 151665
        # Expect base + 2 line tokens + (max_coord_value + 1) coordinate tokens
        expected_len = base_len_known + 2 + (self.config.max_coord_value + 1)

        # Verify presence of all required tokens
        missing = []
        for t in ("<|line_start|>", "<|line_end|>"):
            if t not in final_vocab:
                missing.append(t)
        for coord in range(self.config.max_coord_value + 1):
            tok = f"<|coord_{coord}|>"
            if tok not in final_vocab:
                missing.append(tok)
        if missing:
            raise ValueError(
                f"Missing required tokens after extension: {missing[:5]} ... total={len(missing)}"
            )

        # If we started from official base length, assert exact IDs and final size
        if len(vocab) == base_len_known and self.config.coordinate_tokens_enabled:
            line_start_id = final_vocab["<|line_start|>"]
            line_end_id = final_vocab["<|line_end|>"]
            coord_ids = [
                final_vocab[f"<|coord_{i}|>"]
                for i in range(self.config.max_coord_value + 1)
            ]
            assert line_start_id == 151665 and line_end_id == 151666
            expected_coord_start = 151667
            expected_coord_end_inclusive = (
                expected_coord_start + self.config.max_coord_value
            )
            assert (
                min(coord_ids) == expected_coord_start
                and max(coord_ids) == expected_coord_end_inclusive
            )
            assert len(final_vocab) == expected_len

        return tokenizer

    def extend_model_embeddings(
        self, model: Qwen2VLForConditionalGeneration, tokenizer: PreTrainedTokenizer
    ) -> Qwen2VLForConditionalGeneration:
        """
        Extend model embeddings to accommodate new tokens with optimized performance.

        Args:
            model: Model to extend
            tokenizer: Extended tokenizer

        Returns:
            Model with extended embeddings
        """
        import torch  # local import to ensure availability for no_grad snapshot

        original_vocab_size = model.config.vocab_size
        new_vocab_size = len(tokenizer.get_vocab())

        # Snapshot original rows to ensure they remain unchanged after extension
        with torch.no_grad():
            original_snapshot = (
                model.get_input_embeddings()
                .weight[:original_vocab_size]
                .detach()
                .clone()
            )

        if new_vocab_size > original_vocab_size:
            logger.info(
                f"🔧 Extending model embeddings from {original_vocab_size} to {new_vocab_size}"
            )

            # OPTIMIZATION 1: Check if embeddings are already extended
            current_embed_size = model.get_input_embeddings().weight.shape[0]
            if current_embed_size >= new_vocab_size:
                logger.info(
                    f"🚀 Embeddings already extended to {current_embed_size}, skipping resize"
                )
                # Still run initialization to ensure coordinate tokens are properly set
                self._smart_initialize_new_embeddings(
                    model, original_vocab_size, current_embed_size, tokenizer
                )
                # Strict validation and logging
                self._validate_tokenizer_embedding_alignment(
                    model, tokenizer, original_vocab_size, original_snapshot
                )
                return model

            # Pad rows to multiple of 128 (strict target): ceil(152692 / 128) * 128 = 152704
            import math

            import torch

            target_rows = math.ceil(len(tokenizer.get_vocab()) / 128) * 128

            with torch.no_grad():
                model.resize_token_embeddings(
                    target_rows, mean_resizing=False, pad_to_multiple_of=128
                )

            # Deterministic initialization for new rows
            self._smart_initialize_new_embeddings(
                model, original_vocab_size, target_rows, tokenizer
            )

            # Strict validation and logging
            self._validate_tokenizer_embedding_alignment(
                model, tokenizer, original_vocab_size, original_snapshot
            )

            logger.info(f"✅ Model embeddings extended successfully with optimizations")
        else:
            logger.info("ℹ️ No embedding extension needed")

        return model

    def _smart_initialize_new_embeddings(
        self,
        model: Qwen2VLForConditionalGeneration,
        original_vocab_size: int,
        padded_vocab_size: int,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """
        Smart embedding initialization inspired by ms-swift.
        Only initializes embeddings that are actually zero (need initialization).

        Args:
            model: Model with extended embeddings
            original_vocab_size: Original vocabulary size
            padded_vocab_size: Padded vocabulary size (multiple of 128)
            tokenizer: Extended tokenizer
        """
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        # MS-SWIFT TECHNIQUE: Only initialize embeddings that are zero
        with torch.no_grad():
            # Check which embeddings need initialization (are all zeros)
            input_mask = (input_embeddings.weight == 0).all(dim=-1)
            num_to_initialize = input_mask.sum().item()

            if num_to_initialize > 0:
                logger.info(
                    f"🔧 Smart initialization: {num_to_initialize} embeddings need initialization"
                )

                # Initialize only the embeddings that need it
                embedding_dim = input_embeddings.embedding_dim

                # Use model's initialization method if available, otherwise use simple random
                if hasattr(model.config, "initializer_range"):
                    init_std = model.config.initializer_range
                else:
                    init_std = 0.02

                # Initialize input embeddings (match dtype/device)
                in_w = input_embeddings.weight
                new_embeddings = (
                    torch.randn(
                        num_to_initialize,
                        embedding_dim,
                        device=in_w.device,
                        dtype=in_w.dtype,
                    )
                    * init_std
                )
                in_w[input_mask] = new_embeddings

                # Initialize output embeddings if they exist and need initialization
                if hasattr(output_embeddings, "weight"):
                    out_w = output_embeddings.weight
                    output_mask = (out_w == 0).all(dim=0)
                    num_output_to_init = output_mask.sum().item()

                    if num_output_to_init > 0:
                        new_output_embeddings = (
                            torch.randn(
                                embedding_dim,
                                num_output_to_init,
                                device=out_w.device,
                                dtype=out_w.dtype,
                            )
                            * init_std
                        )
                        out_w[:, output_mask] = new_output_embeddings

                logger.info(
                    f"✅ Smart-initialized {num_to_initialize} input and {num_output_to_init if 'num_output_to_init' in locals() else 0} output embeddings"
                )
            else:
                logger.info(
                    "ℹ️ No embeddings need initialization (all already initialized)"
                )

        # Initialize line tokens deterministically from quad tokens
        vocab = tokenizer.get_vocab()
        self._initialize_geometry_tokens(
            input_embeddings, output_embeddings, vocab, tokenizer
        )

        # Initialize coordinate tokens deterministically
        if self.config.coordinate_tokens_enabled:
            self._initialize_coordinate_tokens_optimized(
                input_embeddings,
                output_embeddings,
                vocab,
                original_vocab_size,
            )

    def _validate_tokenizer_embedding_alignment(
        self,
        model: Qwen2VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        original_vocab_size: int,
        original_snapshot: torch.Tensor,
    ) -> None:
        """Strict validation of tokenizer and embedding alignment, with rich logging.

        Validates:
        - Token counts (base 151665; +1027 = 152692)
        - Embedding matrix rows padded to multiple of 128 (152704)
        - Geometry token IDs and initialization linkage (if reference quad tokens exist)
        - Coordinate token range and non-zero initialization
        - Original pretrained rows unchanged (0..151664)
        """
        import math

        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        in_w = model.get_input_embeddings().weight
        in_rows = in_w.shape[0]
        hidden = in_w.shape[1]

        base_len_known = 151665
        expected_added = 1027  # 1025 coords + 2 geometry
        expected_vocab = base_len_known + expected_added  # 152692
        expected_rows = math.ceil(expected_vocab / 128) * 128  # 152704

        logger.debug(
            f"[VALIDATION] base={base_len_known}, added={expected_added}, final_vocab={expected_vocab}"
        )
        logger.debug(
            f"[VALIDATION] tokenizer_vocab_size={vocab_size}, input_rows={in_rows}, hidden={hidden}"
        )

        # Token count validation
        if vocab_size != expected_vocab:
            raise ValueError(
                f"Tokenizer vocab size mismatch: expected {expected_vocab}, got {vocab_size}"
            )

        # Embedding matrix alignment
        if in_rows < expected_vocab:
            raise ValueError(
                f"Embedding rows insufficient: expected >= {expected_vocab}, got {in_rows}"
            )
        if in_rows != expected_rows:
            raise AssertionError(
                f"Embedding rows must equal strict pad multiple: expected {expected_rows}, got {in_rows}"
            )

        # Geometry tokens: IDs and initialization copy if quad tokens present
        line_start_id = vocab.get("<|line_start|>")
        line_end_id = vocab.get("<|line_end|>")
        quad_start_id = vocab.get("<|quad_start|>")
        quad_end_id = vocab.get("<|quad_end|>")
        logger.debug(
            f"[VALIDATION] IDs: line_start={line_start_id}, line_end={line_end_id}, "
            f"quad_start={quad_start_id}, quad_end={quad_end_id}"
        )
        if quad_start_id is not None and quad_end_id is not None:
            with torch.no_grad():
                ls_eq = torch.allclose(in_w[line_start_id], in_w[quad_start_id])
                le_eq = torch.allclose(in_w[line_end_id], in_w[quad_end_id])
            if not (ls_eq and le_eq):
                raise AssertionError(
                    "Geometry token initialization failed: line tokens not copied from quad tokens"
                )

        # Coordinate tokens range and non-zero init
        coord_ids = [vocab[f"<|coord_{i}|>"] for i in range(0, 1025)]
        if min(coord_ids) != 151667 or max(coord_ids) != 151667 + 1024:
            raise AssertionError(
                "Coordinate token ID range mismatch (expected 151667..152691)"
            )

        with torch.no_grad():
            zeros = torch.zeros_like(in_w[0])
            for cid in coord_ids:
                vec = in_w[cid]
                if torch.allclose(vec, zeros):
                    raise AssertionError(
                        f"Coordinate embedding at id={cid} is all zeros"
                    )

        # Original rows unchanged (strictly for base pretrained region only).
        # Note: Some official checkpoints ship embedding matrices larger than the known
        # base vocabulary (151665) due to internal padding or reserved slots. We only
        # enforce immutability for the true base region [0, 151665), allowing
        # initialization of new geometry/coordinate tokens that may fall below
        # model.config.vocab_size but are not part of the original base.
        freeze_rows = min(original_vocab_size, base_len_known)
        with torch.no_grad():
            cur_snapshot = in_w[:freeze_rows].detach().clone()
            if not torch.allclose(cur_snapshot, original_snapshot[:freeze_rows]):
                raise AssertionError(
                    "Original pretrained base embedding rows changed during extension"
                )

        logger.info(
            "✅ Tokenizer/Embedding alignment validated: vocab=152692, rows=%d, hidden=%d",
            in_rows,
            hidden,
        )

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

        # Perform updates without tracking gradients to avoid in-place ops on leaf Variables
        with torch.no_grad():
            for new_token, ref_token in reference_mapping.items():
                if new_token in vocab and ref_token in vocab:
                    new_id = vocab[new_token]
                    ref_id = vocab[ref_token]

                    # Copy embeddings from reference token safely
                    input_embeddings.weight[new_id].copy_(
                        input_embeddings.weight[ref_id]
                    )

                    # Update LM head if addressable; support both [vocab, hidden] and [hidden, vocab]
                    if hasattr(output_embeddings, "weight") and isinstance(
                        output_embeddings.weight, torch.Tensor
                    ):
                        w = output_embeddings.weight
                        if w.dim() == 2:
                            if w.shape[0] > new_id:
                                # Common HF layout: [vocab, hidden]
                                w[new_id].copy_(w[ref_id])
                            elif w.shape[1] > new_id:
                                # Alternative layout: [hidden, vocab]
                                w[:, new_id].copy_(w[:, ref_id])

                    logger.debug(
                        f"🔧 Initialized {new_token} (ID: {new_id}) from {ref_token} (ID: {ref_id})"
                    )

    def _initialize_coordinate_tokens_optimized(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        original_vocab_size: int,
    ) -> None:
        """
        Deterministically initialize coordinate token embeddings regardless of prior values.
        - Input embeddings: normal(0, init_std)
        - Output embeddings: same vectors transposed into LM head columns
        """
        if not self.config.coordinate_tokens_enabled:
            return

        # Gather coordinate token IDs (0..max)
        coord_token_ids = [
            vocab[f"<|coord_{i}|>"]
            for i in range(self.config.max_coord_value + 1)
            if f"<|coord_{i}|>" in vocab
        ]
        if not coord_token_ids:
            logger.info("No coordinate tokens found in vocabulary")
            return

        embedding_dim = input_embeddings.embedding_dim
        device = input_embeddings.weight.device

        with torch.no_grad():
            existing_std = input_embeddings.weight[:original_vocab_size].std().item()
            init_std = min(existing_std, 0.02) if existing_std > 0 else 0.02

            coord_token_ids_tensor = torch.tensor(coord_token_ids, device=device)
            # Match dtype/device to input embeddings to avoid dtype mismatch (e.g., bf16)
            in_w = input_embeddings.weight
            coord_embeddings = (
                torch.randn(
                    len(coord_token_ids_tensor),
                    embedding_dim,
                    device=in_w.device,
                    dtype=in_w.dtype,
                )
                * init_std
            )

            in_w[coord_token_ids_tensor] = coord_embeddings

            if hasattr(output_embeddings, "weight") and isinstance(
                output_embeddings.weight, torch.Tensor
            ):
                w = output_embeddings.weight
                max_id = int(coord_token_ids_tensor.max())
                if w.dim() == 2:
                    if w.shape[0] > max_id:
                        # Common HF layout: [vocab, hidden]
                        w[coord_token_ids_tensor] = coord_embeddings
                    elif w.shape[1] > max_id:
                        # Alternative layout: [hidden, vocab]
                        w[:, coord_token_ids_tensor] = coord_embeddings.T

        logger.info("✅ Deterministic coordinate token initialization completed")

    def _initialize_coordinate_tokens(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        original_vocab_size: int,
    ) -> None:
        # Deterministic alias
        return self._initialize_coordinate_tokens_optimized(
            input_embeddings, output_embeddings, vocab, original_vocab_size
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

        # OPTIMIZATION: Generate tokens on-demand without building full maps
        # This maintains maximum lazy loading efficiency
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

            # Generate token on-demand
            tokens.append(f"<|coord_{coord}|>")

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

        # OPTIMIZATION: Parse coordinate tokens on-demand without building reverse map
        coordinates = []
        for token in tokens:
            # Parse coordinate token format: <|coord_N|>
            if token.startswith("<|coord_") and token.endswith("|>"):
                try:
                    coord_str = token[8:-2]  # Extract N from <|coord_N|>
                    coord = int(coord_str)
                    if 0 <= coord <= self.config.max_coord_value:
                        coordinates.append(coord)
                    else:
                        logger.warning(f"Coordinate {coord} out of range, setting to 0")
                        coordinates.append(0)
                except ValueError:
                    logger.warning(f"Cannot parse coordinate token: {token}")
                    coordinates.append(0)
            else:
                logger.warning(f"Unknown coordinate token format: {token}")
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
        if "desc" not in obj:
            raise ValueError("Object missing required 'desc' field")
        desc = obj["desc"]

        # Handle different geometry types
        if "bbox_2d" in obj:
            coords = obj["bbox_2d"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>[{coord_str}]<|box_end|>"
            else:
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>{coords}<|box_end|>"

        elif "quad" in obj:
            coords = obj["quad"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|quad_start|>[{coord_str}]<|quad_end|>"
            else:
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|quad_start|>{coords}<|quad_end|>"

        elif "line" in obj:
            coords = obj["line"]
            coord_tokens = self.coordinates_to_tokens(coords)
            if self.config.coordinate_tokens_enabled:
                coord_str = ", ".join(coord_tokens)
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|line_start|>[{coord_str}]<|line_end|>"
            else:
                return f"<|object_ref_start|>{desc}<|object_ref_end|><|line_start|>{coords}<|line_end|>"

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

            # OPTIMIZATION: Check coordinate token format on-demand
            if token_str.startswith("<|coord_") and token_str.endswith("|>"):
                # Found start of coordinate sequence
                start_idx = i
                coordinates = []

                # Collect consecutive coordinate tokens
                while i < len(tokens):
                    current_token = tokenizer.decode([tokens[i]])
                    if current_token.startswith("<|coord_") and current_token.endswith(
                        "|>"
                    ):
                        try:
                            coord_str = current_token[
                                8:-2
                            ]  # Extract N from <|coord_N|>
                            coord = int(coord_str)
                            if 0 <= coord <= self.config.max_coord_value:
                                coordinates.append(coord)
                            else:
                                coordinates.append(0)
                        except ValueError:
                            coordinates.append(0)
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
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
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

        # OPTIMIZATION: Generate coordinate token IDs on-demand
        coord_token_ids = []
        vocab = tokenizer.get_vocab()
        for coord in range(self.config.max_coord_value + 1):
            coord_token = f"<|coord_{coord}|>"
            if coord_token in vocab:
                coord_token_ids.append(vocab[coord_token])

        if not coord_token_ids:
            logger.warning("No coordinate tokens found in tokenizer vocabulary")
            return (0, 0)

        return (min(coord_token_ids), max(coord_token_ids))
