#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token processor for Qwen2.5-VL coordinate token handling.

Implements coordinate token conversion, tokenizer vocabulary extension,
and special token wrapping for multi-geometry annotations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("token_processor")


@dataclass
class TokenConfig:
    """Configuration for token processing."""

    max_coord_value: int
    coordinate_init_mode: str  # Required: "ms_mean" or "fourier_ramp"
    coordinate_tokens_enabled: bool = False
    new_geometry_tokens: Optional[List[str]] = None
    # Make lazy initialization behavior explicit to avoid hasattr checks elsewhere
    lazy_coordinate_init: bool = True

    def __post_init__(self):
        # Validate coordinate_init_mode
        allowed_modes = {"ms_mean", "fourier_ramp"}
        if self.coordinate_init_mode not in allowed_modes:
            raise ValueError(
                f"coordinate_init_mode must be one of {sorted(allowed_modes)}, "
                f"got {self.coordinate_init_mode!r}"
            )
        # Do NOT auto-add geometry tokens here; rely on checkpoint vocabulary
        if self.new_geometry_tokens is None:
            self.new_geometry_tokens = []


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
        self, tokenizer: PreTrainedTokenizerBase
    ) -> PreTrainedTokenizerBase:
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

        # NOTE: Do NOT auto-add geometry wrapper tokens; rely on checkpoint vocab
        # Only consider coordinate tokens below when enabled

        # Add exactly max_coord_value+1 coordinate tokens in fixed order when enabled
        if self.config.coordinate_tokens_enabled:
            for coord in range(self.config.max_coord_value + 1):  # 0..max inclusive
                coord_token = f"<|coord_{coord}|>"
                if coord_token not in tokenizer.get_vocab():
                    new_tokens.append(coord_token)

        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} new tokens to tokenizer vocabulary")
            logger.info(
                f"Sample tokens: {new_tokens[:5]}..."
            )  # Show first 5 tokens for debugging

            # MS-SWIFT OPTIMIZATION: Use add_special_tokens for better integration
            if len(new_tokens) > 0:
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )
                logger.info(
                    f"✅ Added {num_added} special tokens using ms-swift approach"
                )

            logger.info(f"New vocabulary size: {len(tokenizer.get_vocab())}")

        # Post-conditions: verify presence (not absolute IDs)
        final_vocab = tokenizer.get_vocab()
        missing = []
        # No geometry token post-check here; strict validators are called elsewhere
        if self.config.coordinate_tokens_enabled:
            for coord in range(self.config.max_coord_value + 1):
                tok = f"<|coord_{coord}|>"
                if tok not in final_vocab:
                    missing.append(tok)
        if missing:
            raise ValueError(
                f"Missing required tokens after extension: {missing[:5]} ... total={len(missing)}"
            )

        return tokenizer

    def extend_model_embeddings(
        self, model: Qwen2_5_VLForConditionalGeneration, tokenizer: PreTrainedTokenizerBase
    ) -> Qwen2_5_VLForConditionalGeneration:
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
            input_embeddings = model.get_input_embeddings()
            original_snapshot = (
                input_embeddings.weight[:original_vocab_size]  # type: ignore
                .detach()
                .clone()
            )

        if new_vocab_size > original_vocab_size:
            logger.info(
                f"🔧 Extending model embeddings from {original_vocab_size} to {new_vocab_size}"
            )

            # OPTIMIZATION 1: Check if embeddings are already extended
            input_embeddings = model.get_input_embeddings()
            current_embed_size = input_embeddings.weight.shape[0]  # type: ignore
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

            # Pad rows to multiple of 128 (strict target): ceil(vocab/128)*128
            import inspect
            import math

            import torch

            target_rows = math.ceil(len(tokenizer.get_vocab()) / 128) * 128

            # Check if mean_resizing is supported
            supports_mean = (
                "mean_resizing"
                in inspect.signature(model.resize_token_embeddings).parameters
            )

            with torch.no_grad():
                if self.config.coordinate_init_mode == "ms_mean" and supports_mean:
                    logger.info(
                        "🎯 Using ms-swift style mean_resizing for neutral initialization"
                    )
                    model.resize_token_embeddings(
                        target_rows, mean_resizing=True, pad_to_multiple_of=128
                    )
                else:
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
        model: Qwen2_5_VLForConditionalGeneration,
        original_vocab_size: int,
        padded_vocab_size: int,
        tokenizer: PreTrainedTokenizerBase,
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
                in_w[input_mask] = new_embeddings  # type: ignore

                # Initialize output embeddings if they exist and need initialization
                if hasattr(output_embeddings, "weight"):
                    out_w = output_embeddings.weight
                    if out_w.dim() == 2:
                        output_mask = (
                            (out_w == 0).all(dim=0)
                            if out_w.shape[0] == embedding_dim
                            else (out_w == 0).all(dim=1)
                        )
                        num_output_to_init = output_mask.sum().item()
                        if num_output_to_init > 0:
                            if out_w.shape[0] == embedding_dim:
                                # [hidden, vocab]
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
                            else:
                                # [vocab, hidden]
                                new_output_embeddings = (
                                    torch.randn(
                                        num_output_to_init,
                                        embedding_dim,
                                        device=out_w.device,
                                        dtype=out_w.dtype,
                                    )
                                    * init_std
                                )
                                out_w[output_mask, :] = new_output_embeddings

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

        # Initialize coordinate tokens deterministically (selectable mode)
        if self.config.coordinate_tokens_enabled:
            mode = self.config.coordinate_init_mode

            if mode == "ms_mean":
                logger.info(
                    "🎯 Applying ms-swift style neutral initialization for coordinate tokens"
                )
                self._initialize_coordinate_tokens_ms_mean(
                    input_embeddings,
                    output_embeddings,
                    vocab,
                    original_vocab_size,
                    padded_vocab_size,
                )
            elif mode == "fourier_ramp":
                logger.info(
                    "🎯 Applying Fourier ramp initialization for coordinate tokens"
                )
                self._initialize_coordinate_tokens_fourier_ramp(
                    input_embeddings,
                    output_embeddings,
                    vocab,
                    original_vocab_size,
                )
            else:
                # This should never happen due to validation in __post_init__
                raise ValueError(f"Invalid coordinate_init_mode: {mode}")

    def _validate_tokenizer_embedding_alignment(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizerBase,
        original_vocab_size: int,
        original_snapshot: torch.Tensor,
    ) -> None:
        """
        Strict validation of tokenizer and embedding alignment, with rich logging.

        Validates:
        - Embedding matrix rows padded to multiple of 128
        - Geometry token IDs and initialization linkage (if reference quad tokens exist)
        - Coordinate token non-zero initialization (range derived from tokenizer)
        - Original pretrained rows unchanged in the true base region
        """
        
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        input_embeddings = model.get_input_embeddings()
        in_w = input_embeddings.weight  # type: ignore
        in_rows = in_w.shape[0]  # type: ignore
        hidden = in_w.shape[1]  # type: ignore

        logger.debug(
            f"[VALIDATION] tokenizer_vocab_size={vocab_size}, input_rows={in_rows}, hidden={hidden}"
        )

        # Ensure embeddings can cover vocab and are padded to 128
        if in_rows < vocab_size:
            raise ValueError(
                f"Embedding rows insufficient: expected >= {vocab_size}, got {in_rows}"
            )
        if in_rows % 128 != 0:
            raise AssertionError(
                f"Embedding rows should be padded to multiple of 128, got {in_rows}"
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
        if quad_start_id is not None and quad_end_id is not None and line_start_id is not None and line_end_id is not None:
            with torch.no_grad():
                ls_eq = torch.allclose(in_w[line_start_id], in_w[quad_start_id])  # type: ignore
                le_eq = torch.allclose(in_w[line_end_id], in_w[quad_end_id])  # type: ignore
            if not (ls_eq and le_eq):
                raise AssertionError(
                    "Geometry token initialization failed: line tokens not copied from quad tokens"
                )

        # Coordinate tokens range and non-zero init (only when enabled)
        if self.config.coordinate_tokens_enabled:
            # Derive coordinate range from tokenizer rather than fixed IDs
            coord_ids = [
                vocab[t]
                for t in vocab.keys()
                if isinstance(t, str) and t.startswith("<|coord_") and t.endswith("|>")
            ]
            if not coord_ids:
                raise AssertionError(
                    "Coordinate tokens not found in tokenizer after extension"
                )

            with torch.no_grad():
                zeros = torch.zeros_like(in_w[0])  # type: ignore
                for cid in coord_ids:
                    vec = in_w[cid]  # type: ignore
                    if torch.allclose(vec, zeros):
                        raise AssertionError(
                            f"Coordinate embedding at id={cid} is all zeros"
                        )

        # Original rows unchanged (strictly for base pretrained region only).
        # Note: Some official checkpoints ship embedding matrices larger than the
        # true base vocabulary due to internal padding or reserved slots. We only
        # enforce immutability for the true base region [0, original_vocab_size).
        # Freeze only the true base region: derive max base token id by excluding newly added tokens
        base_token_ids = [
            tid
            for tok, tid in vocab.items()
            if not (isinstance(tok, str) and (tok.startswith("<|coord_") or tok in ("<|line_start|>", "<|line_end|>")))
        ]
        if not base_token_ids:
            logger.debug("[VALIDATION] No base token ids detected for immutability check")
        else:
            with torch.no_grad():
                if original_snapshot.shape[0] <= max(base_token_ids):
                    logger.debug("[VALIDATION] Skipping base immutability check due to size mismatch")
                else:
                    if not torch.allclose(
                        original_snapshot[: original_vocab_size],
                        input_embeddings.weight[: original_vocab_size],  # type: ignore
                        atol=1e-6,
                        rtol=1e-6,
                    ):
                        raise AssertionError("Pretrained embedding rows changed unexpectedly during extension")

    def _initialize_new_embeddings(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        original_vocab_size: int,
        new_vocab_size: int,
        tokenizer: PreTrainedTokenizerBase,
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
        tokenizer: PreTrainedTokenizerBase,
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

    def _initialize_coordinate_tokens_fourier_ramp(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        original_vocab_size: int,
    ) -> None:
        """
        Initialize coordinate token embeddings using Fourier + ramp features.

        This method creates embeddings with ordinal structure using sinusoidal features
        plus linear ramps to encode coordinate ordering information.
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
            in_w = input_embeddings.weight
            out_w = getattr(output_embeddings, "weight", None)

            # Match std to existing embeddings
            existing_std = in_w[:original_vocab_size].std().item()
            init_std = min(existing_std, 0.02) if existing_std > 0 else 0.02

            coord_token_ids_tensor = torch.tensor(coord_token_ids, device=device)
            K = self.config.max_coord_value

            # Create Fourier + ramp features
            coord_embeddings = torch.zeros(
                len(coord_token_ids_tensor),
                embedding_dim,
                device=device,
                dtype=in_w.dtype,
            )

            for i, coord_value in enumerate(range(self.config.max_coord_value + 1)):
                u = coord_value / K  # Normalized coordinate [0, 1]
                u_tensor = torch.tensor(u, device=device, dtype=torch.float32)

                # Fill first dimensions with Fourier features
                fourier_dims = min(
                    embedding_dim // 4, 64
                )  # Use up to 64 dims for Fourier
                for m in range(fourier_dims // 2):
                    if 2 * m < embedding_dim:
                        freq = 2**m
                        coord_embeddings[i, 2 * m] = torch.sin(
                            2 * torch.pi * u_tensor * freq
                        )
                    if 2 * m + 1 < embedding_dim:
                        freq = 2**m
                        coord_embeddings[i, 2 * m + 1] = torch.cos(
                            2 * torch.pi * u_tensor * freq
                        )

                # Add linear ramp features
                ramp_start = fourier_dims
                if ramp_start < embedding_dim:
                    coord_embeddings[i, ramp_start] = u_tensor  # Linear ramp
                if ramp_start + 1 < embedding_dim:
                    coord_embeddings[i, ramp_start + 1] = (
                        u_tensor - 0.5
                    ) ** 2  # Quadratic
                if ramp_start + 2 < embedding_dim:
                    coord_embeddings[i, ramp_start + 2] = u_tensor**3  # Cubic

            # Scale to match pretrained std and mean-center
            coord_embeddings = coord_embeddings * init_std
            coord_embeddings = coord_embeddings - coord_embeddings.mean(
                dim=0, keepdim=True
            )

            in_w[coord_token_ids_tensor] = coord_embeddings

            # Mirror into LM head if it exists
            if isinstance(out_w, torch.Tensor) and out_w.dim() == 2:
                max_id = int(coord_token_ids_tensor.max())
                if out_w.shape[0] > max_id:
                    # [vocab, hidden] layout
                    out_w[coord_token_ids_tensor] = coord_embeddings
                elif out_w.shape[1] > max_id:
                    # [hidden, vocab] layout
                    out_w[:, coord_token_ids_tensor] = coord_embeddings.T

        logger.info("✅ Coordinate tokens initialized with Fourier + ramp features")

    def _initialize_coordinate_tokens_ms_mean(
        self,
        input_embeddings: torch.nn.Embedding,
        output_embeddings: torch.nn.Linear,
        vocab: Dict[str, int],
        original_vocab_size: int,
        padded_vocab_size: int,
    ) -> None:
        """
        Initialize coordinate token embeddings using ms-swift style neutral initialization.

        This method applies neutral initialization where new coordinate tokens start
        centered at the mean of existing embeddings with small noise for symmetry breaking.
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
            in_w = input_embeddings.weight
            out_w = getattr(output_embeddings, "weight", None)

            # Manual neutral init: set new rows to mean of existing embeddings + small noise
            base_mean = in_w[:original_vocab_size].mean(dim=0, keepdim=True)
            base_std = in_w[:original_vocab_size].std().item()

            # Apply to coordinate token rows only
            coord_token_ids_tensor = torch.tensor(coord_token_ids, device=device)

            # Initialize coordinate token rows to mean + small noise
            coord_embeddings = base_mean.expand(len(coord_token_ids_tensor), -1).clone()
            coord_embeddings += (
                0.01 * float(base_std) * torch.randn_like(coord_embeddings)
            )

            in_w[coord_token_ids_tensor] = coord_embeddings

            # Mirror into LM head if it exists
            if isinstance(out_w, torch.Tensor) and out_w.dim() == 2:
                max_id = int(coord_token_ids_tensor.max())
                if out_w.shape[0] > max_id:
                    # [vocab, hidden] layout
                    out_w[coord_token_ids_tensor] = coord_embeddings
                elif out_w.shape[1] > max_id:
                    # [hidden, vocab] layout
                    out_w[:, coord_token_ids_tensor] = coord_embeddings.T

        logger.info(
            "✅ Coordinate tokens initialized with ms-swift neutral initialization"
        )

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
            raise ValueError(
                "Coordinate tokens are disabled in configuration. Enable coordinate_tokens_enabled to convert coordinates."
            )

        tokens = []
        for coord in coordinates:
            if not isinstance(coord, int):
                raise ValueError(f"Coordinate must be int, got {type(coord)}: {coord}")
            if coord < 0 or coord > self.config.max_coord_value:
                raise ValueError(
                    f"Coordinate {coord} out of valid range [0, {self.config.max_coord_value}]"
                )
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
            raise ValueError(
                "Coordinate tokens are disabled in configuration. Enable coordinate_tokens_enabled to parse tokens."
            )

        coordinates = []
        for token in tokens:
            # Strict format: <|coord_N|>
            if not (token.startswith("<|coord_") and token.endswith("|>")):
                raise ValueError(f"Unknown coordinate token format: {token}")
            try:
                coord_str = token[8:-2]  # Extract N from <|coord_N|>
                coord = int(coord_str)
            except Exception:
                raise ValueError(f"Cannot parse coordinate token: {token}")
            if coord < 0 or coord > self.config.max_coord_value:
                raise ValueError(
                    f"Coordinate {coord} out of valid range [0, {self.config.max_coord_value}]"
                )
            coordinates.append(coord)

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
            from src_new.processing.special_tokens import GEOMETRY_TOKENS

            raise ValueError(
                f"Object contains unsupported geometry type. "
                f"Expected one of: {list(GEOMETRY_TOKENS.keys())}. "
                f"Found geometry keys: {available_keys}. "
                f"Full object: {obj}"
            )

    def extract_coordinates_from_tokens(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase
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

            # Strictly detect coordinate tokens
            if token_str.startswith("<|coord_") and token_str.endswith("|>"):
                start_idx = i
                coordinates = []

                # Collect consecutive coordinate tokens
                while i < len(tokens):
                    current_token = tokenizer.decode([tokens[i]])
                    if current_token.startswith("<|coord_") and current_token.endswith(
                        "|>"
                    ):
                        try:
                            coord_str = current_token[8:-2]
                            coord = int(coord_str)
                        except Exception:
                            raise ValueError(
                                f"Cannot parse coordinate token: {current_token}"
                            )
                        if coord < 0 or coord > self.config.max_coord_value:
                            raise ValueError(
                                f"Coordinate {coord} out of valid range [0, {self.config.max_coord_value}]"
                            )
                        coordinates.append(coord)
                        i += 1
                    else:
                        break

                end_idx = i - 1
                coordinate_sequences.append((start_idx, end_idx, coordinates))
            else:
                i += 1

        return coordinate_sequences

    def create_coordinate_mask(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase
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

        if not self.config.coordinate_tokens_enabled:
            raise ValueError(
                "Coordinate tokens are disabled in configuration. Enable coordinate_tokens_enabled to create masks."
            )

        coordinate_sequences = self.extract_coordinates_from_tokens(
            input_ids, tokenizer
        )

        for start_idx, end_idx, _ in coordinate_sequences:
            mask[start_idx : end_idx + 1] = True

        return mask

    def get_geometry_token_ids(self, tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
        """
        Get token IDs for geometry start/end tokens.

        Args:
            tokenizer: Extended tokenizer

        Returns:
            Dictionary mapping token names to IDs
        """
        token_ids = {}

        # Standard geometry tokens (must exist)
        standard_tokens = [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
        ]

        # New geometry tokens
        all_tokens = standard_tokens + self.config.new_geometry_tokens

        vocab = tokenizer.get_vocab()
        missing = [t for t in all_tokens if t not in vocab]
        if missing:
            raise ValueError(f"Geometry tokens missing from vocabulary: {missing}")

        for token in all_tokens:
            token_ids[token] = vocab[token]

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
        self, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[int, int]:
        """
        Get the token ID range for coordinate tokens.

        Delegates to centralized special token helper, returning
        (start_id, end_exclusive) strictly. Returns (0, 0) if none.
        """
        if not self.config.coordinate_tokens_enabled:
            return (0, 0)
        try:
            from src_new.processing.special_tokens import get_coord_token_range

            rng = get_coord_token_range(tokenizer)
            return (int(rng.start_id), int(rng.end_exclusive))
        except Exception:
            # Fallback: do not guess; return (0, 0) to force callers to handle absence
            return (0, 0)
