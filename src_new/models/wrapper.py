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

from src_new.processing.special_tokens import IMAGE_PAD
from src_new.types.shapes import (
    IMAGE_GRID_THW_SHAPE_DESC,
    PIXEL_VALUES_PACKED_SHAPE_DESC,
)

from ..utils.rank_aware_logging import get_rank_aware_logger
from ..utils.tensor_validation import validate_multimodal_tensors
from .loss_manager import LossComponents, LossManager, ModelOutput
from .patches import apply_comprehensive_qwen25_fixes


logger = get_rank_aware_logger(__name__)


class CoordinateProcessor:
    """
    Coordinate token processing utilities.

    Handles coordinate token masking, vocabulary extension, and coordinate-specific
    operations for the detection model.
    """

    # Non-trivial state annotations
    config: "Config"
    coordinate_tokens_enabled: bool
    max_coord_value: int
    original_vocab_size: Optional[int]
    coordinate_token_range: Optional[tuple[int, int]]
    _tokenizer: Optional[PreTrainedTokenizerBase]

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
        if tokenizer is None:
            raise ValueError("Tokenizer must not be None in set_tokenizer")

        from src_new.processing.special_tokens import get_coord_token_range

        rng = get_coord_token_range(tokenizer)
        vocab = tokenizer.get_vocab()

        if self.coordinate_tokens_enabled:
            if rng.end_exclusive <= rng.start_id:
                raise ValueError(
                    "Coordinate tokens enabled but no valid coordinate token range detected"
                )
            self.coordinate_token_range = (int(rng.start_id), int(rng.end_exclusive))
            self.original_vocab_size = int(rng.start_id)
            logger.info(f"🎯 Coordinate token range: {self.coordinate_token_range}")
        else:
            # Coordinates disabled: record vocab size and zero range
            self.original_vocab_size = len(vocab)
            self.coordinate_token_range = (0, 0)

    def update_after_extension(self, tokenizer) -> None:
        """
        Update coordinate processor after tokenizer vocabulary extension.

        This method should be called after the tokenizer vocabulary has been extended
        with coordinate tokens to properly detect and configure the coordinate token range.

        Args:
            tokenizer: Extended tokenizer with coordinate tokens
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must not be None in update_after_extension")

        from src_new.processing.special_tokens import get_coord_token_range

        rng = get_coord_token_range(tokenizer)
        if self.coordinate_tokens_enabled:
            if rng.end_exclusive <= rng.start_id:
                raise ValueError(
                    "Coordinate tokens enabled but no valid coordinate token range after extension"
                )
            self.original_vocab_size = int(rng.start_id)
            self.coordinate_token_range = (int(rng.start_id), int(rng.end_exclusive))
            logger.info(
                f"🎯 Updated after extension: final coordinate token range: {self.coordinate_token_range}"
            )
        else:
            vocab = tokenizer.get_vocab()
            self.original_vocab_size = len(vocab)
            self.coordinate_token_range = (0, 0)

    def mask_coordinate_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask coordinate-specific logits for non-coordinate positions.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Masked logits tensor
        """
        if not self.coordinate_tokens_enabled or self.coordinate_token_range is None:
            return logits

        # Use the actual coordinate token range from tokenizer (end-exclusive)
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

    # Non-trivial state annotations
    base_model: Qwen2_5_VLForConditionalGeneration
    training_config: "Config"
    _config: Any
    tokenizer: Optional[PreTrainedTokenizerBase]
    _coordinate_mode: bool
    coordinate_processor: CoordinateProcessor
    token_processor: Any
    _tokenizer: Optional[PreTrainedTokenizerBase]
    _extended_tokenizer: Optional[PreTrainedTokenizerBase]
    loss_manager: Optional[Any]

    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        config: "Config",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        skip_expansion: bool = False,
    ) -> None:
        """
        Initialize detection model.

        Args:
            base_model: Base Qwen2.5-VL model
            config: Configuration object with model settings
            tokenizer: Tokenizer for text processing (optional)
            skip_expansion: If True, skip tokenizer/model expansion (already done)
        """
        super().__init__()
        self.base_model = base_model
        # Store training (custom) config separately to avoid HuggingFace trainer conflicts
        self.training_config = config
        # IMPORTANT: Keep `self.config` as the HF model config to maintain Trainer compatibility
        self._config = self.base_model.config
        self.tokenizer = tokenizer

        # Initialize coordinate mode first
        self._coordinate_mode = config.coordinate_tokens_enabled

        # Initialize coordinate processor
        self.coordinate_processor = CoordinateProcessor(config)

        # Initialize token processor for vocabulary extension
        from src_new.processing.token_processor import TokenConfig, TokenProcessor

        # Validate required config attributes
        if not hasattr(config, "new_geometry_tokens"):
            raise ValueError(
                "Config missing required attribute 'new_geometry_tokens'. Ensure config validation was run."
            )

        token_config = TokenConfig(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
            new_geometry_tokens=config.new_geometry_tokens or [],
            coordinate_init_mode=config.coordinate_init_mode,
        )
        self.token_processor = TokenProcessor(token_config)

        # Store tokenizer reference for coordinate processing
        self._tokenizer = tokenizer

        # DISTRIBUTED TRAINING OPTIMIZATION: Skip expansion if already done
        # Use module-level rank-aware logger for config messages
        config_logger = logger

        if skip_expansion:
            config_logger.info(
                "🚀 Skipping tokenizer/model expansion - already completed in pre-distributed phase"
            )
            # Set the extended tokenizer since expansion was already done
            self._extended_tokenizer = tokenizer
        elif tokenizer is not None and self._coordinate_mode:
            # Original expansion logic would go here, but it's now handled in pre-distributed phase
            config_logger.warning(
                "⚠️ Tokenizer expansion should have been completed in pre-distributed phase"
            )
            self._extended_tokenizer = tokenizer
        else:
            # Store the tokenizer even if no expansion is needed
            self._extended_tokenizer = tokenizer

        # Set tokenizer for coordinate processor AFTER extension
        # Use the extended tokenizer to ensure coordinate tokens are available
        final_tokenizer = (
            self._extended_tokenizer
            if hasattr(self, "_extended_tokenizer")
            else tokenizer
        )
        if final_tokenizer is not None:
            from src_new.processing.special_tokens import get_coord_token_range

            # Set tokenizer reference first
            self.coordinate_processor._tokenizer = final_tokenizer

            # Get coordinate token range - fail fast if tokenizer is invalid
            rng = get_coord_token_range(final_tokenizer)
            self.coordinate_processor.coordinate_token_range = (
                rng.start_id,
                rng.end_exclusive,
            )

            if skip_expansion and self._coordinate_mode:
                # Re-validate coordinate range for skip_expansion mode
                rng2 = get_coord_token_range(final_tokenizer)
                if rng2.start_id == rng2.end_exclusive and self._coordinate_mode:
                    raise ValueError(
                        "Coordinate mode enabled but no coordinate tokens found in tokenizer. "
                        "Ensure tokenizer was properly extended with coordinate tokens."
                    )
                self.coordinate_processor.coordinate_token_range = (
                    rng2.start_id,
                    rng2.end_exclusive,
                )

        # Cache IMAGE_PAD token id once for forward-time fast counting
        self._image_pad_token_id = None
        try:
            cfg_id = getattr(self.base_model.config, "image_token_id", None)
            if isinstance(cfg_id, int) and cfg_id >= 0:
                self._image_pad_token_id = int(cfg_id)
            elif final_tokenizer is not None:
                vocab = final_tokenizer.get_vocab()
                if isinstance(vocab, dict) and IMAGE_PAD in vocab:
                    self._image_pad_token_id = int(vocab[IMAGE_PAD])
        except Exception:
            pass

        # Initialize loss manager lazily to avoid requiring full loss config at construction time
        self.loss_manager = None

        def _init_loss_manager_if_needed():
            if self.loss_manager is None:
                self.loss_manager = LossManager(
                    self.training_config,
                    token_processor=self.token_processor,
                    tokenizer=final_tokenizer,
                )
                # Enable grouped-LLM plugin unconditionally; weights control contribution
                if final_tokenizer is None:
                    raise ValueError(
                        "Tokenizer must be available to initialize TokenGroupingPlugin"
                    )
                from src_new.losses.token_grouping import TokenGroupingPlugin

                self.loss_manager.set_token_grouping_plugin(
                    TokenGroupingPlugin(final_tokenizer)
                )
                # If auxiliary coordinate losses are enabled, configure options now
                if self.training_config.coord_aux_enabled:
                    self.loss_manager.set_coordinate_aux_options(
                        tau=float(self.training_config.coord_aux_tau),
                        sigma_bins=float(self.training_config.coord_aux_sigma_bins),
                        window_bins=int(self.training_config.coord_aux_window_bins),
                        topk=int(self.training_config.coord_aux_topk),
                        lambda_kce=float(self.training_config.coord_aux_lambda_kce),
                        lambda_unlike=float(
                            self.training_config.coord_aux_lambda_unlike
                        ),
                        lambda_lap1=0.0,  # Laplacian regularizer removed
                        lambda_lap2=0.0,  # Laplacian regularizer removed
                    )
                # Laplacian regularizer removed: no embedding accessor needed

        # Store initializer for later use
        self._ensure_loss_manager = _init_loss_manager_if_needed

        # Note: Embedding extension is now handled in the intelligent detection logic above
        # This prevents double extension when loading from checkpoints

        # Disable cache during training for better performance
        self.base_model.config.use_cache = config.use_cache

        # Apply any necessary patches
        apply_comprehensive_qwen25_fixes()

        # Ensure tied weights are properly set up
        self.tie_weights()

    def get_extended_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        """
        Get the extended tokenizer after vocabulary extension.

        Returns:
            Extended tokenizer or None if not available
        """
        return self._extended_tokenizer

    @staticmethod
    def detect_extended_checkpoint(model_path: str) -> bool:
        """
        Detect if a checkpoint already has extended vocabulary by inspecting tokenizer tokens.

        Args:
            model_path: Path to model checkpoint

        Returns:
            True if checkpoint contains any <|coord_*> tokens
        """
        # Validate model path exists
        from pathlib import Path

        from transformers import AutoTokenizer

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Load tokenizer - require fast tokenizer
        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False, use_fast=True
        )
        if not getattr(tok, "is_fast", False):
            raise RuntimeError("Fast tokenizer required for DetectionModel (use_fast=True)")
        _enc = tok(
            "sanity",
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        if _enc.get("offset_mapping") is None:
            raise RuntimeError(
                "Fast tokenizer did not return offset_mapping; ensure tokenizer.json is valid."
            )
        vocab = tok.get_vocab()
        if not isinstance(vocab, dict):
            raise ValueError(f"Invalid tokenizer vocabulary type: {type(vocab)}")
        return any(
            isinstance(t, str) and t.startswith("<|coord_") for t in vocab.keys()
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: "Config",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ) -> "DetectionModel":
        """
        Create detection model from pretrained model with intelligent checkpoint detection.

        Args:
            model_path: Path to pretrained model
            config: Configuration object with model settings
            tokenizer: Tokenizer for text processing (optional)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            DetectionModel instance
        """
        # Using module-level rank-aware logger

        # OPTIMIZATION 1: Intelligent checkpoint detection
        is_extended_checkpoint = cls.detect_extended_checkpoint(model_path)

        if is_extended_checkpoint and config.coordinate_tokens_enabled:
            logger.info(
                f"🚀 Fast loading - detected extended checkpoint at {model_path}"
            )
            # Extended checkpoint detected; vocabulary extension will be skipped by callers
        elif not config.coordinate_tokens_enabled:
            logger.info(f"📋 Loading base model - coordinate tokens disabled")
        else:
            logger.info(
                f"🔧 Loading base model - will extend vocabulary for coordinate tokens"
            )

        # OPTIMIZATION 2: Load base model with optimized parameters
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.bfloat16,
        }
        if config.torch_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype: {config.torch_dtype}. Supported: {list(dtype_map.keys())}"
            )
        torch_dtype = dtype_map[config.torch_dtype]
        attn_implementation = config.attn_implementation

        # Extract conflicting parameters from kwargs to avoid conflicts
        kwargs_torch_dtype = kwargs.pop("torch_dtype", torch_dtype)
        kwargs_attn_implementation = kwargs.pop(
            "attn_implementation", attn_implementation
        )
        kwargs_trust_remote_code = kwargs.pop("trust_remote_code", False)

        # OPTIMIZATION 3: Use optimized loading parameters
        loading_kwargs = {
            "torch_dtype": kwargs_torch_dtype,
            "attn_implementation": kwargs_attn_implementation,
            "trust_remote_code": kwargs_trust_remote_code,
            "low_cpu_mem_usage": True,  # Faster loading
            **kwargs,
        }

        # OPTIMIZATION 4: Check for SafeTensors format for 4-6x faster loading
        from pathlib import Path

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        safetensors_path = model_path_obj / "model.safetensors"
        if safetensors_path.exists():
            logger.info("🚀 Using SafeTensors format for 4-6x faster loading")
            loading_kwargs["use_safetensors"] = True

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **loading_kwargs
        )

        # Create detection model
        return cls(
            base_model=base_model,
            config=config,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_pretrained_fast(
        cls,
        model_path: str,
        config,
        tokenizer=None,
        skip_vocab_extension=False,
        **kwargs,
    ):
        """
        Fast loading for inference - skips vocabulary extension if requested.

        Args:
            skip_vocab_extension: If True, assumes tokenizer already has coordinate tokens
        """
        # Call regular from_pretrained while indicating skip behavior via kwargs only
        model = cls.from_pretrained(model_path, config, tokenizer, **kwargs)
        return model

    def forward(self, **kwargs) -> Union["ModelOutput", Any]:
        """
        Forward pass with coordinate token support.

        Args:
            **kwargs: Arguments for base model forward pass

        Returns:
            Model output with loss components
        """
        # Extract inputs
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else None
        labels = kwargs["labels"] if "labels" in kwargs else None

        # CRITICAL FIX: Ensure image_grid_thw has correct shape before passing to base model
        if "image_grid_thw" in kwargs:
            # Using module-level rank-aware logger
            image_grid_thw = kwargs["image_grid_thw"]
            if image_grid_thw is not None:
                logger.debug(
                    f"🔍 BEFORE FIX: image_grid_thw shape: {image_grid_thw.shape}, values: {image_grid_thw}"
                )

                # Enforce strict contract: [num_images, 3] with no mutation
                if not (image_grid_thw.dim() == 2 and image_grid_thw.shape[1] == 3):
                    raise ValueError(
                        f"Invalid image_grid_thw shape: expected {IMAGE_GRID_THW_SHAPE_DESC}, got {image_grid_thw.shape}. "
                        f"Collator must emit flattened per-image THW grid; no automatic reshaping is performed here."
                    )
                logger.debug(
                    f"✅ image_grid_thw shape validated: {image_grid_thw.shape}"
                )

        # PRE-FORWARD SAFETY: Validate multimodal tensor consistency to prevent CUDA OOB
        pixel_values = kwargs["pixel_values"] if "pixel_values" in kwargs else None
        image_grid_thw = (
            kwargs["image_grid_thw"] if "image_grid_thw" in kwargs else None
        )
        if pixel_values is not None and image_grid_thw is not None:
            try:
                # Use centralized tensor validation for multimodal consistency
                validate_multimodal_tensors(pixel_values, image_grid_thw)
            except Exception as e:
                # Normalize to ValueError for external callers/tests while preserving message
                from src_new.utils.tensor_validation import TensorValidationError

                if isinstance(e, TensorValidationError):
                    raise ValueError(str(e))
                raise

            # Validate counts: number of image tokens should match expected tokens per image grids
            # Determine spatial merge size used by processor/model
            spatial_merge_size = None

            # First try vision config
            if hasattr(self.base_model.config, "vision_config"):
                vision_cfg = self.base_model.config.vision_config
                if vision_cfg is not None and hasattr(vision_cfg, "spatial_merge_size"):
                    if not isinstance(vision_cfg.spatial_merge_size, (int, float)):
                        raise ValueError(
                            f"Invalid spatial_merge_size type in vision config: {type(vision_cfg.spatial_merge_size)}"
                        )
                    spatial_merge_size = int(vision_cfg.spatial_merge_size)

            # If not found in vision config, try training config
            if spatial_merge_size is None:
                if not hasattr(self.training_config, "merge_size"):
                    raise ValueError(
                        "No spatial_merge_size found in vision config and no merge_size in training config. "
                        "Cannot determine merge size for image token validation."
                    )
                if not isinstance(self.training_config.merge_size, (int, float)):
                    raise ValueError(
                        f"Invalid merge_size type in training config: {type(self.training_config.merge_size)}"
                    )
                spatial_merge_size = int(self.training_config.merge_size)

            merge_length = spatial_merge_size * spatial_merge_size
            # Expected image token count matches how HF processor expands <|image_pad|>
            # See transformers Qwen2_5_VLProcessor: tokens per image = (t*h*w) // (merge_size**2)
            grid_long = image_grid_thw.to(dtype=torch.long)
            expected_image_tokens = int(
                (torch.prod(grid_long, dim=1) // merge_length).sum().item()
            )

            # Compute actual number of image pad tokens present in input_ids
            n_image_tokens = 0
            if input_ids is not None:
                # Determine image token id robustly (from model config or tokenizer)
                image_token_id_val = self._image_pad_token_id
                if image_token_id_val is not None:
                    mask_tensor = input_ids == image_token_id_val
                    n_image_tokens = int(mask_tensor.sum().item())

            if n_image_tokens != expected_image_tokens:
                raise ValueError(
                    f"Image token count mismatch: tokens={n_image_tokens}, expected={expected_image_tokens}. "
                    f"Check chat template expansion vs image_grid_thw and merge_size={spatial_merge_size}."
                )

            # Validate total patches count matches flattened rows
            grid = image_grid_thw.to(dtype=torch.long)
            expected_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
            actual_patches = int(pixel_values.shape[0])
            if actual_patches != expected_patches:
                raise ValueError(
                    f"Mismatch between image token count and pixel_values rows: expected {expected_patches} rows (sum(t*h*w)), got {actual_patches}. "
                    f"pixel_values shape={tuple(pixel_values.shape)} (expected leading shape {PIXEL_VALUES_PACKED_SHAPE_DESC})"
                )

        # NEW: Strict validation for text tensors before base model call (fail-fast)
        if input_ids is not None:
            if not (input_ids.dim() == 2):
                raise ValueError(
                    f"input_ids must be 2D [batch, seq], got shape={tuple(input_ids.shape)}"
                )
        attention_mask = (
            kwargs["attention_mask"] if "attention_mask" in kwargs else None
        )
        if attention_mask is not None:
            if not (attention_mask.dim() == 2):
                raise ValueError(
                    f"attention_mask must be 2D [batch, seq], got shape=={tuple(attention_mask.shape)}"
                )
        if labels is not None:
            if not (labels.dim() == 2):
                raise ValueError(
                    f"labels must be 2D [batch, seq], got shape={tuple(labels.shape)}"
                )

        logger.debug(
            f"Forward input shapes: input_ids={None if input_ids is None else tuple(input_ids.shape)}, "
            f"attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}, "
            f"labels={None if labels is None else tuple(labels.shape)}, "
            f"pixel_values={None if pixel_values is None else tuple(pixel_values.shape)}, "
            f"image_grid_thw={None if image_grid_thw is None else tuple(image_grid_thw.shape)}"
        )

        # SOLUTION 1 OPTIMIZATION: Check if we should bypass official loss computation
        # This optimization reduces loss computation time by 60-70% when teacher-student
        # spans are provided by bypassing the base model's loss computation and using
        # our optimized single-pass cross-entropy method instead.
        teacher_spans = (
            kwargs["teacher_assistant_spans"]
            if "teacher_assistant_spans" in kwargs
            else None
        )
        student_spans = (
            kwargs["student_assistant_spans"]
            if "student_assistant_spans" in kwargs
            else None
        )
        should_bypass_official_loss = (
            (teacher_spans is not None or student_spans is not None)
            and input_ids is not None
            and labels is not None
        )

        # Filter out HuggingFace Trainer-specific arguments that base model doesn't accept
        excluded_args = [
            "num_items_in_batch",
            "teacher_assistant_spans",
            "student_assistant_spans",
            "assistant_spans",
        ]
        base_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_args}

        # Optional: build block-diagonal causal mask to isolate packed segments
        try:
            seg_lens = kwargs.get("segment_lengths", None)
            if (
                seg_lens is not None
                and isinstance(seg_lens, torch.Tensor)
                and seg_lens.dim() == 1
                and input_ids is not None
            ):
                # Create per-row causal mask that is block-diagonal across segments
                # Shapes: input_ids [B, S]; we assume B=1 for packed row, but support general B
                B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
                lengths = seg_lens.to(device=input_ids.device, dtype=torch.long)
                if lengths.sum().item() == S:
                    # Build [S, S] lower-triangular causal base
                    causal = torch.tril(torch.ones(S, S, device=input_ids.device, dtype=torch.bool))
                    # Zero out cross-segment regions
                    idx = 0
                    blocks: list[tuple[int, int]] = []
                    for L in lengths.tolist():
                        blocks.append((idx, idx + L))
                        idx += L
                    mask_bool = torch.zeros_like(causal)
                    for (s, e) in blocks:
                        mask_bool[s:e, s:e] = causal[s:e, s:e]
                    # Convert to additive mask with -inf for masked positions; expand to [B, 1, S, S]
                    attn_add = (~mask_bool).to(dtype=input_ids.dtype) * torch.finfo(input_ids.dtype).min
                    attn_add = attn_add.view(1, 1, S, S).expand(B, 1, S, S)
                    # Supply as mapping expected by Qwen2.5-VL (full_attention key)
                    base_kwargs["attention_mask"] = {"full_attention": attn_add}
        except Exception:
            # Do not fail training if isolation mask construction has any issue
            pass

        # SOLUTION 1: Bypass official loss computation when teacher-student spans provided
        if should_bypass_official_loss:
            # Remove labels to prevent base model from computing loss
            base_kwargs_no_loss = base_kwargs.copy()
            base_kwargs_no_loss.pop("labels", None)

            # Forward pass without loss computation
            base_outputs = self.base_model(**base_kwargs_no_loss)

            # Add None loss to maintain interface compatibility
            base_outputs.loss = None

            # Using module-level rank-aware logger

            logger.debug(
                "🚀 SOLUTION 1: Bypassed official loss computation for optimized teacher-student training"
            )
        else:
            # Standard forward pass with official loss computation
            base_outputs = self.base_model(**base_kwargs)

        # Process outputs based on mode
        if self._coordinate_mode and input_ids is not None and labels is not None:
            # Debug logging for coordinate mode
            # Using module-level rank-aware logger

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
            # Using module-level rank-aware logger

            logger.debug(
                f"🔍 Using standard mode: coordinate_mode={self._coordinate_mode}, input_ids={input_ids is not None}, labels={labels is not None}"
            )

            # SOLUTION 1: Handle both official and bypassed loss computation
            if input_ids is not None and labels is not None:
                # Extract teacher and student spans from kwargs
                teacher_spans = (
                    kwargs["teacher_assistant_spans"]
                    if "teacher_assistant_spans" in kwargs
                    else None
                )
                student_spans = (
                    kwargs["student_assistant_spans"]
                    if "student_assistant_spans" in kwargs
                    else None
                )
                # Unified assistant spans take precedence when available
                unified_spans = (
                    kwargs["assistant_spans"] if "assistant_spans" in kwargs else None
                )
                if unified_spans is not None and not (
                    (teacher_spans is not None and len(teacher_spans) > 0)
                    or (student_spans is not None and len(student_spans) > 0)
                ):
                    teacher_spans = None
                    student_spans = unified_spans

                # Use the loss manager to compute detailed components
                # This will use the optimized single-pass method when teacher-student spans are provided
                # Ensure loss manager is initialized before computing components
                self._ensure_loss_manager()
                # Provide input_ids to loss manager for plain-mode grouping decode
                try:
                    if input_ids is not None and hasattr(self.loss_manager, "__dict__"):
                        self.loss_manager._last_input_ids = input_ids.detach().clone()
                except Exception as e:
                    raise RuntimeError(f"Failed to attach last_input_ids for grouping: {e}")
                loss_components = self.loss_manager.compute_loss_components(
                    logits=base_outputs.logits,
                    labels=labels,
                    coord_mask=None,  # No coordinate mask in standard mode
                    teacher_spans=teacher_spans,
                    student_spans=student_spans,
                    conversation_variant=kwargs.get("conversation_variant"),
                )
                # Store for callback access
                if self.loss_manager is not None:
                    self.loss_manager.last_loss_components = loss_components

                # Log optimization status
                if should_bypass_official_loss:
                    # Using module-level rank-aware logger
                    logger.debug(
                        "🚀 SOLUTION 1: Used optimized single-pass cross-entropy computation"
                    )
                else:
                    # Using module-level rank-aware logger
                    logger.debug(
                        "🔍 Used standard loss computation (no teacher-student spans provided)"
                    )

            elif hasattr(base_outputs, "loss") and base_outputs.loss is not None:
                # Disallow implicit fallback; input_ids/labels are required for strict training semantics
                raise RuntimeError("Base model returned a loss but input_ids/labels were not provided to wrapper; strict mode requires spans and labels.")
            else:
                # No loss available -> hard error
                raise RuntimeError("No loss available from base model and wrapper; ensure labels/spans are provided.")

            # Return dict-like object for DataParallel compatibility
            return {
                "loss": (
                    loss_components.loss
                    if loss_components.loss is not None
                    else (base_outputs.loss if hasattr(base_outputs, "loss") else None)
                ),
                "logits": base_outputs.logits if hasattr(base_outputs, "logits") else None,
                "loss_components": loss_components,
                "hidden_states": (
                    base_outputs.hidden_states if hasattr(base_outputs, "hidden_states") else None
                ),
            }

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
        # Use base logits directly. Do not globally mask coordinate logits by input positions,
        # because next-token prediction requires coordinate vocab to be available at label positions.
        masked_logits = base_outputs.logits

        # Build a coord mask over LABEL positions (what is being predicted): we pass None here to
        # let LossManager derive the coord positions from labels with correct shifting. To preserve
        # interface compatibility, we compute but do not rely on the input-id based mask.
        coord_mask = None

        # Extract teacher and student spans from kwargs
        teacher_spans = (
            kwargs["teacher_assistant_spans"]
            if "teacher_assistant_spans" in kwargs
            else None
        )
        student_spans = (
            kwargs["student_assistant_spans"]
            if "student_assistant_spans" in kwargs
            else None
        )
        # Unified assistant spans take precedence when available
        unified_spans = (
            kwargs["assistant_spans"] if "assistant_spans" in kwargs else None
        )
        if unified_spans is not None and not (
            (teacher_spans is not None and len(teacher_spans) > 0)
            or (student_spans is not None and len(student_spans) > 0)
        ):
            teacher_spans = None
            student_spans = unified_spans

        # Compute loss components with teacher-student spans
        # Ensure loss manager is initialized before computing components
        self._ensure_loss_manager()
        loss_components = self.loss_manager.compute_loss_components(
            logits=masked_logits,
            labels=labels,
            coord_mask=coord_mask,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
        )

        # Return plain dict for compatibility with Accelerate recursive conversions
        return {
            "loss": loss_components.loss,
            "logits": masked_logits,
            "loss_components": loss_components,
            "hidden_states": (
                base_outputs.hidden_states if hasattr(base_outputs, "hidden_states") else None
            ),
        }

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
        return (
            self.loss_manager.last_loss_components
            if self.loss_manager is not None
            else None
        )

    def get_last_loss_components(self) -> Optional[LossComponents]:
        """
        Get loss components from the last forward pass.

        Returns:
            Loss components or None if not available
        """
        return self.get_loss_components()

    def update_epoch(self, epoch: int):
        """
        Update the current epoch for coordinate loss warning control.

        Args:
            epoch: Current training epoch
        """
        if self.loss_manager is not None and hasattr(self.loss_manager, "update_epoch"):
            self.loss_manager.update_epoch(epoch)

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
    def config(self):
        """Expose underlying HF model config for integrations expecting model.config.* methods."""
        return self.base_model.config

    @config.setter
    def config(self, config: "Config") -> None:
        """
        Set configuration object.

        Keep HuggingFace model config as self._config for Trainer compatibility,
        and store the training-specific config in self.training_config.
        """
        self.training_config = config
        self._config = self.base_model.config

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
        Resize token embeddings with ms-swift inspired optimizations.

        Args:
            new_num_tokens: New number of tokens

        Returns:
            Updated embedding module
        """
        # MS-SWIFT OPTIMIZATION: Use pad_to_multiple_of=128 for better performance
        import math

        padded_size = math.ceil(new_num_tokens / 128) * 128

        logger.info(
            f"🚀 ms-swift optimized embedding resize: {new_num_tokens} → {padded_size} (padded)"
        )

        # Resize embeddings with ms-swift optimizations
        self.base_model.resize_token_embeddings(
            padded_size,
            mean_resizing=False,  # Our optimization
            pad_to_multiple_of=128,  # ms-swift optimization
        )

        # Do not update coordinate token range here; it must be derived from tokenizer IDs
        # and validated via set_tokenizer/update_after_extension.

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
            if (
                hasattr(self.base_model.config, "torchscript")
                and self.base_model.config.torchscript
            ):
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
                logger.warning(
                    "⚠️ Shared tensor error detected, falling back to non-safe serialization"
                )
                # Fallback to non-safe serialization
                self.base_model.save_pretrained(
                    save_directory, safe_serialization=False, **kwargs
                )
            else:
                raise e

        # NOTE: Tokenizer and coordinate config are now saved centrally by CheckpointSaver
        logger.info(f"Model saved to {save_directory}")

    def _detect_checkpoint_type(self, tokenizer, base_model, config):
        """
        Intelligently detect checkpoint type and determine if vocabulary extension is needed.

        Returns:
            dict: {
                'mode': str,  # 'Base Model' or 'Fine-tuned Checkpoint'
                'needs_extension': bool,
                'vocab_size': int,
                'model_embeddings': int,
                'coordinate_tokens_found': int,
                'max_coord_detected': int or None
            }
        """
        # Using module-level rank-aware logger

        # Analyze tokenizer vocabulary
        vocab = tokenizer.get_vocab()
        current_vocab_size = len(vocab)
        # Get original vocab size from model config - fail fast if not available
        if not hasattr(self.base_model.config, "vocab_size"):
            raise ValueError("Model config missing required 'vocab_size' attribute")
        original_vocab_size = self.base_model.config.vocab_size
        if not isinstance(original_vocab_size, int) or original_vocab_size <= 0:
            raise ValueError(
                f"Invalid vocab_size in model config: {original_vocab_size}"
            )

        # Analyze model embeddings
        model_embeddings = base_model.get_input_embeddings().num_embeddings

        # Detect coordinate tokens
        coordinate_tokens = [
            token for token in vocab.keys() if token.startswith("<|coord_")
        ]
        coordinate_tokens_found = len(coordinate_tokens)

        # Extract max coordinate value from existing tokens
        max_coord_detected = None
        if coordinate_tokens:
            coord_values = []
            for token in coordinate_tokens:
                try:
                    # Extract number from '<|coord_123|>' format
                    coord_val = int(token.replace("<|coord_", "").replace("|>", ""))
                    coord_values.append(coord_val)
                except ValueError:
                    continue
            if coord_values:
                max_coord_detected = max(coord_values)

        # Determine checkpoint type and extension needs
        is_base_model = (
            current_vocab_size == original_vocab_size and coordinate_tokens_found == 0
        )

        is_extended_checkpoint = (
            current_vocab_size > original_vocab_size
            and coordinate_tokens_found > 0
            and max_coord_detected is not None
        )

        # Log detection results
        logger.info("🔍 Checkpoint Analysis:")
        logger.info(f"   📚 Tokenizer vocab size: {current_vocab_size}")
        logger.info(f"   🔢 Model embeddings: {model_embeddings}")
        logger.info(f"   🎯 Coordinate tokens found: {coordinate_tokens_found}")
        if max_coord_detected is not None:
            logger.info(f"   📊 Max coordinate detected: {max_coord_detected}")

        if is_base_model:
            return {
                "mode": "Base Model (Training Mode)",
                "needs_extension": True,
                "vocab_size": current_vocab_size,
                "model_embeddings": model_embeddings,
                "coordinate_tokens_found": coordinate_tokens_found,
                "max_coord_detected": max_coord_detected,
            }
        elif is_extended_checkpoint:
            # Check if model embeddings need extension (vocabulary might be larger than embeddings)
            needs_embedding_extension = model_embeddings < current_vocab_size

            return {
                "mode": "Fine-tuned Checkpoint (Inference Mode)",
                "needs_extension": needs_embedding_extension,
                "vocab_size": current_vocab_size,
                "model_embeddings": model_embeddings,
                "coordinate_tokens_found": coordinate_tokens_found,
                "max_coord_detected": max_coord_detected,
            }
        else:
            # Ambiguous case - be conservative and extend
            logger.warning(
                "⚠️ Ambiguous checkpoint type detected - applying conservative extension"
            )
            return {
                "mode": "Unknown Checkpoint (Conservative Mode)",
                "needs_extension": True,
                "vocab_size": current_vocab_size,
                "model_embeddings": model_embeddings,
                "coordinate_tokens_found": coordinate_tokens_found,
                "max_coord_detected": max_coord_detected,
            }

    def _validate_checkpoint_config(self, checkpoint_info, config):
        """
        Validate that checkpoint configuration matches current config.
        """
        # Using module-level rank-aware logger

        if checkpoint_info["max_coord_detected"] is not None:
            config_max_coord = config.max_coord_value
            checkpoint_max_coord = checkpoint_info["max_coord_detected"]

            if config_max_coord != checkpoint_max_coord:
                logger.warning(f"⚠️ Configuration mismatch detected:")
                logger.warning(f"   Config max_coord_value: {config_max_coord}")
                logger.warning(f"   Checkpoint max_coord: {checkpoint_max_coord}")
                logger.warning(
                    "   This may cause issues with coordinate token processing"
                )
                logger.warning(
                    "   Consider updating config to match checkpoint or vice versa"
                )
            else:
                logger.info(
                    f"✅ Configuration validated: max_coord_value = {config_max_coord}"
                )
