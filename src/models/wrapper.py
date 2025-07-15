"""
Qwen2.5-VL Model Wrapper with Detection Capabilities

This module provides a wrapper around the official Qwen2.5-VL model
that adds object detection capabilities while preserving all original functionality.
"""

import json
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

from src.config import config
from src.models.patches import apply_comprehensive_qwen25_fixes


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


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16,  # Default to bfloat16 for auto
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


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

        # Store tokenizer for detection head initialization
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        # Store coordinate configuration
        self.coordinate_config = coordinate_config or CoordinateConfig()

        # Coordinate token tracking
        self.coordinate_tokens_enabled = self.coordinate_config.enable_coordinate_tokens
        self.original_vocab_size = None
        self.extended_vocab_size = None
        self.extended_embeddings = None
        self.extended_lm_head = None

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

        # CRITICAL: Move base model to GPU if available
        if torch.cuda.is_available():
            self.base_model = self.base_model.to("cuda:0")

        # CRITICAL: Apply fixes for mRoPE and visual processing
        apply_comprehensive_qwen25_fixes()

        # Store original vocab size before any modifications
        self.original_vocab_size = self.base_model.config.vocab_size

        # Initialize coordinate token support if enabled
        if self.coordinate_tokens_enabled:
            self._setup_coordinate_tokens()
        else:
            # No coordinate tokens - use original vocab
            self.extended_vocab_size = self.original_vocab_size

        # Legacy detection head removed - using coordinate token approach
        self.detection_head = None

        # Legacy detection head setup (disabled)
        # NOTE: Detection head functionality replaced by coordinate tokens
        # if self.detection_head is not None:
        #     self.detection_head.set_token_embedding(self.base_model.get_input_embeddings())
        #     device = next(self.base_model.parameters()).device
        #     self.detection_head = self.detection_head.to(device=device)

        # Store our custom config for internal use, but expose base model config for DeepSpeed
        self._custom_config = config

        # Move coordinate token components to same device as base model
        device = next(self.base_model.parameters()).device

        # Move extended components to device if they exist
        if self.extended_embeddings is not None:
            self.extended_embeddings = self.extended_embeddings.to(device=device)
        if self.extended_lm_head is not None:
            self.extended_lm_head = self.extended_lm_head.to(device=device)

    def _init_detection_head(self):
        """LEGACY: Initialize the detection head - REPLACED BY COORDINATE TOKENS."""
        # NOTE: This method is preserved for reference but no longer used.
        # Detection functionality moved to coordinate token soft expectation.
        # Original implementation moved to legacy/detection/
        raise NotImplementedError(
            "Detection head replaced by coordinate token soft expectation. "
            "Enable coordinate tokens via CoordinateConfig instead."
        )
        
        # Legacy implementation (commented out):
        # from src.config import config
        # from legacy.detection.detection_head import DetectionHead
        # ... (original implementation moved to legacy/)

    def forward(
        self, **inputs: Any
    ) -> Union[Tuple[Any, ...], Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Forward pass that preserves all functionality and supports coordinate tokens.
        """
        # Store original ground truth objects for detection loss (don't pop them)
        # The trainer will handle detection loss computation

        # Remove ground truth objects from model inputs (but keep them in original inputs)
        model_inputs = inputs.copy()
        model_inputs.pop("ground_truth_objects", None)
        model_inputs.pop("image_counts_per_sample", None)

        # Handle coordinate token processing if enabled
        if self.coordinate_tokens_enabled and "input_ids" in model_inputs:
            return self._forward_with_coordinate_tokens(model_inputs, inputs)
        else:
            # Standard Qwen2.5-VL forward pass with all parameters preserved
            outputs = self.base_model(**model_inputs)
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
        print("🚀 Setting up coordinate token support...")

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

        print(
            f"✅ Extended vocab from {self.original_vocab_size} to {self.extended_vocab_size}"
        )
        print(
            f"✅ Added {num_new_tokens} coordinate tokens (reusing official box tokens)"
        )
        print("✅ All pretrained weights preserved")

    def _extend_tokenizer(self):
        """Add coordinate tokens to tokenizer (reuse existing box tokens)."""
        # Only add coordinate tokens - box tokens already exist
        coordinate_tokens = [
            f"<coord_{i}>" for i in range(self.coordinate_config.max_coord_value)
        ]

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": coordinate_tokens}
        )

        print(f"✅ Added {num_added} coordinate tokens to tokenizer")
        print(
            f"✅ Reusing official box tokens: <|box_start|> ({self.box_start_id}), <|box_end|> ({self.box_end_id})"
        )

    def _create_extended_embeddings(self):
        """Create extended embeddings while preserving pretrained weights."""
        original_embeddings = self.base_model.get_input_embeddings()
        hidden_size = original_embeddings.weight.shape[1]

        # Create new embedding layer
        self.extended_embeddings = nn.Embedding(
            self.extended_vocab_size,
            hidden_size,
            device=original_embeddings.weight.device,
            dtype=original_embeddings.weight.dtype,
        )

        # Copy pretrained weights (PRESERVE)
        with torch.no_grad():
            self.extended_embeddings.weight[: self.original_vocab_size].copy_(
                original_embeddings.weight
            )

            # Initialize new coordinate tokens
            nn.init.normal_(
                self.extended_embeddings.weight[self.original_vocab_size :],
                std=self.coordinate_config.coord_token_init_std,
            )

        # Freeze original embeddings by detaching them first
        with torch.no_grad():
            self.extended_embeddings.weight[: self.original_vocab_size].requires_grad_(False)

    def _create_extended_lm_head(self):
        """Create extended LM head while preserving pretrained weights."""
        original_lm_head = self.base_model.get_output_embeddings()
        hidden_size = original_lm_head.weight.shape[1]

        # Create new LM head
        self.extended_lm_head = nn.Linear(
            hidden_size,
            self.extended_vocab_size,
            bias=False,
            device=original_lm_head.weight.device,
            dtype=original_lm_head.weight.dtype,
        )

        # Copy pretrained weights (PRESERVE)
        with torch.no_grad():
            self.extended_lm_head.weight[: self.original_vocab_size].copy_(
                original_lm_head.weight
            )

            # Initialize new coordinate projections
            nn.init.normal_(
                self.extended_lm_head.weight[self.original_vocab_size :],
                std=self.coordinate_config.coord_token_init_std,
            )

        # Freeze original projections by detaching them first
        with torch.no_grad():
            self.extended_lm_head.weight[: self.original_vocab_size].requires_grad_(False)

    def _forward_with_coordinate_tokens(
        self, model_inputs: Dict, original_inputs: Dict
    ):
        """Forward pass with coordinate token processing."""
        input_ids = model_inputs.get("input_ids")
        labels = original_inputs.get("labels")

        # Replace input_ids with extended embeddings
        if input_ids is not None:
            inputs_embeds = self.extended_embeddings(input_ids)
            model_inputs["inputs_embeds"] = inputs_embeds
            # Keep input_ids for shape information but mark to use inputs_embeds
            # Some models need input_ids for position/attention mask calculation

        # Forward through base model (exclude output_hidden_states if not needed)
        model_inputs["labels"] = None  # Remove labels to compute loss ourselves
        model_inputs["output_hidden_states"] = True  # Ensure we get hidden states
        outputs = self.base_model(**model_inputs)

        # Use extended LM head - get hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        logits = self.extended_lm_head(hidden_states)

        # Compute coordinate-aware loss if labels provided
        loss = None
        if labels is not None:
            loss = self._compute_coordinate_aware_loss(logits, labels)

        # Return with updated logits and loss
        if hasattr(outputs, "loss"):
            outputs.loss = loss
        if hasattr(outputs, "logits"):
            outputs.logits = logits

        return outputs

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
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # Create coordinate mask
        coord_mask = self._get_coordinate_mask(valid_labels)
        regular_mask = ~coord_mask

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
                total_loss += self.coordinate_config.regular_loss_weight * regular_loss

        # Coordinate token loss (soft expectation)
        if coord_mask.any():
            coord_logits = valid_logits[coord_mask]
            coord_labels = valid_labels[coord_mask]

            coord_loss = self._compute_soft_expectation_loss(coord_logits, coord_labels)
            total_loss += self.coordinate_config.coordinate_loss_weight * coord_loss

        return total_loss

    def _get_coordinate_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for coordinate tokens."""
        coord_start = self.original_vocab_size  # Start immediately after original vocab
        coord_end = coord_start + self.coordinate_config.max_coord_value
        return (token_ids >= coord_start) & (token_ids < coord_end)

    def _compute_soft_expectation_loss(
        self, coord_logits: torch.Tensor, coord_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft expectation loss for coordinate tokens."""
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

        # L1 loss on expected coordinates
        l1_loss = F.l1_loss(expected_coords, coord_indices.float())

        # Optional: Add focal loss on distribution for sharpness
        focal_loss = self._compute_focal_loss_on_distribution(
            soft_weights, coord_indices
        )

        return l1_loss + 0.1 * focal_loss  # Weighted combination

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
                raise ValueError(f"Coordinate {i} must be integer, got {type(coord)}: {coord}")
            if not (0 <= coord < self.coordinate_config.max_coord_value):
                raise ValueError(f"Coordinate {i} = {coord} out of bounds [0, {self.coordinate_config.max_coord_value})")

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

    @property
    def device(self):
        """Return the device of the base model"""
        return next(self.base_model.parameters()).device

    def train(self, mode=True):
        """Override train mode to handle both base model and detection head"""
        super().train(mode)
        self.base_model.train(mode)
        if self.detection_head is not None:
            self.detection_head.train(mode)
        return self

    def eval(self):
        """Override eval mode to handle both base model and detection head"""
        super().eval()
        self.base_model.eval()
        if self.detection_head is not None:
            self.detection_head.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model"""
        return self.base_model.gradient_checkpointing_disable()

    def load_detection_head_weights(self, detection_head_path: str):
        """
        Load detection head weights from a saved checkpoint.

        Args:
            detection_head_path: Path to the detection head weights file (.pth)
        """
        import os

        import torch

        if not os.path.exists(detection_head_path):
            raise FileNotFoundError(
                f"Detection head weights not found: {detection_head_path}"
            )

        # Load detection head state dict
        detection_state_dict = torch.load(detection_head_path, map_location=self.device)

        # Load weights into detection head
        self.detection_head.load_state_dict(detection_state_dict)

        print(f"✅ Detection head weights loaded from: {detection_head_path}")

    def save_detection_head_weights(self, output_dir: str):
        """
        Save detection head weights and configuration to `output_dir`.
        """
        import os

        import torch

        os.makedirs(output_dir, exist_ok=True)

        # Save detection head weights
        detection_state_dict = self.detection_head.state_dict()
        detection_path = os.path.join(output_dir, "detection_head.pth")
        torch.save(detection_state_dict, detection_path)

        # Save detection head config with UNIFIED filename (same as trainer)
        detection_config = {
            "num_queries": self.detection_head.num_queries,
            "max_caption_length": self.detection_head.max_caption_length,
            "hidden_size": self.detection_head.hidden_size,
            "vocab_size": self.detection_head.vocab_size,
            "coordinate_tokens_enabled": True,
            "checkpoint_type": "unified",  # Marker for unified checkpoint
        }

        # Use the UNIFIED config filename (same as trainer)
        config_path = os.path.join(output_dir, "detection_config.json")
        with open(config_path, "w") as f:
            json.dump(detection_config, f, indent=2)

        print(f"✅ Detection head weights saved to: {detection_path}")
        print(f"✅ Detection head config saved to: {config_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_queries: int = None,
        max_caption_length: int = None,
        tokenizer=None,
        load_detection_head: bool = True,
        attn_implementation: str = None,
        **kwargs,
    ):
        """
        Unified loading method that automatically detects checkpoint type.

        This method works with:
        1. Base Qwen2.5-VL models (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        2. Unified checkpoints with detection head (created by our trainer)

        Args:
            model_path: Path to model (base model or checkpoint directory)
            num_queries: Number of detection queries (auto-detected from checkpoint)
            max_caption_length: Max caption length (auto-detected from checkpoint)
            tokenizer: Tokenizer for the model
            load_detection_head: Whether to load the detection head weights
            attn_implementation: Override attention implementation ('flash_attention_2', 'eager', etc.)
            **kwargs: Additional arguments

        Returns:
            Qwen25VLWithDetection: Model with appropriate weights loaded
        """

        # Check what type of checkpoint this is
        checkpoint_info = cls._analyze_checkpoint(model_path)

        if checkpoint_info["type"] == "unified":
            if load_detection_head:
                return cls._load_unified_checkpoint(
                    model_path,
                    num_queries,
                    max_caption_length,
                    tokenizer,
                    attn_implementation,
                    **kwargs,
                )
            else:
                # Load just the base model part inside the unified directory
                return cls._load_base_model(
                    model_path,
                    num_queries,
                    max_caption_length,
                    tokenizer,
                    attn_implementation,
                    **kwargs,
                )
        else:  # base model
            return cls._load_base_model(
                model_path,
                num_queries,
                max_caption_length,
                tokenizer,
                attn_implementation,
                **kwargs,
            )

    @classmethod
    def _analyze_checkpoint(cls, model_path: str) -> dict:
        """Analyze checkpoint to determine its type and available components."""
        import os

        # Check for unified checkpoint markers
        detection_config_path = os.path.join(model_path, "detection_config.json")
        detection_head_path = os.path.join(model_path, "detection_head.pth")

        if os.path.exists(detection_config_path) and os.path.exists(
            detection_head_path
        ):
            # Load detection config to get parameters
            with open(detection_config_path, "r") as f:
                detection_config = json.load(f)

            return {
                "type": "unified",
                "has_detection": True,
                "detection_config": detection_config,
                "detection_head_path": detection_head_path,
            }
        else:
            # Base model without detection head
            return {
                "type": "base",
                "has_detection": False,
            }

    @classmethod
    def _load_unified_checkpoint(
        cls,
        model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        attn_implementation: str = None,
        **kwargs,
    ):
        """Load from unified checkpoint created by our trainer."""
        checkpoint_info = cls._analyze_checkpoint(model_path)

        # ------------------------------------------------------------------
        # 🔒 Strict validation – missing keys raise immediately
        # ------------------------------------------------------------------

        @dataclass
        class DetectionHeadConfig:
            num_queries: int
            max_caption_length: int

            def __post_init__(self):
                if self.num_queries <= 0:
                    raise ValueError("num_queries must be > 0")
                if self.max_caption_length <= 0:
                    raise ValueError("max_caption_length must be > 0")

        raw_cfg: dict = checkpoint_info["detection_config"]

        # Fail-fast if required keys are absent → KeyError surfaces naturally.
        cfg = DetectionHeadConfig(
            num_queries=raw_cfg["num_queries"],
            max_caption_length=raw_cfg["max_caption_length"],
        )

        # Command-line / caller overrides still take precedence ----------------
        if num_queries is None:
            num_queries = cfg.num_queries
        if max_caption_length is None:
            max_caption_length = cfg.max_caption_length

        # Final sanity check – values must now be concrete integers.
        if num_queries is None or max_caption_length is None:
            raise ValueError(
                "num_queries and max_caption_length must be provided either via "
                "function arguments or checkpoint detection_config.json"
            )

        print(f"🔄 Loading unified checkpoint from: {model_path}")
        print(f"   Detection queries: {num_queries}")
        print(f"   Max caption length: {max_caption_length}")

        # Create model with checkpoint as base model path
        model = cls(
            base_model_path=model_path,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
            attn_implementation=attn_implementation,
        )

        # Load detection head weights
        model.load_detection_head_weights(checkpoint_info["detection_head_path"])

        # CRITICAL: Load coordinate token extensions if they exist
        model._load_coordinate_extensions(model_path)

        print(f"✅ Unified checkpoint loaded successfully")
        return model

    @classmethod
    def _load_base_model(
        cls,
        model_path: str,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        attn_implementation: str = None,
        **kwargs,
    ):
        """Load base model without detection head (randomly initialized)."""
        # ------------------------------------------------------------------
        # No silent defaults – the caller *must* specify the detection head
        # dimensions when loading a *base* model.
        # ------------------------------------------------------------------

        if num_queries is None or max_caption_length is None:
            raise ValueError(
                "Loading a base model requires explicit num_queries and "
                "max_caption_length parameters. No implicit defaults are "
                "provided."
            )

        print(f"🔄 Loading base model from: {model_path}")
        print(f"   Detection head will be randomly initialized")
        print(f"   Detection queries: {num_queries}")
        print(f"   Max caption length: {max_caption_length}")

        # Create model with randomly initialized detection head
        model = cls(
            base_model_path=model_path,
            num_queries=num_queries,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
            attn_implementation=attn_implementation,
        )

        print(f"✅ Base model loaded with random detection head")
        return model

    @staticmethod
    def inspect_checkpoint(model_path: str) -> dict:
        """
        Inspect a checkpoint to understand its type and contents.

        This is a utility function to help users understand what type of
        checkpoint they have without loading the full model.

        Args:
            model_path: Path to model or checkpoint directory

        Returns:
            dict: Information about the checkpoint
        """
        import os

        checkpoint_info = Qwen25VLWithDetection._analyze_checkpoint(model_path)

        # Add more detailed information
        result = {
            "path": model_path,
            "type": checkpoint_info["type"],
            "has_detection_head": checkpoint_info["has_detection"],
            "description": "",
            "files_found": [],
        }

        # Check what files exist
        common_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "detection_head.pth",
            "detection_config.json",
        ]

        for file in common_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                result["files_found"].append(file)

        # Add descriptions
        if checkpoint_info["type"] == "unified":
            result["description"] = (
                "Unified checkpoint with both base model and detection head"
            )
            if "detection_config" in checkpoint_info:
                result["detection_config"] = checkpoint_info["detection_config"]
        else:
            result["description"] = "Base Qwen2.5-VL model without detection head"

        return result

    @staticmethod
    def print_checkpoint_info(model_path: str):
        """
        Print human-readable information about a checkpoint.

        Args:
            model_path: Path to model or checkpoint directory
        """
        info = Qwen25VLWithDetection.inspect_checkpoint(model_path)

        print(f"📁 Checkpoint Analysis: {model_path}")
        print(f"   Type: {info['type'].upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Has Detection Head: {'✅' if info['has_detection_head'] else '❌'}")

        if info["files_found"]:
            print(f"   Files Found:")
            for file in info["files_found"]:
                print(f"     - {file}")

        if "detection_config" in info:
            config = info["detection_config"]
            print(f"   Detection Config:")
            print(f"     - Queries: {config.get('num_queries', 'N/A')}")
            print(
                f"     - Max Caption Length: {config.get('max_caption_length', 'N/A')}"
            )
            print(f"     - Hidden Size: {config.get('hidden_size', 'N/A')}")
            print(f"     - Vocab Size: {config.get('vocab_size', 'N/A')}")

        print()

    def set_detection_loss_fn(self, detection_loss_fn):
        """Set the detection loss function for use in forward pass."""
        self._detection_loss_fn = detection_loss_fn

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
        # 2. Save detection-specific weights & metadata
        # ------------------------------------------------------------------
        if self.detection_head is not None:
            self.save_detection_head_weights(save_directory)

        # ------------------------------------------------------------------
        # 3. CRITICAL: Save coordinate token extensions if enabled
        # ------------------------------------------------------------------
        if self.coordinate_tokens_enabled and hasattr(self, 'extended_embeddings') and hasattr(self, 'extended_lm_head'):
            import torch
            
            coord_weights_path = os.path.join(save_directory, "coordinate_extensions.pt")
            coord_metadata = {
                'coordinate_tokens_enabled': True,
                'original_vocab_size': self.original_vocab_size,
                'extended_vocab_size': self.extended_vocab_size,
                'coordinate_config': {
                    'max_coord_value': self.coordinate_config.max_coord_value,
                    'enable_coordinate_tokens': self.coordinate_config.enable_coordinate_tokens,
                    'use_official_box_tokens': self.coordinate_config.use_official_box_tokens,
                },
                'extended_embeddings': self.extended_embeddings.state_dict(),
                'extended_lm_head': self.extended_lm_head.state_dict(),
                'box_start_id': self.box_start_id,
                'box_end_id': self.box_end_id,
            }
            
            torch.save(coord_metadata, coord_weights_path)
            print(f"✅ Saved coordinate token extensions to {coord_weights_path}")
            print(f"   Original vocab: {self.original_vocab_size}")
            print(f"   Extended vocab: {self.extended_vocab_size}")
            print(f"   Coordinate tokens: {self.coordinate_config.max_coord_value}")

        # NOTE: Tokenizer / processor saving is handled by Trainer once per
        # checkpoint; duplicating here is unnecessary and may overwrite user
        # modifications.
        print(f"✅ save_pretrained completed for directory: {save_directory}")

    def _load_coordinate_extensions(self, model_path: str):
        """Load coordinate token extensions if they exist."""
        import os
        import torch
        
        coord_weights_path = os.path.join(model_path, "coordinate_extensions.pt")
        
        if not os.path.exists(coord_weights_path):
            print("📄 No coordinate extensions found - using standard model")
            return
        
        try:
            print(f"🔧 Loading coordinate token extensions from {coord_weights_path}")
            coord_metadata = torch.load(coord_weights_path, map_location='cpu')
            
            # Restore coordinate configuration
            self.coordinate_tokens_enabled = coord_metadata['coordinate_tokens_enabled']
            self.original_vocab_size = coord_metadata['original_vocab_size']
            self.extended_vocab_size = coord_metadata['extended_vocab_size']
            self.box_start_id = coord_metadata['box_start_id']
            self.box_end_id = coord_metadata['box_end_id']
            
            # Restore coordinate config
            coord_config_data = coord_metadata['coordinate_config']
            self.coordinate_config = CoordinateConfig(
                enable_coordinate_tokens=coord_config_data['enable_coordinate_tokens'],
                max_coord_value=coord_config_data['max_coord_value'],
                use_official_box_tokens=coord_config_data['use_official_box_tokens'],
            )
            
            # Recreate extended embeddings and LM head with correct shapes
            hidden_size = self.base_model.get_input_embeddings().weight.shape[1]
            
            # Extended embeddings
            self.extended_embeddings = nn.Embedding(
                self.extended_vocab_size,
                hidden_size,
                device=self.base_model.device,
                dtype=self.base_model.dtype,
            )
            self.extended_embeddings.load_state_dict(coord_metadata['extended_embeddings'])
            
            # Extended LM head
            self.extended_lm_head = nn.Linear(
                hidden_size,
                self.extended_vocab_size,
                bias=False,
                device=self.base_model.device,
                dtype=self.base_model.dtype,
            )
            self.extended_lm_head.load_state_dict(coord_metadata['extended_lm_head'])
            
            print(f"✅ Coordinate token extensions loaded successfully")
            print(f"   Original vocab: {self.original_vocab_size}")
            print(f"   Extended vocab: {self.extended_vocab_size}")
            print(f"   Coordinate tokens: {self.coordinate_config.max_coord_value}")
            
            # Coordinate token models loaded successfully
            
            # CRITICAL: Add coordinate tokens to tokenizer if missing
            coordinate_tokens = [f"<coord_{i}>" for i in range(self.coordinate_config.max_coord_value)]
            existing_tokens = set(self.tokenizer.get_vocab().keys())
            missing_tokens = [token for token in coordinate_tokens if token not in existing_tokens]
            
            if missing_tokens:
                print(f"🔧 Adding {len(missing_tokens)} coordinate tokens to tokenizer")
                num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
                print(f"✅ Added {num_added} coordinate tokens to tokenizer")
            else:
                print(f"✅ All {len(coordinate_tokens)} coordinate tokens already in tokenizer")
            
        except Exception as e:
            print(f"❌ Failed to load coordinate extensions: {e}")
            print("📄 Falling back to standard model")
            self.coordinate_tokens_enabled = False
