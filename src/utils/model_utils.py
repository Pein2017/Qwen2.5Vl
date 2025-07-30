"""
Model Utilities

This module contains utilities for model loading, patching helpers, and
model-specific schema definitions for the BBU training pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torchtyping import TensorType

from src.logger_utils import get_logger


logger = get_logger("model_utils")

# ==============================================================================
# Model Schema Definitions
# ==============================================================================

# Type aliases for tensor shapes
AttentionMaskType = TensorType["B", "S"]
PixelValuesType = TensorType["I", "C", "H", "W"]
ImageGridThwType = TensorType["I", 3]
InputIdsType = TensorType["B", "S"]
LabelsType = TensorType["B", "S"]
PredBoxesType = TensorType["B", "N", 4]
PredBoxesRawType = TensorType["B", "N", 4]
PredObjectnessType = TensorType["B", "N"]
ObjectFeaturesType = TensorType["B", "N", "D"]

# Collated batch pixel and grid shapes
CollatedPixelType = PixelValuesType
CollatedGridThwType = TensorType["B_IMAGES", 3]  # B_IMAGES = total images across batch


@dataclass
class ModelAssets:  # noqa: D401 – composite holder
    """Convenience container bundling all core HF components.

    Useful for IDE navigation (*jump-to-definition*) and for explicit type
    signatures inside the training / inference helpers.
    """

    config: Any  # HF config object
    tokenizer: Any  # HF tokenizer
    image_processor: Any  # HF image processor
    model: torch.nn.Module

    def __post_init__(self) -> None:  # noqa: D401
        assert hasattr(self.config, "model_type"), "config appears invalid."
        # Minimal attribute sanity checks
        for attr in [
            (self.tokenizer, "pad_token_id"),
            (self.image_processor, "size"),
        ]:
            obj, name = attr
            if not hasattr(obj, name):
                raise AttributeError(f"{obj.__class__.__name__} missing '{name}'")


@dataclass
class ModelInputs:
    """Structure for input dict into Qwen2.5-VL forward pass."""

    input_ids: InputIdsType
    attention_mask: AttentionMaskType
    position_ids: Optional[InputIdsType] = None
    pixel_values: Optional[PixelValuesType] = None
    image_grid_thw: Optional[ImageGridThwType] = None
    labels: Optional[LabelsType] = None
    # Additional keys (after collation)
    image_counts_per_sample: Optional[list] = None
    ground_truth_objects: Optional[list] = None

    def __post_init__(self):
        B, S = self.input_ids.shape
        if self.attention_mask.shape != (B, S):
            raise AssertionError(
                f"ModelInputs: attention_mask shape {self.attention_mask.shape} must be {(B, S)}"
            )
        if self.labels is not None and self.labels.shape != (B, S):
            raise AssertionError(
                f"ModelInputs: labels shape {self.labels.shape} must be {(B, S)}"
            )
        # Validate pixel and grid consistency
        if self.pixel_values is not None or self.image_grid_thw is not None:
            if self.pixel_values is None or self.image_grid_thw is None:
                raise AssertionError(
                    "ModelInputs: pixel_values and image_grid_thw must be both present or both None"
                )


@dataclass
class ModelOutput:
    """Structure of Qwen2.5-VL forward output (before detection head)."""

    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None  # Scalar
    rope_deltas: Optional[torch.Tensor] = None

    def __post_init__(self):
        B, S, _ = self.logits.shape
        if self.hidden_states is not None:
            for h in self.hidden_states:
                if h.shape[:2] != (B, S):
                    raise AssertionError(
                        f"ModelOutput: hidden_states each must have shape (B,S,*) got {h.shape}"
                    )
        if self.attentions is not None:
            for att in self.attentions:
                if att.ndim != 4 or att.shape[0] != B or att.shape[-1] != S:
                    raise AssertionError(
                        f"ModelOutput: attentions must be (B,h,S,S), got {att.shape}"
                    )


@dataclass
class LLMHiddenStates:
    """Final hidden states from Qwen model."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self):
        if self.hidden_states.shape[:2] != self.attention_mask.shape:
            raise AssertionError(
                f"LLMHiddenStates: hidden_states batch and sequence dims {self.hidden_states.shape[:2]} must equal attention_mask shape {self.attention_mask.shape}"
            )


@dataclass
class DetectionHeadOutputs:
    """Output of DetectionHead.forward."""

    pred_boxes: torch.Tensor
    pred_boxes_raw: torch.Tensor
    pred_objectness: torch.Tensor
    caption_logits: torch.Tensor

    def __post_init__(self):
        B, N = self.pred_boxes.shape[:2]
        if self.pred_boxes.shape != (B, N, 4):
            raise AssertionError(
                f"DetectionHeadOutputs: pred_boxes must be (B,N,4), got {self.pred_boxes.shape}"
            )
        if self.pred_boxes_raw.shape != (B, N, 4):
            raise AssertionError(
                f"DetectionHeadOutputs: pred_boxes_raw must be (B,N,4), got {self.pred_boxes_raw.shape}"
            )
        if self.pred_objectness.shape != (B, N):
            raise AssertionError(
                f"DetectionHeadOutputs: pred_objectness must be (B,N), got {self.pred_objectness.shape}"
            )


@dataclass
class VisionFeatures:
    """Container for vision model features."""

    features: torch.Tensor  # [B, N_patches, D]
    patch_embeddings: Optional[torch.Tensor] = None
    spatial_features: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.features.dim() != 3:
            raise AssertionError(
                f"VisionFeatures: features must be 3D tensor, got shape {self.features.shape}"
            )


# ==============================================================================
# Model Input Processing Functions
# ==============================================================================


def filter_inputs_for_model(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter inputs to keep only keys needed for model forward pass."""
    # Define keys that the model expects
    model_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "labels",
    }

    filtered_inputs = {}
    for key, value in inputs.items():
        if key in model_keys and value is not None:
            filtered_inputs[key] = value

    return filtered_inputs


def filter_inputs_for_generation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter inputs for generation (removes labels and training-specific keys)."""
    # Define keys needed for generation
    generation_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        # Generation parameters
        "max_length",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "pad_token_id",
        "eos_token_id",
    }

    filtered_inputs = {}
    for key, value in inputs.items():
        if key in generation_keys and value is not None:
            filtered_inputs[key] = value

    return filtered_inputs


def _validate_and_fix_shapes(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix tensor shapes for model compatibility."""
    fixed_inputs = {}

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            # Ensure proper tensor dimensions
            if key == "input_ids":
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # Add batch dimension
                elif value.dim() > 2:
                    logger.warning(f"input_ids has unexpected shape: {value.shape}")

            elif key == "attention_mask":
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # Add batch dimension
                # Ensure boolean type for efficiency
                if value.dtype != torch.bool:
                    value = value.bool()

            elif key == "labels":
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # Add batch dimension
                # Ensure long type for cross entropy
                if value.dtype != torch.long:
                    value = value.long()

            elif key == "position_ids":
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # Add batch dimension

            fixed_inputs[key] = value
        else:
            fixed_inputs[key] = value

    return fixed_inputs


# ==============================================================================
# Model Loading and Asset Management
# ==============================================================================


def load_model_assets(
    model_path: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **kwargs,
) -> ModelAssets:
    """Load all model assets (config, tokenizer, image processor, model)."""
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.float16 if device != "cpu" else torch.float32

    logger.info(f"Loading model assets from {model_path}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Load config
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=device if isinstance(device, str) else str(device),
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    return ModelAssets(
        config=config, tokenizer=tokenizer, image_processor=image_processor, model=model
    )


def prepare_model_for_training(
    model: torch.nn.Module,
    gradient_checkpointing: bool = True,
    compile_model: bool = False,
    **kwargs,
) -> torch.nn.Module:
    """Prepare model for training with optimizations."""
    logger.info("Preparing model for training")

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Compile model for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = cast(torch.nn.Module, torch.compile(model, **kwargs))
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")

    return model


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, Any]:
    """Get model memory usage information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage (rough approximation)
    param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

    # Check if model is on GPU
    device_info = {}
    if next(model.parameters()).is_cuda:
        device = next(model.parameters()).device
        device_info = {
            "device": str(device),
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
            "cached_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        }

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_memory_mb": param_memory_mb,
        "device_info": device_info,
    }


def log_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Log comprehensive model information."""
    logger.info(f"📋 {model_name.upper()} INFORMATION")

    # Memory usage
    memory_info = get_model_memory_usage(model)
    logger.info(f"  Total parameters: {memory_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {memory_info['trainable_parameters']:,}")
    logger.info(f"  Parameter memory: {memory_info['parameter_memory_mb']:.1f}MB")

    # Device information
    if memory_info["device_info"]:
        device_info = memory_info["device_info"]
        logger.info(f"  Device: {device_info['device']}")
        logger.info(f"  GPU memory allocated: {device_info['allocated_mb']:.1f}MB")
        logger.info(f"  GPU memory cached: {device_info['cached_mb']:.1f}MB")

    # Model architecture info
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "hidden_size"):
            logger.info(f"  Hidden size: {config.hidden_size}")
        if hasattr(config, "num_hidden_layers"):
            logger.info(f"  Number of layers: {config.num_hidden_layers}")
        if hasattr(config, "num_attention_heads"):
            logger.info(f"  Attention heads: {config.num_attention_heads}")


# ==============================================================================
# Model Input/Output Validation
# ==============================================================================


def assert_model_inputs(inputs: ModelInputs):
    """Assert correctness of a model input bundle."""
    # The __post_init__ method already validates the structure
    pass


def assert_model_output(output: ModelOutput):
    """Assert correctness of a model forward output."""
    # The __post_init__ method already validates the structure
    pass


def assert_detection_head_outputs(outputs: DetectionHeadOutputs):
    """Validate detection head outputs."""
    # The __post_init__ method already validates the structure
    pass


def assert_llm_hidden_states(
    hidden_states: LLMHiddenStates, attention_mask: torch.Tensor
):
    """Validate LLM hidden states."""
    if hidden_states.attention_mask.shape != attention_mask.shape:
        raise AssertionError(
            f"Hidden states attention mask {hidden_states.attention_mask.shape} "
            f"must match provided attention mask {attention_mask.shape}"
        )


def assert_vision_features(vision_feats: VisionFeatures):
    """Validate vision features."""
    # The __post_init__ method already validates the structure
    pass


# ==============================================================================
# Utility Functions for Vision Processing
# ==============================================================================


def ensure_batched_vision_feats(
    vision_feats: Optional[torch.Tensor], batch_size: int
) -> Optional[torch.Tensor]:
    """Ensure vision features have correct batch dimension."""
    if vision_feats is None:
        return None

    if vision_feats.dim() == 2:
        # Add batch dimension if missing
        vision_feats = vision_feats.unsqueeze(0)

    # Repeat for batch if needed
    if vision_feats.shape[0] == 1 and batch_size > 1:
        vision_feats = vision_feats.expand(batch_size, -1, -1)
    elif vision_feats.shape[0] != batch_size:
        logger.warning(
            f"Vision features batch size {vision_feats.shape[0]} "
            f"doesn't match expected batch size {batch_size}"
        )

    return vision_feats


def merge_vision_tokens(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, merge_size: int = 2
) -> torch.Tensor:
    """Merge vision tokens according to Qwen2.5-VL specifications."""
    # This is a simplified version - full implementation would handle
    # the complex vision token merging process
    batch_size = pixel_values.shape[0]

    # Calculate merged dimensions
    merged_tokens = []
    for i in range(batch_size):
        grid = image_grid_thw[i]
        t, h, w = grid[0], grid[1], grid[2]

        # Calculate number of tokens after merging
        merged_h = h // merge_size
        merged_w = w // merge_size
        num_tokens = t * merged_h * merged_w

        merged_tokens.append(num_tokens)

    logger.debug(
        f"Vision token merging: {pixel_values.shape[0]} -> {sum(merged_tokens)} tokens"
    )
    return pixel_values  # Simplified - would return merged tokens


# ==============================================================================
# Model Patching and Compatibility
# ==============================================================================


def patch_model_for_coordinate_tokens(
    model: torch.nn.Module,
    special_tokens: Dict[str, int],
    coordinate_vocab_size: int = 1000,
) -> torch.nn.Module:
    """Patch model to support coordinate tokens."""
    logger.info("Patching model for coordinate token support")

    # Get current vocab size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        current_vocab_size = model.config.vocab_size
    else:
        logger.warning("Could not determine model vocab size")
        return model

    # Calculate new vocab size
    new_vocab_size = current_vocab_size + len(special_tokens) + coordinate_vocab_size

    # Resize model embeddings if needed
    if hasattr(model, "resize_token_embeddings"):
        old_embeddings = model.get_input_embeddings()
        logger.info(f"Resizing embeddings: {current_vocab_size} -> {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)

        # Initialize new embeddings
        with torch.no_grad():
            new_embeddings = model.get_input_embeddings()
            # Copy old weights
            new_embeddings.weight[:current_vocab_size] = old_embeddings.weight
            # Initialize special tokens with small random values
            new_embeddings.weight[current_vocab_size:].normal_(mean=0.0, std=0.02)

    logger.info(f"Model patched for coordinate tokens: vocab_size = {new_vocab_size}")
    return model


def apply_model_patches(
    model: torch.nn.Module, patches: Dict[str, Any]
) -> torch.nn.Module:
    """Apply various model patches based on configuration."""
    for patch_name, patch_config in patches.items():
        if patch_name == "coordinate_tokens" and patch_config.get("enabled", False):
            model = patch_model_for_coordinate_tokens(
                model,
                special_tokens=patch_config.get("special_tokens", {}),
                coordinate_vocab_size=patch_config.get("vocab_size", 1000),
            )
        elif patch_name == "flash_attention" and patch_config.get("enabled", False):
            logger.info(
                "Flash attention patch requested (requires model-specific implementation)"
            )
        else:
            logger.debug(f"Unknown or disabled patch: {patch_name}")

    return model


# ==============================================================================
# Export All Functions and Classes
# ==============================================================================

__all__ = [
    # Type aliases
    "AttentionMaskType",
    "PixelValuesType",
    "ImageGridThwType",
    "InputIdsType",
    "LabelsType",
    "PredBoxesType",
    "PredBoxesRawType",
    "PredObjectnessType",
    "ObjectFeaturesType",
    "CollatedPixelType",
    "CollatedGridThwType",
    # Schema classes
    "ModelAssets",
    "ModelInputs",
    "ModelOutput",
    "LLMHiddenStates",
    "DetectionHeadOutputs",
    "VisionFeatures",
    # Input processing functions
    "filter_inputs_for_model",
    "filter_inputs_for_generation",
    "_validate_and_fix_shapes",
    # Model loading functions
    "load_model_assets",
    "prepare_model_for_training",
    "get_model_memory_usage",
    "log_model_info",
    # Validation functions
    "assert_model_inputs",
    "assert_model_output",
    "assert_detection_head_outputs",
    "assert_llm_hidden_states",
    "assert_vision_features",
    # Vision processing functions
    "ensure_batched_vision_feats",
    "merge_vision_tokens",
    # Model patching functions
    "patch_model_for_coordinate_tokens",
    "apply_model_patches",
]
