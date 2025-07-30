"""
Training Utilities

This module contains utilities for training helpers, metrics, callbacks, and
training-specific schema definitions for the BBU training pipeline.
"""

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping, Sequence

from src.logger_utils import get_logger

logger = get_logger("training_utils")

# Constants for training
IGNORE_INDEX = -100

# ==============================================================================
# Training Schema Definitions
# ==============================================================================

# Type alias for loss dictionaries
LossDictType = Dict[str, torch.Tensor]

@dataclass
class ChatProcessorOutput:
    """Output from ChatProcessor after processing a sample."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    ground_truth_objects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Token spans for teacher-student loss splitting
    teacher_assistant_spans: List[Tuple[int, int]] = field(default_factory=list)
    student_assistant_spans: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Validate the output structure."""
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D, got shape {self.input_ids.shape}")
        if self.labels.shape != self.input_ids.shape:
            raise ValueError(
                f"labels shape {self.labels.shape} must match input_ids shape {self.input_ids.shape}"
            )
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"attention_mask shape {self.attention_mask.shape} must match input_ids shape {self.input_ids.shape}"
            )


@dataclass
class CollatedBatch:
    """Output of data collator with validated structure."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    image_counts_per_sample: list[int]
    ground_truth_objects: list[list[Dict[str, Any]]]
    position_ids: Optional[torch.Tensor] = None

    # Token spans for teacher-student loss splitting (batch-level)
    teacher_assistant_spans: list[list[tuple[int, int]]] = field(
        default_factory=list
    )  # [[(start, end), ...], ...]
    student_assistant_spans: list[list[tuple[int, int]]] = field(
        default_factory=list
    )  # [[(start, end), ...], ...]

    def __post_init__(self):
        """Validate batch structure."""
        B, S = self.input_ids.shape
        if self.labels.shape != (B, S) or self.attention_mask.shape != (B, S):
            raise AssertionError(
                f"CollatedBatch: mismatched shapes input_ids {self.input_ids.shape}, labels {self.labels.shape}, attention_mask {self.attention_mask.shape}"
            )
        if self.position_ids is not None:
            # Accept either (B,S) or (3,B,S)
            ok = False
            if self.position_ids.shape == (B, S):
                ok = True
            elif self.position_ids.shape == (3, B, S):
                ok = True
            if not ok:
                raise AssertionError(
                    f"CollatedBatch: position_ids shape {self.position_ids.shape} must be (B,S) or (3,B,S) with B={B}, S={S}"
                )
        
        total_images = sum(self.image_counts_per_sample)
        if self.pixel_values is not None:
            if self.image_grid_thw is None:
                raise AssertionError(
                    "CollatedBatch: pixel_values present but image_grid_thw is None"
                )
            if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
                raise AssertionError(
                    f"CollatedBatch: image_grid_thw must have shape (N,3), got {self.image_grid_thw.shape}"
                )

            # Compute expected number of flattened patches across **all** images
            tokens_expected = int(
                (
                    self.image_grid_thw[:, 0]
                    * self.image_grid_thw[:, 1]
                    * self.image_grid_thw[:, 2]
                )
                .sum()
                .item()
            )

            if self.pixel_values.shape[0] != tokens_expected:
                raise AssertionError(
                    f"CollatedBatch: pixel_values first dimension ({self.pixel_values.shape[0]}) does not match total flattened patches ({tokens_expected}) from image_grid_thw"
                )
        else:
            if total_images != 0:
                raise AssertionError(
                    f"CollatedBatch: no pixel_values but image_counts_per_sample sum is {total_images}"
                )
        
        if len(self.ground_truth_objects) != B:
            raise AssertionError(
                f"CollatedBatch: ground_truth_objects length {len(self.ground_truth_objects)} must equal batch size {B}"
            )


# ==============================================================================
# Training Helper Functions
# ==============================================================================

def debug_input_shapes(
    inputs: Dict[str, Any],
    labels: Optional[torch.Tensor] = None,
    batch_info: Optional[str] = None,
    detailed: bool = False,
) -> None:
    """Debug input shapes for training step."""
    logger.debug(f"🔍 DEBUG INPUT SHAPES {batch_info or ''}")
    logger.debug(f"Input keys: {list(inputs.keys())}")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            logger.debug(f"  {key}: {value.shape} ({value.dtype})")
            if detailed and value.numel() > 0:
                logger.debug(f"    min: {value.min().item():.4f}, max: {value.max().item():.4f}")
        elif isinstance(value, (list, tuple)):
            logger.debug(f"  {key}: {type(value).__name__} len={len(value)}")
        else:
            logger.debug(f"  {key}: {type(value).__name__}")
    
    if labels is not None:
        logger.debug(f"  labels: {labels.shape} ({labels.dtype})")
        if detailed:
            non_ignore = (labels != IGNORE_INDEX).sum().item()
            logger.debug(f"    non-ignore tokens: {non_ignore}/{labels.numel()}")


def prepare_inputs_for_forward(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare inputs for model forward pass."""
    # Create a copy to avoid modifying the original
    prepared_inputs = {}
    
    # Essential fields for forward pass
    forward_keys = {
        "input_ids", "attention_mask", "labels", 
        "pixel_values", "image_grid_thw", "position_ids"
    }
    
    for key, value in inputs.items():
        if key in forward_keys and value is not None:
            if isinstance(value, torch.Tensor):
                prepared_inputs[key] = value
            else:
                # Convert to tensor if needed
                try:
                    prepared_inputs[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    # Skip if can't convert
                    continue
    
    # Validate essential fields
    if "input_ids" not in prepared_inputs:
        raise ValueError("input_ids is required for forward pass")
    if "attention_mask" not in prepared_inputs:
        raise ValueError("attention_mask is required for forward pass")
    
    return prepared_inputs


def prepare_inputs_for_generate(
    inputs: Dict[str, Any],
    generation_config: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Prepare inputs for model generation."""
    # Start with forward inputs
    prepared_inputs = prepare_inputs_for_forward(inputs)
    
    # Remove labels for generation
    prepared_inputs.pop("labels", None)
    
    # Add generation parameters
    if generation_config:
        prepared_inputs.update(generation_config)
    else:
        prepared_inputs.update({
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": inputs.get("pad_token_id", 0),
            "eos_token_id": inputs.get("eos_token_id", 2),
        })
    
    return prepared_inputs


def validate_attention_mask_consistency(inputs: Dict[str, Any]) -> bool:
    """Validate attention mask consistency with input_ids."""
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        return True  # Can't validate without both
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    if input_ids.shape != attention_mask.shape:
        logger.warning(
            f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}"
        )
        return False
    
    # Check for reasonable attention patterns
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        mask = attention_mask[i]
        if mask.sum() == 0:
            logger.warning(f"Sample {i} has all-zero attention mask")
            return False
        
        # Check if mask starts with padding (left padding)
        first_true = torch.argmax(mask.float())
        if first_true > 0:
            # Verify left padding pattern
            if mask[:first_true].any():
                logger.warning(f"Sample {i} has invalid padding pattern")
                return False
    
    return True


def fix_attention_mask_mismatch(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Fix attention mask mismatch issues."""
    if "input_ids" not in inputs:
        return inputs
    
    input_ids = inputs["input_ids"]
    batch_size, seq_len = input_ids.shape
    
    # Create new attention mask if missing or wrong shape
    if "attention_mask" not in inputs or inputs["attention_mask"].shape != input_ids.shape:
        logger.info("Creating new attention mask")
        inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.bool)
    
    return inputs


def safe_prepare_inputs(
    inputs: Dict[str, Any], 
    mode: str = "forward",
    **kwargs
) -> Dict[str, Any]:
    """Safely prepare inputs with validation and error recovery."""
    try:
        # Validate inputs first
        if not validate_attention_mask_consistency(inputs):
            inputs = fix_attention_mask_mismatch(inputs)
        
        # Prepare based on mode
        if mode == "forward":
            return prepare_inputs_for_forward(inputs)
        elif mode == "generate":
            return prepare_inputs_for_generate(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    except Exception as e:
        logger.error(f"Failed to prepare inputs for {mode}: {e}")
        raise


# ==============================================================================
# Validation Functions
# ==============================================================================

def assert_tensor_shape(fn):
    """Decorator to assert tensor shapes match expected patterns."""
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            # Add shape validation logic here if needed
            return result
        except Exception as e:
            logger.error(f"Tensor shape assertion failed in {fn.__name__}: {e}")
            raise
    return wrapper


def assert_chat_processor_output(sample: ChatProcessorOutput):
    """Assert ChatProcessorOutput has valid structure."""
    if not isinstance(sample, ChatProcessorOutput):
        raise AssertionError(f"Expected ChatProcessorOutput, got {type(sample)}")
    
    # Validate tensor shapes
    B, S = sample.input_ids.shape
    if sample.labels.shape != (B, S):
        raise AssertionError(f"Labels shape {sample.labels.shape} must match input_ids {(B, S)}")
    if sample.attention_mask.shape != (B, S):
        raise AssertionError(f"Attention mask shape {sample.attention_mask.shape} must match input_ids {(B, S)}")


def assert_collated_batch(batch: CollatedBatch):
    """Assert CollatedBatch has valid structure."""
    if not isinstance(batch, CollatedBatch):
        raise AssertionError(f"Expected CollatedBatch, got {type(batch)}")
    
    # The __post_init__ method already handles validation


def assert_model_inputs(inputs: Dict[str, Any]):
    """Assert model inputs are valid for training."""
    required_keys = ["input_ids", "attention_mask"]
    for key in required_keys:
        if key not in inputs:
            raise AssertionError(f"Missing required input key: {key}")
        if not isinstance(inputs[key], torch.Tensor):
            raise AssertionError(f"Input {key} must be a tensor, got {type(inputs[key])}")
    
    # Validate shapes
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.shape != attention_mask.shape:
        raise AssertionError(
            f"input_ids shape {input_ids.shape} must match attention_mask shape {attention_mask.shape}"
        )


def assert_model_output(output: Dict[str, Any]):
    """Assert model output is valid."""
    if "loss" in output:
        loss = output["loss"]
        if not isinstance(loss, torch.Tensor):
            raise AssertionError(f"Loss must be a tensor, got {type(loss)}")
        if loss.dim() != 0:
            raise AssertionError(f"Loss must be a scalar tensor, got shape {loss.shape}")
        if torch.isnan(loss) or torch.isinf(loss):
            raise AssertionError(f"Loss contains invalid values: {loss.item()}")


# ==============================================================================
# Memory and Performance Utilities
# ==============================================================================

def get_tensor_memory_usage(tensor: torch.Tensor) -> Dict[str, Any]:
    """Get memory usage information for a tensor."""
    if not isinstance(tensor, torch.Tensor):
        return {"error": "Not a tensor"}
    
    element_size = tensor.element_size()
    numel = tensor.numel()
    total_bytes = element_size * numel
    
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "element_size_bytes": element_size,
        "total_elements": numel,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "is_contiguous": tensor.is_contiguous(),
    }


def log_batch_memory_usage(batch: Dict[str, Any], batch_info: str = ""):
    """Log memory usage for a batch."""
    logger.debug(f"📊 BATCH MEMORY USAGE {batch_info}")
    
    total_memory_mb = 0
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            usage = get_tensor_memory_usage(value)
            total_memory_mb += usage["total_mb"]
            logger.debug(f"  {key}: {usage['shape']} ({usage['dtype']}) = {usage['total_mb']:.2f}MB")
        elif isinstance(value, (list, tuple)):
            logger.debug(f"  {key}: {type(value).__name__} len={len(value)}")
    
    logger.debug(f"  Total tensor memory: {total_memory_mb:.2f}MB")


def optimize_batch_for_memory(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize batch for memory usage."""
    optimized_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Ensure tensors are contiguous for better memory access
            if not value.is_contiguous():
                value = value.contiguous()
            
            # Move to appropriate dtype if needed
            if key in ["attention_mask"] and value.dtype != torch.bool:
                value = value.bool()
            elif key in ["labels"] and value.dtype != torch.long:
                value = value.long()
            
            optimized_batch[key] = value
        else:
            optimized_batch[key] = value
    
    return optimized_batch


# ==============================================================================
# Training State Management
# ==============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    total_loss: float = 0.0
    teacher_loss: float = 0.0
    student_loss: float = 0.0
    coordinate_loss: float = 0.0
    language_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    step: int = 0
    epoch: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "total_loss": self.total_loss,
            "teacher_loss": self.teacher_loss,
            "student_loss": self.student_loss,
            "coordinate_loss": self.coordinate_loss,
            "language_loss": self.language_loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "step": float(self.step),
            "epoch": float(self.epoch),
        }
    
    def reset(self):
        """Reset metrics to zero."""
        self.total_loss = 0.0
        self.teacher_loss = 0.0
        self.student_loss = 0.0
        self.coordinate_loss = 0.0
        self.language_loss = 0.0
        self.gradient_norm = 0.0


def create_training_metrics(
    loss_dict: Dict[str, torch.Tensor],
    learning_rate: float = 0.0,
    gradient_norm: float = 0.0,
    step: int = 0,
    epoch: int = 0
) -> TrainingMetrics:
    """Create TrainingMetrics from loss dictionary."""
    metrics = TrainingMetrics(
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        step=step,
        epoch=epoch
    )
    
    # Extract losses from dictionary
    if "loss" in loss_dict:
        metrics.total_loss = loss_dict["loss"].item()
    if "teacher_loss" in loss_dict:
        metrics.teacher_loss = loss_dict["teacher_loss"].item()
    if "student_loss" in loss_dict:
        metrics.student_loss = loss_dict["student_loss"].item()
    if "coordinate_loss" in loss_dict:
        metrics.coordinate_loss = loss_dict["coordinate_loss"].item()
    if "language_loss" in loss_dict:
        metrics.language_loss = loss_dict["language_loss"].item()
    
    return metrics


def log_training_metrics(metrics: TrainingMetrics, phase: str = "train"):
    """Log training metrics."""
    logger.info(f"📊 {phase.upper()} METRICS - Step {metrics.step}, Epoch {metrics.epoch}")
    logger.info(f"  Total Loss: {metrics.total_loss:.6f}")
    if metrics.teacher_loss > 0:
        logger.info(f"  Teacher Loss: {metrics.teacher_loss:.6f}")
    if metrics.student_loss > 0:
        logger.info(f"  Student Loss: {metrics.student_loss:.6f}")
    if metrics.coordinate_loss > 0:
        logger.info(f"  Coordinate Loss: {metrics.coordinate_loss:.6f}")
    if metrics.language_loss > 0:
        logger.info(f"  Language Loss: {metrics.language_loss:.6f}")
    logger.info(f"  Learning Rate: {metrics.learning_rate:.2e}")
    if metrics.gradient_norm > 0:
        logger.info(f"  Gradient Norm: {metrics.gradient_norm:.6f}")


# ==============================================================================
# Export All Functions and Classes
# ==============================================================================

__all__ = [
    # Constants
    "IGNORE_INDEX",
    # Types
    "LossDictType",
    # Schema classes
    "ChatProcessorOutput",
    "CollatedBatch",
    "TrainingMetrics",
    # Helper functions
    "debug_input_shapes",
    "prepare_inputs_for_forward",
    "prepare_inputs_for_generate",
    "validate_attention_mask_consistency",
    "fix_attention_mask_mismatch",
    "safe_prepare_inputs",
    # Validation functions
    "assert_tensor_shape",
    "assert_chat_processor_output", 
    "assert_collated_batch",
    "assert_model_inputs",
    "assert_model_output",
    # Memory utilities
    "get_tensor_memory_usage",
    "log_batch_memory_usage",
    "optimize_batch_for_memory",
    # Metrics functions
    "create_training_metrics",
    "log_training_metrics",
]