"""
Training utilities for Qwen2.5-VL.

This module provides utilities for training Qwen2.5-VL models, including:
- Loss computation functions
- Optimizer creation
- Learning rate scheduling
- Gradient accumulation
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch.optim import Optimizer


if TYPE_CHECKING:
    from src_new.config.config import Config


def get_utils_logger() -> logging.Logger:
    """Get rank-aware logger for training utils."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("training_utils")
    except ImportError:
        # Fallback to standard logging
        logger = logging.getLogger("training_utils")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_utils_logger()


def compute_loss_with_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss with optional mask.

    Args:
        logits: Prediction logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        mask: Optional boolean mask [batch_size, seq_len]

    Returns:
        Loss tensor
    """
    # Get loss function
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Reshape logits and labels
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute per-token loss
    loss = loss_fct(logits_flat, labels_flat)

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.view(-1)
        loss = loss * mask_flat

    # Ignore padding tokens
    padding_mask = labels_flat != -100
    if padding_mask.sum() > 0:
        loss = loss[padding_mask].mean()
    else:
        loss = loss.mean()

    return loss


def compute_coordinate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coord_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute coordinate token loss.

    Args:
        logits: Prediction logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        coord_mask: Coordinate token mask [batch_size, seq_len]

    Returns:
        Coordinate loss tensor
    """
    # Compute loss only for coordinate tokens
    return compute_loss_with_mask(logits, labels, coord_mask)


def compute_teacher_student_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_spans: List[List[Tuple[int, int]]],
    student_spans: List[List[Tuple[int, int]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute teacher and student loss components.

    Args:
        logits: Prediction logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        teacher_spans: List of teacher spans per batch item
        student_spans: List of student spans per batch item

    Returns:
        Tuple of (teacher_loss, student_loss)
    """
    batch_size, seq_len, _ = logits.shape

    # Create masks for teacher and student tokens
    teacher_mask = torch.zeros_like(labels, dtype=torch.bool)
    student_mask = torch.zeros_like(labels, dtype=torch.bool)

    # Fill in masks from spans
    for i in range(min(batch_size, len(teacher_spans))):
        for start, end in teacher_spans[i]:
            if 0 <= start < end <= seq_len:
                teacher_mask[i, start:end] = True

    for i in range(min(batch_size, len(student_spans))):
        for start, end in student_spans[i]:
            if 0 <= start < end <= seq_len:
                student_mask[i, start:end] = True

    # Compute losses if masks are not empty
    teacher_loss = None
    if teacher_mask.any():
        teacher_loss = compute_loss_with_mask(logits, labels, teacher_mask)

    student_loss = None
    if student_mask.any():
        student_loss = compute_loss_with_mask(logits, labels, student_mask)

    return teacher_loss, student_loss


def get_learning_rate(optimizer: Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: Optimizer instance

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

    return 0.0


def prepare_model_for_training(
    model: torch.nn.Module, config: "Config"
) -> torch.nn.Module:
    """
    Prepare model for training by applying necessary configurations.

    Args:
        model: Model to prepare
        config: Configuration object

    Returns:
        Prepared model
    """
    # Enable gradient checkpointing if specified
    if getattr(config, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")

    # Set model to training mode
    model.train()

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ Model prepared for training")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Trainable parameters: {trainable_params:,}")
    logger.info(f"   - Trainable ratio: {trainable_params / total_params:.2%}")

    return model


def compute_memory_usage() -> dict:
    """
    Compute current GPU memory usage.

    Returns:
        Dictionary with memory usage information
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        memory_info[f"gpu_{i}"] = {"allocated_gb": allocated, "reserved_gb": reserved}

    return memory_info


def apply_gradient_scaling(
    loss: torch.Tensor, scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> torch.Tensor:
    """
    Apply gradient scaling for mixed precision training.

    Args:
        loss: Loss tensor
        scaler: Optional gradient scaler

    Returns:
        Scaled loss tensor
    """
    if scaler is not None:
        return scaler.scale(loss)
    return loss
