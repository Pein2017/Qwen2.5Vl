"""
Dual-Loss Management for Qwen2.5-VL BBU Detection Training

This module implements the dual-loss training architecture that combines:
- **LLM Loss**: Standard cross-entropy loss for language modeling
- **Coordinate Loss**: Soft expectation + L1 loss for coordinate token regression

Key Components:
- LossComponents: Structure for dual-loss tracking (LLM + coordinate)
- ModelOutput: Model output with loss components
- LossManager: Manager for computing and combining dual-loss components
- Dual-Mask System: Separate masks for LLM vs coordinate token positions

The system uses soft expectation coordinate loss instead of cross-entropy for
better coordinate regression performance in the BBU detection pipeline.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src_new.models.coordinate_loss import SoftExpectationCoordinateLoss  # noqa


def get_loss_logger() -> logging.Logger:
    """Get logger for loss management."""
    from src_new.config.config import _CONFIGURED_LOGGERS, _GLOBAL_LOG_LEVEL

    logger = logging.getLogger("loss_manager")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_GLOBAL_LOG_LEVEL)
        _CONFIGURED_LOGGERS.add("loss_manager")
    return logger


logger = get_loss_logger()


@dataclass
class LossComponents:
    """Multi-component loss structure for tracking."""

    loss: torch.Tensor  # Main loss for trainer
    llm_loss: Optional[torch.Tensor] = None
    coordinate_loss: Optional[torch.Tensor] = None
    teacher_loss: Optional[torch.Tensor] = None
    student_loss: Optional[torch.Tensor] = None

    # Granular teacher-student loss components
    teacher_llm_loss: Optional[torch.Tensor] = None
    teacher_l1_loss: Optional[torch.Tensor] = None
    student_llm_loss: Optional[torch.Tensor] = None
    student_l1_loss: Optional[torch.Tensor] = None


class ModelOutput:
    """Model output with loss components that's compatible with HuggingFace trainer and DataParallel."""

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        loss_components: Optional[LossComponents] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Store as attributes for direct access
        self.loss = loss
        self.logits = logits
        self.loss_components = loss_components
        self.hidden_states = hidden_states

        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        """Make the output subscriptable for HuggingFace trainer compatibility."""
        if key == "loss":
            return self.loss
        elif key == "logits":
            return self.logits
        elif key == "hidden_states":
            return self.hidden_states
        elif key == 0:  # For tuple-like access
            return self.loss
        else:
            return getattr(self, key, None)

    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)

    def keys(self):
        """Return available keys."""
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]


class LossManager:
    """
    Manager for computing and combining loss components.

    Handles coordinate token loss, teacher-student loss, and combines
    them with appropriate weights.
    """

    def __init__(
        self,
        config,
        token_processor,
        tokenizer,
    ) -> None:
        """
        Initialize loss manager.

        Args:
            config: Configuration object with loss weights
            token_processor: TokenProcessor for coordinate token ID mapping
            tokenizer: Extended tokenizer with coordinate tokens
        """
        # Extract configuration parameters
        self.coordinate_loss_weight = getattr(config, "coordinate_loss_weight", 0.05)
        self.regular_loss_weight = getattr(config, "regular_loss_weight", 1.0)
        self.teacher_loss_weight = getattr(config, "teacher_loss_weight", 0.3)
        self.student_loss_weight = getattr(config, "student_loss_weight", 1.0)

        self.last_loss_components = None

        # Initialize soft expectation coordinate loss function with token processor
        from .coordinate_loss import create_coordinate_loss_from_token_processor

        self.coordinate_loss_fn = create_coordinate_loss_from_token_processor(
            token_processor=token_processor,
            tokenizer=tokenizer,
            temperature=1.0,
            numerical_stability=True,
        )

    def compute_loss_components(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
        teacher_spans: Optional[List[List[Tuple[int, int]]]] = None,
        student_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> LossComponents:
        """
        Compute all loss components with granular teacher-student breakdown.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]
            teacher_spans: List of teacher spans per batch item
            student_spans: List of student spans per batch item

        Returns:
            Loss components with granular teacher-student breakdown
        """
        # Compute regular LLM loss (fallback)
        total_llm_loss = self._compute_llm_loss(logits, labels)
        total_coordinate_loss = None

        # Compute coordinate loss if mask provided
        if coord_mask is not None and coord_mask.any():
            total_coordinate_loss = self._compute_coordinate_loss(
                logits, labels, coord_mask
            )

        # Initialize loss components
        loss_components = LossComponents(
            loss=total_llm_loss,
            llm_loss=total_llm_loss,
            coordinate_loss=total_coordinate_loss,
        )

        # Compute granular teacher-student loss breakdown
        has_teacher_spans = teacher_spans and any(spans for spans in teacher_spans)
        has_student_spans = student_spans and any(spans for spans in student_spans)

        if has_teacher_spans or has_student_spans:
            # Compute granular loss components
            granular_losses = self._compute_granular_teacher_student_loss(
                logits, labels, coord_mask, teacher_spans, student_spans
            )

            # Update loss components with granular breakdown
            loss_components.teacher_llm_loss = granular_losses["teacher_llm_loss"]
            loss_components.teacher_l1_loss = granular_losses["teacher_l1_loss"]
            loss_components.student_llm_loss = granular_losses["student_llm_loss"]
            loss_components.student_l1_loss = granular_losses["student_l1_loss"]

            # Compute aggregated teacher and student losses
            teacher_loss = self._aggregate_teacher_loss(
                granular_losses["teacher_llm_loss"], granular_losses["teacher_l1_loss"]
            )
            student_loss = self._aggregate_student_loss(
                granular_losses["student_llm_loss"], granular_losses["student_l1_loss"]
            )

            loss_components.teacher_loss = teacher_loss
            loss_components.student_loss = student_loss

            # Compute total loss as sum of teacher and student losses
            if teacher_loss is not None and student_loss is not None:
                loss_components.loss = teacher_loss + student_loss
            elif teacher_loss is not None:
                loss_components.loss = teacher_loss
            elif student_loss is not None:
                loss_components.loss = student_loss

        # Store loss components for later retrieval
        self.last_loss_components = loss_components

        return loss_components

    def _compute_llm_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regular LLM loss.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]

        Returns:
            LLM loss tensor
        """
        # Get loss function
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Reshape logits and labels
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute loss
        return loss_fct(logits_flat, labels_flat)

    def _compute_coordinate_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, coord_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coordinate token loss using soft expectation + L1 loss.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]

        Returns:
            Coordinate loss tensor
        """
        # Use soft expectation coordinate loss function
        coordinate_loss, loss_info = self.coordinate_loss_fn(
            logits=logits, labels=labels, coord_mask=coord_mask
        )

        # Log detailed coordinate loss information
        if loss_info["num_coord_tokens"] > 0:
            logger.debug(
                f"🎯 Coordinate loss details: "
                f"L1={loss_info['coordinate_l1_loss']:.6f}, "
                f"tokens={loss_info['num_coord_tokens']}, "
                f"expected_avg={loss_info['mean_expected_coord']:.2f}, "
                f"target_avg={loss_info['mean_target_coord']:.2f}"
            )

        return coordinate_loss

    def _compute_teacher_student_loss(
        self,
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

        # Compute teacher loss
        teacher_loss = None
        if teacher_mask.any():
            teacher_loss = self._compute_masked_loss(logits, labels, teacher_mask)

        # Compute student loss
        student_loss = None
        if student_mask.any():
            student_loss = self._compute_masked_loss(logits, labels, student_mask)

        return teacher_loss, student_loss

    def _compute_granular_teacher_student_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: Optional[torch.Tensor],
        teacher_spans: Optional[List[List[Tuple[int, int]]]],
        student_spans: Optional[List[List[Tuple[int, int]]]],
    ) -> dict:
        """
        Compute granular teacher-student loss breakdown.

        Returns:
            Dictionary with teacher_llm_loss, teacher_l1_loss, student_llm_loss, student_l1_loss
        """
        batch_size, seq_len, _ = logits.shape

        # Create masks for teacher and student tokens
        teacher_mask = torch.zeros_like(labels, dtype=torch.bool)
        student_mask = torch.zeros_like(labels, dtype=torch.bool)

        # Fill teacher mask
        if teacher_spans:
            for i in range(min(batch_size, len(teacher_spans))):
                for start, end in teacher_spans[i]:
                    if 0 <= start < end <= seq_len:
                        teacher_mask[i, start:end] = True

        # Fill student mask
        if student_spans:
            for i in range(min(batch_size, len(student_spans))):
                for start, end in student_spans[i]:
                    if 0 <= start < end <= seq_len:
                        student_mask[i, start:end] = True

        # Compute teacher LLM loss
        teacher_llm_loss = None
        if teacher_mask.any():
            teacher_llm_loss = self._compute_masked_loss(logits, labels, teacher_mask)

        # Compute student LLM loss
        student_llm_loss = None
        if student_mask.any():
            student_llm_loss = self._compute_masked_loss(logits, labels, student_mask)

        # Compute coordinate losses if coordinate mask provided
        teacher_l1_loss = None
        student_l1_loss = None

        if coord_mask is not None and coord_mask.any():
            # Teacher coordinate loss
            teacher_coord_mask = teacher_mask & coord_mask
            if teacher_coord_mask.any():
                teacher_l1_loss = self._compute_coordinate_loss(
                    logits, labels, teacher_coord_mask
                )

            # Student coordinate loss
            student_coord_mask = student_mask & coord_mask
            if student_coord_mask.any():
                student_l1_loss = self._compute_coordinate_loss(
                    logits, labels, student_coord_mask
                )

        return {
            "teacher_llm_loss": teacher_llm_loss,
            "teacher_l1_loss": teacher_l1_loss,
            "student_llm_loss": student_llm_loss,
            "student_l1_loss": student_l1_loss,
        }

    def _aggregate_teacher_loss(
        self,
        teacher_llm_loss: Optional[torch.Tensor],
        teacher_l1_loss: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Aggregate teacher loss components."""
        if teacher_llm_loss is None and teacher_l1_loss is None:
            return None

        total_loss = torch.tensor(
            0.0,
            device=teacher_llm_loss.device
            if teacher_llm_loss is not None
            else teacher_l1_loss.device,
        )

        if teacher_llm_loss is not None:
            total_loss += self.regular_loss_weight * teacher_llm_loss

        if teacher_l1_loss is not None:
            total_loss += self.coordinate_loss_weight * teacher_l1_loss

        return total_loss

    def _aggregate_student_loss(
        self,
        student_llm_loss: Optional[torch.Tensor],
        student_l1_loss: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Aggregate student loss components."""
        if student_llm_loss is None and student_l1_loss is None:
            return None

        total_loss = torch.tensor(
            0.0,
            device=student_llm_loss.device
            if student_llm_loss is not None
            else student_l1_loss.device,
        )

        if student_llm_loss is not None:
            total_loss += self.regular_loss_weight * student_llm_loss

        if student_l1_loss is not None:
            total_loss += self.coordinate_loss_weight * student_l1_loss

        return total_loss

    def _compute_masked_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with mask.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            mask: Boolean mask [batch_size, seq_len]

        Returns:
            Masked loss tensor
        """
        # Get loss function with no reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        # Reshape logits and labels
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute per-token loss
        per_token_loss = loss_fct(logits_flat, labels_flat)

        # Apply mask
        mask_flat = mask.view(-1)
        masked_loss = per_token_loss * mask_flat.float()

        # Get mean loss for masked tokens
        if mask_flat.sum() > 0:
            return masked_loss.sum() / mask_flat.sum()
        else:
            return torch.tensor(0.0, device=logits.device)
