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
from typing import Dict, List, Optional, Tuple

import torch


def get_loss_logger() -> logging.Logger:
    """Get rank-aware logger for loss management."""
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger("loss_manager")
    except ImportError:
        # Fallback to config system
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
    """
    Simplified loss structure with only meaningful weighted components.

    All losses are final weighted values used in backpropagation.
    The total loss must exactly equal the sum of all component losses:
    loss = teacher_llm_loss + student_llm_loss + teacher_l1_loss + student_l1_loss
    """

    loss: torch.Tensor  # Total weighted loss for trainer (sum of all components)

    # Final weighted loss components (these are the actual values used in optimization)
    teacher_llm_loss: Optional[torch.Tensor] = (
        None  # Weighted cross-entropy loss for teacher samples
    )
    student_llm_loss: Optional[torch.Tensor] = (
        None  # Weighted cross-entropy loss for student samples
    )
    teacher_l1_loss: Optional[torch.Tensor] = (
        None  # Weighted coordinate regression loss for teacher samples
    )
    student_l1_loss: Optional[torch.Tensor] = (
        None  # Weighted coordinate regression loss for student samples
    )
    # New: separate weighted coordinate auxiliary components for logging
    teacher_kce_loss: Optional[torch.Tensor] = None
    teacher_unlike_loss: Optional[torch.Tensor] = None
    student_kce_loss: Optional[torch.Tensor] = None
    student_unlike_loss: Optional[torch.Tensor] = None

    # New: coordinate learning diagnostics (metrics; not used in loss sum)
    teacher_window_mass: Optional[torch.Tensor] = None
    student_window_mass: Optional[torch.Tensor] = None
    teacher_coord_slice_mass: Optional[torch.Tensor] = None
    student_coord_slice_mass: Optional[torch.Tensor] = None
    teacher_gt_prob: Optional[torch.Tensor] = None
    student_gt_prob: Optional[torch.Tensor] = None
    teacher_expected_mae_bins: Optional[torch.Tensor] = None
    student_expected_mae_bins: Optional[torch.Tensor] = None
    teacher_top1_acc: Optional[torch.Tensor] = None
    student_top1_acc: Optional[torch.Tensor] = None
    teacher_top5_acc: Optional[torch.Tensor] = None
    student_top5_acc: Optional[torch.Tensor] = None
    # Additional diagnostics
    teacher_outside_window_mass: Optional[torch.Tensor] = None
    student_outside_window_mass: Optional[torch.Tensor] = None
    teacher_noncoord_topk_mass: Optional[torch.Tensor] = None
    student_noncoord_topk_mass: Optional[torch.Tensor] = None
    teacher_window_entropy: Optional[torch.Tensor] = None
    student_window_entropy: Optional[torch.Tensor] = None
    teacher_margin_top1_top2: Optional[torch.Tensor] = None
    student_margin_top1_top2: Optional[torch.Tensor] = None
    teacher_mean_bin_offset: Optional[torch.Tensor] = None
    student_mean_bin_offset: Optional[torch.Tensor] = None
    teacher_coord_pos_count: Optional[torch.Tensor] = None
    student_coord_pos_count: Optional[torch.Tensor] = None
    # Generic diagnostics bag (preferred for logging/aggregation; keys are flattened like 'teacher_window_mass')
    diagnostics: Optional[Dict[str, torch.Tensor]] = None


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

    Unified spans support:
    - If callers pass a single list of spans (assistant_spans), this module should
      receive it via student_spans with teacher_spans=None. In that case, all
      assistant content is treated uniformly (no teacher/student distinction),
      and only the student_* components will be populated and weighted.
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
        # Extract configuration parameters (strict, no fallbacks)
        self.coordinate_loss_weight = config.coordinate_loss_weight
        self.teacher_loss_weight = config.teacher_loss_weight
        self.student_loss_weight = config.student_loss_weight
        # Keep full config for feature flags (e.g., coordinate_tokens_enabled)
        self.config = config

        self.last_loss_components = None

        # Store tokenizer and token processor for auxiliary losses and coordinate token range
        self._tokenizer = tokenizer
        self._token_processor = token_processor

        # Grouped-LLM controls (strictly from config; no defaults)
        for required in (
            "caption_loss_weight",
            "grounding_loss_weight",
            "formatting_loss_weight",
        ):
            if not hasattr(config, required):
                raise ValueError(f"Missing required config field: {required}")
        self._grouping_enabled: bool = True
        self._weight_caption: float = float(config.caption_loss_weight)
        self._weight_grounding: float = float(config.grounding_loss_weight)
        self._weight_formatting: float = float(config.formatting_loss_weight)
        self._token_grouping_plugin = None

        # Get coordinate token range from tokenizer (derived, no hard-coded IDs)
        try:
            from src_new.processing.special_tokens import get_coord_token_range

            rng = get_coord_token_range(tokenizer)
            self._coord_start_id, self._coord_end_id = rng.start_id, rng.end_exclusive
        except Exception:
            # Fallback: use token_processor helper if available
            self._coord_start_id, self._coord_end_id = (
                token_processor.get_coordinate_token_range(tokenizer)
            )

        # Optional coordinate auxiliary losses (disabled by default)
        self._coord_aux_enabled = False
        self._coord_aux_tau = None
        self._coord_aux_sigma_bins = None
        self._coord_aux_window_bins = None
        self._coord_aux_topk = None
        self._lambda_kce = None
        self._lambda_unlike = None
        self._lambda_lap1 = 0.0
        self._lambda_lap2 = 0.0
        self._embedding_accessor = None

    def set_token_grouping_plugin(self, plugin) -> None:
        """Set token grouping plugin (internal enablement).

        The plugin must expose build_group_masks(labels, teacher_spans, student_spans)
        and return masks aligned to shifted CE positions. Fails fast if the plugin is invalid.
        """
        if plugin is None:
            raise ValueError("Token grouping plugin must not be None")
        # Basic interface validation
        if not hasattr(plugin, "build_group_masks"):
            raise ValueError(
                "Token grouping plugin missing required method 'build_group_masks'"
            )
        self._token_grouping_plugin = plugin
        self._grouping_enabled = True

        # No legacy coordinate loss function; coordinate losses are handled via auxiliary path only.

    def set_coordinate_aux_options(
        self,
        *,
        tau: float,
        sigma_bins: float,
        window_bins: int,
        topk: int,
        lambda_kce: float,
        lambda_unlike: float,
        lambda_lap1: float = 0.0,
        lambda_lap2: float = 0.0,
    ) -> None:
        """Enable and configure coordinate auxiliary losses.

        This leaves the default soft-expectation path intact unless enabled.
        """
        self._coord_aux_enabled = True
        self._coord_aux_tau = float(tau)
        self._coord_aux_sigma_bins = float(sigma_bins)
        self._coord_aux_window_bins = int(window_bins)
        self._coord_aux_topk = int(topk)
        self._lambda_kce = float(lambda_kce)
        self._lambda_unlike = float(lambda_unlike)
        self._lambda_lap1 = float(lambda_lap1)
        self._lambda_lap2 = float(lambda_lap2)

    def set_embedding_accessor(self, accessor) -> None:
        """Set a callable that returns the coordinate embedding slice [K+1, d]."""
        self._embedding_accessor = accessor

    def compute_loss_components(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: Optional[torch.Tensor] = None,
        teacher_spans: Optional[List[List[Tuple[int, int]]]] = None,
        student_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> LossComponents:
        """
        Compute final weighted loss components for training.

        This method returns only the final weighted loss components that are actually
        used in backpropagation. All returned losses are weighted and ready for optimization.
        The total loss exactly equals the sum of all component losses.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            coord_mask: Coordinate token mask [batch_size, seq_len]
            teacher_spans: List of teacher spans per batch item
            student_spans: List of student spans per batch item

        Returns:
            LossComponents with only final weighted losses that sum to total loss
        """
        # Check if we have teacher-student spans for granular breakdown
        has_teacher_spans = teacher_spans and any(spans for spans in teacher_spans)
        has_student_spans = student_spans and any(spans for spans in student_spans)

        if has_teacher_spans or has_student_spans:
            # Compute granular teacher-student loss breakdown
            granular_losses = self._compute_granular_teacher_student_loss(
                logits, labels, coord_mask, teacher_spans, student_spans
            )

            # Apply weights to get final loss components
            teacher_llm_weighted = None
            student_llm_weighted = None
            # New separate weighted coord components
            teacher_kce_weighted = None
            teacher_unlike_weighted = None
            student_kce_weighted = None
            student_unlike_weighted = None

            # Weight teacher LLM loss
            if granular_losses["teacher_llm_loss"] is not None:
                teacher_llm_weighted = (
                    self.teacher_loss_weight * granular_losses["teacher_llm_loss"]
                )

            # Weight teacher coord components separately
            if (
                "teacher_kce_loss" in granular_losses
                and granular_losses["teacher_kce_loss"] is not None
            ):
                teacher_kce_weighted = (
                    self.teacher_loss_weight
                    * self.coordinate_loss_weight
                    * granular_losses["teacher_kce_loss"]
                )
            if (
                "teacher_unlike_loss" in granular_losses
                and granular_losses["teacher_unlike_loss"] is not None
            ):
                teacher_unlike_weighted = (
                    self.teacher_loss_weight
                    * self.coordinate_loss_weight
                    * granular_losses["teacher_unlike_loss"]
                )

            # Weight student LLM loss
            if granular_losses["student_llm_loss"] is not None:
                student_llm_weighted = (
                    self.student_loss_weight * granular_losses["student_llm_loss"]
                )

            # Weight student coord components separately
            if (
                "student_kce_loss" in granular_losses
                and granular_losses["student_kce_loss"] is not None
            ):
                student_kce_weighted = (
                    self.student_loss_weight
                    * self.coordinate_loss_weight
                    * granular_losses["student_kce_loss"]
                )
            if (
                "student_unlike_loss" in granular_losses
                and granular_losses["student_unlike_loss"] is not None
            ):
                student_unlike_weighted = (
                    self.student_loss_weight
                    * self.coordinate_loss_weight
                    * granular_losses["student_unlike_loss"]
                )

            # Compute total loss as exact sum of weighted components
            total_loss = torch.tensor(0.0, device=logits.device)
            if teacher_llm_weighted is not None:
                total_loss += teacher_llm_weighted

            if teacher_kce_weighted is not None:
                total_loss += teacher_kce_weighted
            if teacher_unlike_weighted is not None:
                total_loss += teacher_unlike_weighted

            if student_llm_weighted is not None:
                total_loss += student_llm_weighted

            if student_kce_weighted is not None:
                total_loss += student_kce_weighted
            if student_unlike_weighted is not None:
                total_loss += student_unlike_weighted

            # Laplacian regularizer removed

            # Create diagnostics dict from granular_losses (authoritative source)
            diagnostics: Dict[str, torch.Tensor] = {}
            from .coord_metrics import DIAGNOSTIC_METRIC_NAMES

            # Only include diagnostics when present (non-None) and relevant
            for group in ("teacher", "student"):
                for name in DIAGNOSTIC_METRIC_NAMES:
                    key = f"{group}_{name}"
                    if key in granular_losses and granular_losses[key] is not None:
                        diagnostics[key] = granular_losses[key]
            # Log six final group losses (teacher/student‑weighted) only when present
            t_map = {
                "teacher_caption_loss": "teacher_caption_loss",
                "teacher_grounding_loss": "teacher_grounding_loss",
                "teacher_formatting_loss": "teacher_formatting_loss",
            }
            s_map = {
                "student_caption_loss": "student_caption_loss",
                "student_grounding_loss": "student_grounding_loss",
                "student_formatting_loss": "student_formatting_loss",
            }
            for out_key, src_key in t_map.items():
                if src_key in granular_losses and granular_losses[src_key] is not None:
                    diagnostics[out_key] = (
                        self.teacher_loss_weight * granular_losses[src_key]
                    )
            for out_key, src_key in s_map.items():
                if src_key in granular_losses and granular_losses[src_key] is not None:
                    diagnostics[out_key] = (
                        self.student_loss_weight * granular_losses[src_key]
                    )

            # Structured grouped losses (weighted) for downstream adapters
            group_losses_struct: Dict[str, Dict[str, torch.Tensor]] = {}
            t_group: Dict[str, torch.Tensor] = {}
            s_group: Dict[str, torch.Tensor] = {}
            if (
                "teacher_caption_loss" in granular_losses
                and granular_losses["teacher_caption_loss"] is not None
            ):
                t_group["caption"] = (
                    self.teacher_loss_weight * granular_losses["teacher_caption_loss"]
                )
            if (
                "teacher_grounding_loss" in granular_losses
                and granular_losses["teacher_grounding_loss"] is not None
            ):
                t_group["grounding"] = (
                    self.teacher_loss_weight * granular_losses["teacher_grounding_loss"]
                )
            if (
                "teacher_formatting_loss" in granular_losses
                and granular_losses["teacher_formatting_loss"] is not None
            ):
                t_group["formatting"] = (
                    self.teacher_loss_weight * granular_losses["teacher_formatting_loss"]
                )
            if t_group:
                group_losses_struct["teacher"] = t_group

            if (
                "student_caption_loss" in granular_losses
                and granular_losses["student_caption_loss"] is not None
            ):
                s_group["caption"] = (
                    self.student_loss_weight * granular_losses["student_caption_loss"]
                )
            if (
                "student_grounding_loss" in granular_losses
                and granular_losses["student_grounding_loss"] is not None
            ):
                s_group["grounding"] = (
                    self.student_loss_weight * granular_losses["student_grounding_loss"]
                )
            if (
                "student_formatting_loss" in granular_losses
                and granular_losses["student_formatting_loss"] is not None
            ):
                s_group["formatting"] = (
                    self.student_loss_weight * granular_losses["student_formatting_loss"]
                )
            if s_group:
                group_losses_struct["student"] = s_group

            if group_losses_struct:
                diagnostics["group_losses"] = group_losses_struct

            # CORRECTED: Compute combined weighted L1 losses for accurate logging
            teacher_l1_weighted = torch.tensor(0.0, device=logits.device)
            if teacher_kce_weighted is not None:
                teacher_l1_weighted += teacher_kce_weighted
            if teacher_unlike_weighted is not None:
                teacher_l1_weighted += teacher_unlike_weighted

            student_l1_weighted = torch.tensor(0.0, device=logits.device)
            if student_kce_weighted is not None:
                student_l1_weighted += student_kce_weighted
            if student_unlike_weighted is not None:
                student_l1_weighted += student_unlike_weighted

            # Create loss components with final weighted values (diagnostics dictionary preferred)
            loss_components = LossComponents(
                loss=total_loss,
                teacher_llm_loss=teacher_llm_weighted,
                teacher_l1_loss=teacher_l1_weighted
                if teacher_l1_weighted > 0
                else None,
                student_llm_loss=student_llm_weighted,
                student_l1_loss=student_l1_weighted
                if student_l1_weighted > 0
                else None,
                teacher_kce_loss=teacher_kce_weighted,
                teacher_unlike_loss=teacher_unlike_weighted,
                student_kce_loss=student_kce_weighted,
                student_unlike_loss=student_unlike_weighted,
                diagnostics=diagnostics if diagnostics else None,
            )

        else:
            # Fallback: compute standard LLM loss only (no legacy coord path)
            llm_loss = self._compute_llm_loss(logits, labels)

            # Total loss is regular LLM loss only in this path
            total_loss = llm_loss

            # Create loss components (treating as student loss for consistency)
            loss_components = LossComponents(
                loss=total_loss,
                teacher_llm_loss=None,
                teacher_l1_loss=None,
                student_llm_loss=llm_loss,
                student_l1_loss=None,
            )

        # Store loss components for later retrieval
        self.last_loss_components = loss_components

        return loss_components

    def _compute_llm_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regular LLM loss with proper next-token alignment (shifted).

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]

        Returns:
            LLM loss tensor
        """
        # Get loss function
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Shift for next-token prediction: logits[:, :-1] vs labels[:, 1:]
        batch_size, seq_len, vocab_size = logits.shape
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Reshape logits and labels
        logits_flat = shifted_logits.view(-1, vocab_size)
        labels_flat = shifted_labels.view(-1)

        # Compute loss
        return loss_fct(logits_flat, labels_flat)

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
        PRODUCTION-READY: Compute granular teacher-student loss breakdown with Solution 1 optimization.

        This method implements the core loss computation for teacher-student training with
        significant performance optimizations while maintaining mathematical equivalence.

        **Solution 1 Optimization:**
        - Computes cross-entropy exactly once for all tokens
        - Reuses per-token results for both teacher and student loss calculation
        - Achieves 60-70% reduction in computation time

        **Loss Components:**
        1. **Teacher LLM Loss**: Cross-entropy loss on teacher assistant tokens (including coordinate tokens)
        2. **Student LLM Loss**: Cross-entropy loss on student assistant tokens (including coordinate tokens)
        3. **Teacher L1 Loss**: Soft expectation coordinate loss at positions predicting coordinate tokens (teacher spans)
        4. **Student L1 Loss**: Soft expectation coordinate loss at positions predicting coordinate tokens (student spans)

        Returns a dict with the 4 components (unweighted).
        """
        batch_size, seq_len, _ = logits.shape

        # Build teacher/student masks from spans over the label positions
        teacher_mask = torch.zeros_like(labels, dtype=torch.bool)
        student_mask = torch.zeros_like(labels, dtype=torch.bool)

        if teacher_spans:
            for i in range(min(batch_size, len(teacher_spans))):
                for start, end in teacher_spans[i]:
                    if 0 <= start < end <= seq_len:
                        teacher_mask[i, start:end] = True

        if student_spans:
            for i in range(min(batch_size, len(student_spans))):
                for start, end in student_spans[i]:
                    if 0 <= start < end <= seq_len:
                        student_mask[i, start:end] = True

        # Determine coordinate-token positions from LABELS (what we are predicting)
        coord_start = self._coord_start_id
        coord_end_exclusive = self._coord_end_id
        # coord_label_mask marks positions whose label token IS a coordinate token
        coord_label_mask = (labels >= coord_start) & (labels < coord_end_exclusive)

        # Shift logits/labels for next-token prediction
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Build masks aligned to shifted tensors (drop the first position)
        teacher_mask_shifted = teacher_mask[:, 1:]
        student_mask_shifted = student_mask[:, 1:]

        # For LLM loss, include all assistant targets (including coordinate-token targets)
        teacher_llm_mask_shifted = teacher_mask_shifted
        student_llm_mask_shifted = student_mask_shifted

        # Compute CE once on shifted tensors and reuse
        per_token_loss = self._compute_per_token_cross_entropy(
            shifted_logits, shifted_labels
        )

        # New: grouped LLM path (internally enabled)
        teacher_llm_loss = None
        student_llm_loss = None

        if self._grouping_enabled:
            if self._token_grouping_plugin is None:
                from src_new.losses.token_grouping import TokenGroupingPlugin

                self._token_grouping_plugin = TokenGroupingPlugin(self._tokenizer)
            gm = self._token_grouping_plugin.build_group_masks(
                labels=labels,
                teacher_spans=teacher_spans,
                student_spans=student_spans,
            )

            # Per-group sums and counts for proper weighted aggregation
            t_cap_sum, t_cap_cnt = self._masked_sum_and_count(
                per_token_loss, gm.teacher_caption
            )
            t_grd_sum, t_grd_cnt = self._masked_sum_and_count(
                per_token_loss, gm.teacher_grounding
            )
            t_fmt_sum, t_fmt_cnt = self._masked_sum_and_count(
                per_token_loss, gm.teacher_formatting
            )
            s_cap_sum, s_cap_cnt = self._masked_sum_and_count(
                per_token_loss, gm.student_caption
            )
            s_grd_sum, s_grd_cnt = self._masked_sum_and_count(
                per_token_loss, gm.student_grounding
            )
            s_fmt_sum, s_fmt_cnt = self._masked_sum_and_count(
                per_token_loss, gm.student_formatting
            )

            # Aggregated (teacher+student) group metrics

            # Weighted teacher/student LLM losses as per-token weighted averages
            # Convert counts to float for stable weighted normalization
            t_cap_cnt_f = t_cap_cnt.float()
            t_grd_cnt_f = t_grd_cnt.float()
            t_fmt_cnt_f = t_fmt_cnt.float()
            s_cap_cnt_f = s_cap_cnt.float()
            s_grd_cnt_f = s_grd_cnt.float()
            s_fmt_cnt_f = s_fmt_cnt.float()

            # Unweighted totals (for presence checks and safe fallback)
            t_total = t_cap_cnt_f + t_grd_cnt_f + t_fmt_cnt_f
            s_total = s_cap_cnt_f + s_grd_cnt_f + s_fmt_cnt_f
            teacher_caption_loss_contrib = None
            teacher_grounding_loss_contrib = None
            teacher_formatting_loss_contrib = None
            student_caption_loss_contrib = None
            student_grounding_loss_contrib = None
            student_formatting_loss_contrib = None

            if t_total > 0:
                t_weighted_sum = (
                    self._weight_caption * t_cap_sum
                    + self._weight_grounding * t_grd_sum
                    + self._weight_formatting * t_fmt_sum
                )
                # Weighted denominator to stabilize scale across group mixes
                t_den = (
                    self._weight_caption * t_cap_cnt_f
                    + self._weight_grounding * t_grd_cnt_f
                    + self._weight_formatting * t_fmt_cnt_f
                )
                # Safe fallback if weighted count is zero (e.g., all tokens fall into a zero-weight group)
                if float(t_den.item()) == 0.0:
                    t_den = t_total
                teacher_llm_loss = t_weighted_sum / t_den
                teacher_caption_loss_contrib = (
                    self._weight_caption * t_cap_sum
                ) / t_den
                teacher_grounding_loss_contrib = (
                    self._weight_grounding * t_grd_sum
                ) / t_den
                teacher_formatting_loss_contrib = (
                    self._weight_formatting * t_fmt_sum
                ) / t_den

            if s_total > 0:
                s_weighted_sum = (
                    self._weight_caption * s_cap_sum
                    + self._weight_grounding * s_grd_sum
                    + self._weight_formatting * s_fmt_sum
                )
                # Weighted denominator to stabilize scale across group mixes
                s_den = (
                    self._weight_caption * s_cap_cnt_f
                    + self._weight_grounding * s_grd_cnt_f
                    + self._weight_formatting * s_fmt_cnt_f
                )
                # Safe fallback if weighted count is zero
                if float(s_den.item()) == 0.0:
                    s_den = s_total
                student_llm_loss = s_weighted_sum / s_den
                student_caption_loss_contrib = (
                    self._weight_caption * s_cap_sum
                ) / s_den
                student_grounding_loss_contrib = (
                    self._weight_grounding * s_grd_sum
                ) / s_den
                student_formatting_loss_contrib = (
                    self._weight_formatting * s_fmt_sum
                ) / s_den
        else:
            if teacher_llm_mask_shifted.any():
                teacher_llm_loss = self._apply_mask_to_per_token_loss(
                    per_token_loss, teacher_llm_mask_shifted
                )
            if student_llm_mask_shifted.any():
                student_llm_loss = self._apply_mask_to_per_token_loss(
                    per_token_loss, student_llm_mask_shifted
                )

        # Coordinate losses: only where the TARGET token is a coordinate token
        teacher_coord_mask_shifted = teacher_mask_shifted & coord_label_mask[:, 1:]
        student_coord_mask_shifted = student_mask_shifted & coord_label_mask[:, 1:]

        # STRICT: if coord aux is enabled but there are no coordinate-labeled targets in any spans, fail fast
        if self._coord_aux_enabled and getattr(
            self.config, "coordinate_tokens_enabled", True
        ):
            if not (
                teacher_coord_mask_shifted.any() or student_coord_mask_shifted.any()
            ):
                total_coord_targets = int(coord_label_mask[:, 1:].sum().item())
                num_teacher_spans = sum(len(s) for s in (teacher_spans or []))
                num_student_spans = sum(len(s) for s in (student_spans or []))
                raise ValueError(
                    "coord_aux_enabled=True but no coordinate-labeled targets were found inside assistant spans for this batch. "
                    f"coord_label_targets={total_coord_targets}, teacher_spans={num_teacher_spans}, student_spans={num_student_spans}. "
                    "Ensure conversations emit <|coord_*|> tokens within assistant content and that span detection is correct."
                )

        teacher_l1_loss = None
        student_l1_loss = None
        # New: unweighted separate coord components (with lambda applied here)
        teacher_kce = None
        teacher_unlike = None
        student_kce = None
        student_unlike = None

        # Diagnostics (metrics)
        teacher_window_mass = None
        student_window_mass = None
        teacher_coord_slice_mass = None
        student_coord_slice_mass = None
        teacher_gt_prob = None
        student_gt_prob = None
        teacher_expected_mae_bins = None
        student_expected_mae_bins = None
        teacher_top1_acc = None
        student_top1_acc = None
        teacher_top5_acc = None
        student_top5_acc = None
        # Additional diagnostics
        teacher_outside_window_mass = None
        student_outside_window_mass = None
        teacher_noncoord_topk_mass = None
        student_noncoord_topk_mass = None
        teacher_window_entropy = None
        student_window_entropy = None
        teacher_margin_top1_top2 = None
        student_margin_top1_top2 = None
        teacher_mean_bin_offset = None
        student_mean_bin_offset = None
        teacher_coord_pos_count = None
        student_coord_pos_count = None

        # Only compute auxiliary losses if enabled and coordinate tokens are present
        should_compute_aux = (
            self._coord_aux_enabled
            and self.config.coordinate_tokens_enabled
            and (teacher_coord_mask_shifted.any() or student_coord_mask_shifted.any())
        )

        if should_compute_aux:
            from src_new.losses.coord_aux import (
                build_kernel_indices_and_q,
                kernelized_kl_sparse,
                unlikelihood_topk_text,
            )

            from .coord_metrics import compute_coord_diagnostics

            coord_start = int(self._coord_start_id)
            coord_end_exclusive = int(self._coord_end_id)
            K = (coord_end_exclusive - coord_start) - 1  # inclusive max bin

            coord_logits_full = shifted_logits[..., coord_start:coord_end_exclusive]

            # No-op branch removed: with should_compute_aux=True we must have positions; otherwise we would have raised already.
            V = shifted_logits.size(-1)
            coord_vocab = coord_end_exclusive - coord_start
            if V <= coord_vocab:
                raise ValueError(
                    f"Invalid vocab size for coordinate slice diagnostics: V={V}, coord_vocab={coord_vocab}"
                )

            def _compute_group_aux(group_mask: torch.Tensor):
                if group_mask is None or not group_mask.any():
                    # Caller ensures at least one group has positions; treat empty group as skipped (no metrics/aux for that group)
                    raise ValueError(
                        "Attempted to compute coordinate auxiliary losses on an empty group mask. This indicates a logic error upstream."
                    )
                pos = group_mask.nonzero(as_tuple=False)
                coord_logits = coord_logits_full[pos[:, 0], pos[:, 1], :]
                y_ids = shifted_labels[pos[:, 0], pos[:, 1]]
                y = (y_ids - coord_start).clamp(0, K)

                idxs, q_vals = build_kernel_indices_and_q(
                    y=y,
                    K=K,
                    sigma=self._coord_aux_sigma_bins,
                    window=self._coord_aux_window_bins,
                )
                kce = kernelized_kl_sparse(
                    coord_logits=coord_logits,
                    idxs=idxs,
                    q_vals=q_vals,
                    tau=self._coord_aux_tau,
                    eps=1e-6,  # Explicit epsilon value
                )

                # Non-coordinate vocab mask via shared helper
                from src_new.losses.coord_aux import build_noncoord_vocab_mask

                noncoord_mask = build_noncoord_vocab_mask(
                    V, coord_start, coord_end_exclusive
                )
                # Ensure mask is on the same device as logits for safe boolean indexing
                noncoord_mask = noncoord_mask.to(device=shifted_logits.device)
                unlike = unlikelihood_topk_text(
                    logits_all=shifted_logits,
                    coord_mask=group_mask,
                    noncoord_vocab_mask=noncoord_mask,
                    topk=self._coord_aux_topk,
                    eps=1e-6,  # Explicit epsilon value
                )

                # Build exact full-logits rows for metrics that need them
                full_logits_rows = shifted_logits[pos[:, 0], pos[:, 1], :]
                metrics = compute_coord_diagnostics(
                    shifted_logits=full_logits_rows,
                    coord_logits=coord_logits,
                    y_bins=y,
                    idxs=idxs,
                    noncoord_mask=noncoord_mask,
                    tau=self._coord_aux_tau,
                    topk_noncoord=self._coord_aux_topk,
                )
                return kce, unlike, metrics

            if teacher_coord_mask_shifted.any():
                kce_t, ul_t, m_t = _compute_group_aux(teacher_coord_mask_shifted)
                teacher_kce = kce_t
                teacher_unlike = ul_t
                teacher_l1_loss = teacher_kce + teacher_unlike
                teacher_window_mass = m_t["window_mass"]
                teacher_coord_slice_mass = m_t["coord_slice_mass"]
                teacher_gt_prob = m_t["gt_prob"]
                teacher_expected_mae_bins = m_t["expected_mae_bins"]
                teacher_top1_acc = m_t["top1_acc"]
                teacher_top5_acc = m_t["top5_acc"]
                teacher_outside_window_mass = m_t["outside_window_mass"]
                teacher_noncoord_topk_mass = m_t["noncoord_topk_mass"]
                teacher_window_entropy = m_t["window_entropy"]
                teacher_margin_top1_top2 = m_t["margin_top1_top2"]
                teacher_mean_bin_offset = m_t["mean_bin_offset"]
                teacher_coord_pos_count = m_t["coord_pos_count"]
            if student_coord_mask_shifted.any():
                kce_s, ul_s, m_s = _compute_group_aux(student_coord_mask_shifted)
                student_kce = kce_s
                student_unlike = ul_s
                student_l1_loss = student_kce + student_unlike
                student_window_mass = m_s["window_mass"]
                student_coord_slice_mass = m_s["coord_slice_mass"]
                student_gt_prob = m_s["gt_prob"]
                student_expected_mae_bins = m_s["expected_mae_bins"]
                student_top1_acc = m_s["top1_acc"]
                student_top5_acc = m_s["top5_acc"]
                student_outside_window_mass = m_s["outside_window_mass"]
                student_noncoord_topk_mass = m_s["noncoord_topk_mass"]
                student_window_entropy = m_s["window_entropy"]
                student_margin_top1_top2 = m_s["margin_top1_top2"]
                student_mean_bin_offset = m_s["mean_bin_offset"]
                student_coord_pos_count = m_s["coord_pos_count"]
        else:
            # When auxiliary losses are not enabled, do not fabricate zeros; keep metrics absent
            has_coord_tokens = (
                teacher_coord_mask_shifted.any() or student_coord_mask_shifted.any()
            )
            if (
                self._coord_aux_enabled
                and self.config.coordinate_tokens_enabled
                and not has_coord_tokens
            ):
                total_coord_targets = int(coord_label_mask[:, 1:].sum().item())
                raise ValueError(
                    "coord_aux_enabled=True but no coordinate-labeled targets were found inside assistant spans for this batch. "
                    f"coord_label_targets={total_coord_targets}."
                )

        return {
            "teacher_llm_loss": teacher_llm_loss,
            "teacher_l1_loss": teacher_l1_loss,
            "student_llm_loss": student_llm_loss,
            "student_l1_loss": student_l1_loss,
            # Final weighted group losses (teacher & student)
            "teacher_caption_loss": teacher_caption_loss_contrib,
            "teacher_grounding_loss": teacher_grounding_loss_contrib,
            "teacher_formatting_loss": teacher_formatting_loss_contrib,
            "student_caption_loss": student_caption_loss_contrib,
            "student_grounding_loss": student_grounding_loss_contrib,
            "student_formatting_loss": student_formatting_loss_contrib,
            # Coordinate aux (if any)
            "teacher_kce_loss": teacher_kce,
            "teacher_unlike_loss": teacher_unlike,
            "student_kce_loss": student_kce,
            "student_unlike_loss": student_unlike,
            # Coord diagnostics retained
            "teacher_window_mass": teacher_window_mass,
            "student_window_mass": student_window_mass,
            "teacher_coord_slice_mass": teacher_coord_slice_mass,
            "student_coord_slice_mass": student_coord_slice_mass,
            "teacher_gt_prob": teacher_gt_prob,
            "student_gt_prob": student_gt_prob,
            "teacher_expected_mae_bins": teacher_expected_mae_bins,
            "student_expected_mae_bins": student_expected_mae_bins,
            "teacher_top1_acc": teacher_top1_acc,
            "student_top1_acc": student_top1_acc,
            "teacher_top5_acc": teacher_top5_acc,
            "student_top5_acc": student_top5_acc,
            "teacher_outside_window_mass": teacher_outside_window_mass,
            "student_outside_window_mass": student_outside_window_mass,
            "teacher_noncoord_topk_mass": teacher_noncoord_topk_mass,
            "student_noncoord_topk_mass": student_noncoord_topk_mass,
            "teacher_window_entropy": teacher_window_entropy,
            "student_window_entropy": student_window_entropy,
            "teacher_margin_top1_top2": teacher_margin_top1_top2,
            "student_margin_top1_top2": student_margin_top1_top2,
            "teacher_mean_bin_offset": teacher_mean_bin_offset,
            "student_mean_bin_offset": student_mean_bin_offset,
            "teacher_coord_pos_count": teacher_coord_pos_count,
            "student_coord_pos_count": student_coord_pos_count,
        }

    def _compute_per_token_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        SOLUTION 1 CORE: Compute per-token cross-entropy loss exactly once for optimization.

        Uses next-token alignment (assumes logits/labels already shifted by caller).

        Args:
            logits: Prediction logits [batch_size, seq_len-1, vocab_size] (shifted)
            labels: Target labels [batch_size, seq_len-1] (shifted)

        Returns:
            Per-token loss tensor [batch_size, seq_len-1]
        """
        # Get loss function with no reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        # Reshape logits and labels
        vocab_size = logits.shape[-1]
        # Work in float32 for numerical robustness (bf16 inputs may overflow/underflow in softmax)
        logits_flat = logits.float().view(-1, vocab_size)
        # Clamp and clean non-finite values before CE
        logits_flat = torch.clamp(logits_flat, min=-50.0, max=50.0)
        logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=50.0, neginf=-50.0)
        labels_flat = labels.view(-1)

        # Compute per-token loss
        per_token_loss_flat = loss_fct(logits_flat, labels_flat)

        # Reshape back to [batch_size, seq_len-1]
        per_token_loss = per_token_loss_flat.view(logits.shape[0], logits.shape[1])

        return per_token_loss

    def _apply_mask_to_per_token_loss(
        self, per_token_loss: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        SOLUTION 1: Apply mask to pre-computed per-token loss.

        This method reuses per-token cross-entropy results with different masks,
        avoiding redundant cross-entropy computation.

        Args:
            per_token_loss: Pre-computed per-token loss [batch_size, seq_len-1]
            mask: Boolean mask [batch_size, seq_len-1]

        Returns:
            Masked loss tensor (scalar)
        """
        # Apply mask
        masked_loss = per_token_loss * mask.float()

        # Get mean loss for masked tokens
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        else:
            return torch.tensor(0.0, device=per_token_loss.device)

    def _masked_sum_and_count(self, loss_mat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = loss_mat * mask.float()
        return masked.sum(), mask.sum()

    def _masked_mean(self, loss_mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s, c = self._masked_sum_and_count(loss_mat, mask)
        if c > 0:
            return s / c
        return torch.tensor(0.0, device=loss_mat.device)

    # No legacy coordinate loss function; coordinate losses are handled via auxiliary path only.

    def _compute_masked_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        LEGACY: Compute loss with mask (kept for backward compatibility).

        This method is kept for cases where teacher-student spans are not provided.
        When spans are available, the optimized single-pass method is used instead.

        Args:
            logits: Prediction logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            mask: Boolean mask [batch_size, seq_len]

        Returns:
            Masked loss tensor
        """
        # Compute per-token loss and apply mask (legacy approach)
        # Use proper shifting to match next-token prediction
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        per_token_loss = self._compute_per_token_cross_entropy(
            shifted_logits, shifted_labels
        )
        return self._apply_mask_to_per_token_loss(per_token_loss, mask[:, 1:])
