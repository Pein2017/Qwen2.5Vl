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
import os


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
        # Debug controls
        self._debug_mode: Optional[str] = None  # 'train' | 'eval'
        self._debug_dumped_train: bool = False
        self._debug_dumped_eval: bool = False
        self._debug_detailed: bool = os.getenv("BBU_DEBUG_DETAILED", "0").strip() not in ("", "0", "false", "False")

    def set_debug_mode(self, mode: str) -> None:
        self._debug_mode = str(mode).lower()

    def _maybe_dump_detailed(self,
                              labels: torch.Tensor,
                              teacher_spans: Optional[List[List[Tuple[int, int]]]],
                              student_spans: Optional[List[List[Tuple[int, int]]]],
                              gm) -> None:
        if not self._debug_detailed:
            return
        mode = self._debug_mode or "unknown"
        if mode == "train" and self._debug_dumped_train:
            return
        if mode == "eval" and self._debug_dumped_eval:
            return
        try:
            # Only row 0
            b = 0
            input_ids = getattr(self, "_last_input_ids", None)
            if input_ids is None or not isinstance(input_ids, torch.Tensor):
                logger.info("[DetailedDebug] input_ids unavailable; skipping detailed dump")
                return
            row_ids = input_ids[b]
            ids_list = [int(x) for x in row_ids.tolist()]
            text = self._tokenizer.decode(ids_list, skip_special_tokens=False)
            tok = self._tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False,
            )
            offsets = tok["offset_mapping"][0].tolist()
            # Build assistant masks (unshifted)
            seq_len = labels.size(1)
            tmask = torch.zeros(seq_len, dtype=torch.bool)
            smask = torch.zeros(seq_len, dtype=torch.bool)
            if teacher_spans and len(teacher_spans) > b:
                for st, ed in teacher_spans[b]:
                    tmask[st:ed] = True
            if student_spans and len(student_spans) > b:
                for st, ed in student_spans[b]:
                    smask[st:ed] = True
            # Group masks (shifted) row 0
            def _to_set(mask: torch.Tensor) -> set:
                return {i for i, v in enumerate(mask[0].tolist()) if v}
            s_cap = _to_set(gm.student_caption)
            s_grd = _to_set(gm.student_grounding)
            s_fmt = _to_set(gm.student_formatting)
            t_cap = _to_set(gm.teacher_caption)
            t_grd = _to_set(gm.teacher_grounding)
            t_fmt = _to_set(gm.teacher_formatting)

            # Dump header
            logger.info("[DetailedDebug:%s] assistant spans (tokens): teacher=%s student=%s",
                        mode,
                        list(teacher_spans[b]) if teacher_spans and len(teacher_spans) > b else [],
                        list(student_spans[b]) if student_spans and len(student_spans) > b else [])
            logger.info("[DetailedDebug:%s] decoded (truncated 800): %s", mode, text[:800].replace("\n", " ⏎ "))

            # Iterate tokens in assistant regions only
            def _tag_for(idx: int, is_teacher: bool) -> str:
                # idx is unshifted; groups are shifted by 1
                sidx = idx - 1
                if sidx < 0:
                    return "-"
                if is_teacher:
                    if sidx in t_cap:
                        return "caption"
                    if sidx in t_grd:
                        return "grounding"
                    if sidx in t_fmt:
                        return "formatting"
                else:
                    if sidx in s_cap:
                        return "caption"
                    if sidx in s_grd:
                        return "grounding"
                    if sidx in s_fmt:
                        return "formatting"
                return "-"

            for idx in range(min(seq_len, len(offsets))):
                is_t = bool(tmask[idx])
                is_s = bool(smask[idx])
                if not (is_t or is_s):
                    continue
                ch0, ch1 = offsets[idx]
                token_str = self._tokenizer.convert_ids_to_tokens(ids_list[idx])
                tag = _tag_for(idx, is_t)
                role = "teacher" if is_t else "student"
                snippet = text[ch0:ch1].replace("\n", " ⏎ ")
                logger.info("[DetailedDebug:%s] i=%d [%d:%d] role=%s tag=%s tok=%s text='%s'",
                            mode, idx, ch0, ch1, role, tag, token_str, snippet)

            if mode == "train":
                self._debug_dumped_train = True
            elif mode == "eval":
                self._debug_dumped_eval = True
        except Exception as e:
            logger.info(f"[DetailedDebug] failed: {e}")

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
        conversation_variant: Optional[object] = None,
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
                logits, labels, coord_mask, teacher_spans, student_spans, conversation_variant
            )

            # Apply weights to get final loss components
            teacher_llm_weighted = None
            student_llm_weighted = None

            # Weight teacher LLM loss
            if granular_losses["teacher_llm_loss"] is not None:
                teacher_llm_weighted = (
                    self.teacher_loss_weight * granular_losses["teacher_llm_loss"]
                )
            if granular_losses["student_llm_loss"] is not None:
                student_llm_weighted = (
                    self.student_loss_weight * granular_losses["student_llm_loss"]
                )

            total_loss = torch.tensor(0.0, device=logits.device)
            if teacher_llm_weighted is not None:
                total_loss += teacher_llm_weighted
            if student_llm_weighted is not None:
                total_loss += student_llm_weighted

            loss_components = LossComponents(
                loss=total_loss,
                teacher_llm_loss=teacher_llm_weighted,
                student_llm_loss=student_llm_weighted,
            )
        else:
            # Fallback: compute standard LLM loss only (treat as student role)
            llm_loss = self._compute_llm_loss(logits, labels)
            total_loss = llm_loss
            loss_components = LossComponents(
                loss=total_loss,
                student_llm_loss=llm_loss,
            )

        # Store loss components for later retrieval
        self.last_loss_components = loss_components
        return loss_components

    def _compute_granular_teacher_student_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        coord_mask: Optional[torch.Tensor],
        teacher_spans: Optional[List[List[Tuple[int, int]]]],
        student_spans: Optional[List[List[Tuple[int, int]]]],
        conversation_variant: Optional[object] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute per-role loss breakdown using single-pass CE with optional grouping."""

        del coord_mask  # Coordinate losses are disabled in the refactored config system

        batch_size, seq_len, _ = logits.shape

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

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        per_token_loss = self._compute_per_token_cross_entropy(
            shifted_logits, shifted_labels
        )

        teacher_mask_shifted = teacher_mask[:, 1:]
        student_mask_shifted = student_mask[:, 1:]

        teacher_llm_loss: Optional[torch.Tensor] = None
        student_llm_loss: Optional[torch.Tensor] = None
        teacher_caption_loss: Optional[torch.Tensor] = None
        teacher_grounding_loss: Optional[torch.Tensor] = None
        teacher_formatting_loss: Optional[torch.Tensor] = None
        student_caption_loss: Optional[torch.Tensor] = None
        student_grounding_loss: Optional[torch.Tensor] = None
        student_formatting_loss: Optional[torch.Tensor] = None

        if self._grouping_enabled:
            if self._token_grouping_plugin is None:
                from src_new.losses.token_grouping import TokenGroupingPlugin

                self._token_grouping_plugin = TokenGroupingPlugin(self._tokenizer)

            variant_key = None
            if conversation_variant is not None:
                variant_key = str(getattr(conversation_variant, "value", conversation_variant)).strip().lower()

            gm = self._token_grouping_plugin.build_group_masks(
                labels=labels,
                teacher_spans=teacher_spans,
                student_spans=student_spans,
                input_ids=getattr(self, "_last_input_ids", None),
                variant_key=variant_key,
            )

            try:
                self._maybe_dump_detailed(labels, teacher_spans, student_spans, gm)
            except Exception:
                pass

            t_cap_sum, t_cap_cnt = self._masked_sum_and_count(per_token_loss, gm.teacher_caption)
            t_grd_sum, t_grd_cnt = self._masked_sum_and_count(per_token_loss, gm.teacher_grounding)
            t_fmt_sum, t_fmt_cnt = self._masked_sum_and_count(per_token_loss, gm.teacher_formatting)
            s_cap_sum, s_cap_cnt = self._masked_sum_and_count(per_token_loss, gm.student_caption)
            s_grd_sum, s_grd_cnt = self._masked_sum_and_count(per_token_loss, gm.student_grounding)
            s_fmt_sum, s_fmt_cnt = self._masked_sum_and_count(per_token_loss, gm.student_formatting)

            t_cap_cnt_f = t_cap_cnt.float()
            t_grd_cnt_f = t_grd_cnt.float()
            t_fmt_cnt_f = t_fmt_cnt.float()
            s_cap_cnt_f = s_cap_cnt.float()
            s_grd_cnt_f = s_grd_cnt.float()
            s_fmt_cnt_f = s_fmt_cnt.float()

            t_total = t_cap_cnt_f + t_grd_cnt_f + t_fmt_cnt_f
            s_total = s_cap_cnt_f + s_grd_cnt_f + s_fmt_cnt_f

            if t_total > 0:
                t_den = (
                    self._weight_caption * t_cap_cnt_f
                    + self._weight_grounding * t_grd_cnt_f
                    + self._weight_formatting * t_fmt_cnt_f
                )
                if float(t_den.item()) == 0.0:
                    raise ValueError(
                        "Grouped loss denominator is zero for teacher. Check group weights and masks."
                    )
                weighted_sum = (
                    self._weight_caption * t_cap_sum
                    + self._weight_grounding * t_grd_sum
                    + self._weight_formatting * t_fmt_sum
                )
                teacher_llm_loss = weighted_sum / t_den
                teacher_caption_loss = (
                    self._weight_caption * t_cap_sum
                ) / t_den
                teacher_grounding_loss = (
                    self._weight_grounding * t_grd_sum
                ) / t_den
                teacher_formatting_loss = (
                    self._weight_formatting * t_fmt_sum
                ) / t_den

            if s_total > 0:
                s_den = (
                    self._weight_caption * s_cap_cnt_f
                    + self._weight_grounding * s_grd_cnt_f
                    + self._weight_formatting * s_fmt_cnt_f
                )
                if float(s_den.item()) == 0.0:
                    raise ValueError(
                        "Grouped loss denominator is zero for student. Check group weights and masks."
                    )
                weighted_sum = (
                    self._weight_caption * s_cap_sum
                    + self._weight_grounding * s_grd_sum
                    + self._weight_formatting * s_fmt_sum
                )
                student_llm_loss = weighted_sum / s_den
                student_caption_loss = (
                    self._weight_caption * s_cap_sum
                ) / s_den
                student_grounding_loss = (
                    self._weight_grounding * s_grd_sum
                ) / s_den
                student_formatting_loss = (
                    self._weight_formatting * s_fmt_sum
                ) / s_den

            if (
                teacher_llm_loss is None
                and student_llm_loss is None
                and (teacher_mask_shifted.any() or student_mask_shifted.any())
            ):
                raise ValueError(
                    "Grouped masks are empty but assistant spans exist; token grouping configuration failed."
                )
        else:
            if teacher_mask_shifted.any():
                teacher_llm_loss = self._apply_mask_to_per_token_loss(
                    per_token_loss, teacher_mask_shifted
                )
            if student_mask_shifted.any():
                student_llm_loss = self._apply_mask_to_per_token_loss(
                    per_token_loss, student_mask_shifted
                )

        # Ensure fallback masks provide losses if grouping skipped a role
        if teacher_llm_loss is None and teacher_mask_shifted.any():
            teacher_llm_loss = self._apply_mask_to_per_token_loss(
                per_token_loss, teacher_mask_shifted
            )
        if student_llm_loss is None and student_mask_shifted.any():
            student_llm_loss = self._apply_mask_to_per_token_loss(
                per_token_loss, student_mask_shifted
            )

        per_sample_student_vec: Optional[torch.Tensor] = None
        if student_mask_shifted.any():
            try:
                mask_float = student_mask_shifted.float()
                per_sample_sum = (per_token_loss * mask_float).sum(dim=1)
                per_sample_cnt = mask_float.sum(dim=1)
                per_sample_vec = torch.where(
                    per_sample_cnt > 0,
                    per_sample_sum / torch.clamp(per_sample_cnt, min=1.0),
                    torch.zeros_like(per_sample_cnt),
                )
                per_sample_student_vec = per_sample_vec.detach().cpu()
            except Exception:
                per_sample_student_vec = None

        result: Dict[str, Optional[torch.Tensor]] = {
            "teacher_llm_loss": teacher_llm_loss,
            "student_llm_loss": student_llm_loss,
        }

        if teacher_caption_loss is not None:
            result["teacher_caption_loss"] = teacher_caption_loss
        if teacher_grounding_loss is not None:
            result["teacher_grounding_loss"] = teacher_grounding_loss
        if teacher_formatting_loss is not None:
            result["teacher_formatting_loss"] = teacher_formatting_loss
        if student_caption_loss is not None:
            result["student_caption_loss"] = student_caption_loss
        if student_grounding_loss is not None:
            result["student_grounding_loss"] = student_grounding_loss
        if student_formatting_loss is not None:
            result["student_formatting_loss"] = student_formatting_loss
        if per_sample_student_vec is not None:
            result["per_sample_student_llm_loss"] = per_sample_student_vec

        return result

    def _compute_llm_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard LLM loss with next-token alignment."""

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        vocab_size = shifted_logits.size(-1)
        logits_flat = shifted_logits.view(-1, vocab_size)
        labels_flat = shifted_labels.view(-1)

        return loss_fct(logits_flat, labels_flat)

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
