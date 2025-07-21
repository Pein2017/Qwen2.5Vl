"""
Simplified Loss Manager for BBU Training

Streamlined loss computation without excessive validation and accumulation complexity.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger


class LossManager:
    """
    Simplified manager for loss computation in BBU training.

    Delegates most loss computation to the model wrapper and focuses
    on clean loss extraction and logging.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        """Initialize simplified loss manager."""
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        self.tokenizer = tokenizer
        self.logger = get_training_logger()
        self._micro_batch_count: int = 0
        self._current_losses: Dict[str, float] = {}

    def compute_total_loss(
        self,
        model_outputs: Any,
        inputs: Dict[str, Any],
        is_training: bool = True,
        detection_training_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Simplified loss computation.

        Args:
            model_outputs: Output from model forward pass
            inputs: Batch inputs containing labels and ground truth
            is_training: Whether in training mode
            detection_training_enabled: Whether detection training is active

        Returns:
            Tuple of (total_loss_tensor, loss_components_dict)
        """
        if is_training:
            self._micro_batch_count += 1

        # Extract loss from model outputs (model handles coordinate/standard loss internally)
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            total_loss = model_outputs.loss
            self.logger.debug(
                f"🔍 TRAINING LOSS CHECK: model_outputs.loss = {total_loss.item():.6f}"
            )
            if total_loss.item() == 0.0:
                self.logger.error(
                    f"🚨 MODEL OUTPUT LOSS IS 0.0! This means the model is not computing loss correctly!"
                )
                self.logger.error(f"   - model_outputs type: {type(model_outputs)}")
                self.logger.error(
                    f"   - hasattr(loss): {hasattr(model_outputs, 'loss')}"
                )
                self.logger.error(f"   - loss value: {model_outputs.loss}")
                self.logger.error(
                    f"   - loss shape: {model_outputs.loss.shape if hasattr(model_outputs.loss, 'shape') else 'no shape'}"
                )
                raise RuntimeError(
                    f"Model is returning 0.0 loss! Check model forward pass."
                )
        else:
            # EXPOSE ERROR: This should never happen in training!
            labels = inputs.get("labels")
            self.logger.error(f"🚨 LOSS COMPUTATION FALLBACK TRIGGERED!")
            self.logger.error(
                f"   - model_outputs.loss is None: {not hasattr(model_outputs, 'loss') or model_outputs.loss is None}"
            )
            self.logger.error(f"   - labels present: {labels is not None}")
            if labels is not None:
                self.logger.error(f"   - labels shape: {labels.shape}")
                total_loss = F.cross_entropy(
                    model_outputs.logits.view(-1, model_outputs.logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                self.logger.error(
                    f"   - Computed fallback loss: {total_loss.item():.6f}"
                )
            else:
                self.logger.error(
                    f"   - NO LABELS FOUND - Setting loss to 0.0 (THIS IS WRONG!)"
                )
                total_loss = torch.tensor(0.0, device=model_outputs.logits.device)
                raise RuntimeError(
                    "Training loss is 0.0 because no labels found in inputs! This means data loading is broken."
                )

        # Clean loss separation: LLM loss vs Coordinate loss components
        loss_components = {}

        # 1. LLM Loss (standard shifted cross entropy from Qwen2.5)
        llm_loss = (
            self._safe_item(model_outputs._regular_loss)
            if hasattr(model_outputs, "_regular_loss")
            else self._safe_item(model_outputs.loss)
        )
        loss_components["llm_loss"] = llm_loss

        # 2. Individual Coordinate Loss Components (no duplicates or summations)
        focal_loss = (
            self._safe_item(model_outputs._focal_loss)
            if hasattr(model_outputs, "_focal_loss")
            else 0.0
        )
        l1_loss = (
            self._safe_item(model_outputs._l1_loss)
            if hasattr(model_outputs, "_l1_loss")
            else 0.0
        )
        giou_loss = (
            self._safe_item(model_outputs._giou_loss)
            if hasattr(model_outputs, "_giou_loss")
            else 0.0
        )

        loss_components["focal_loss"] = focal_loss
        loss_components["l1_loss"] = l1_loss
        loss_components["giou_loss"] = giou_loss

        # Compute total coordinate loss for teacher-student differentiation
        coord_loss_total = focal_loss + l1_loss + giou_loss

        # 3. Teacher-Student differentiation (use span-based approach)
        teacher_lm_loss, student_lm_loss = self._compute_span_based_losses(
            inputs, llm_loss, coord_loss_total
        )
        loss_components["teacher_lm_loss"] = teacher_lm_loss
        loss_components["student_lm_loss"] = student_lm_loss

        # Remove coordinate_loss duplication - it's same as l1_loss
        # Don't add coordinate_loss separately to avoid confusion
        # objectness_loss removed - not needed for coordinate token system

        # Store current losses
        self._current_losses = loss_components.copy()

        return total_loss, loss_components

    def reset_loss_accumulation(self):
        """Reset loss accumulation for new gradient accumulation cycle."""
        self._micro_batch_count = 0

    def get_averaged_losses(self) -> Dict[str, float]:
        """Get averaged losses for logging."""
        # Return current losses instead of empty dict to support coordinator
        return self._current_losses.copy()

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "micro_batch_count": self._micro_batch_count,
        }

    def get_current_losses(self) -> Dict[str, float]:
        """Get current loss components."""
        return self._current_losses.copy()

    def save_training_state(self) -> Dict[str, Any]:
        """Save training state for evaluation."""
        return {
            "micro_batch_count": self._micro_batch_count,
            "current_losses": self._current_losses.copy(),
        }

    def restore_training_state(self, state: Dict[str, Any]):
        """Restore training state after evaluation."""
        self._micro_batch_count = state["micro_batch_count"]
        self._current_losses = state["current_losses"]

    def _safe_item(self, value) -> float:
        """Extract scalar value from tensor or float - FAIL FAST on unexpected types."""
        if hasattr(value, "item"):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(
                f"Expected tensor or numeric value, got {type(value)}: {value}"
            )

    def _compute_span_based_losses(
        self, inputs: Dict[str, Any], total_llm_loss: float, coord_loss_total: float
    ) -> Tuple[float, float]:
        """
        Compute teacher and student losses based on assistant spans within the same sequence.
        
        Args:
            inputs: Batch inputs containing span information
            total_llm_loss: Total LLM loss for the sequence
            coord_loss_total: Total coordinate loss for the sequence
            
        Returns:
            Tuple of (teacher_lm_loss, student_lm_loss)
        """
        teacher_spans = inputs.get("teacher_assistant_spans", [])
        student_spans = inputs.get("student_assistant_spans", [])
        
        # Debug: Log span summary (detailed debug removed since structure is now understood)
        self.logger.debug(f"🔍 SPAN SUMMARY: {len(teacher_spans)} teacher groups, {len(student_spans)} student groups")
        
        # If no span information is available, fall back to simple heuristic
        if not teacher_spans and not student_spans:
            self.logger.debug("No span information available, using fallback logic")
            # Assume this is a student sample if no explicit span information
            return 0.0, total_llm_loss + coord_loss_total
        
        # Calculate total number of tokens in teacher vs student spans
        # Handle the actual nested structure: [[[start, end], [start, end]], ...]
        def _flatten_spans(span_groups):
            """Flatten nested span groups into individual [start, end] pairs."""
            flattened = []
            for group in span_groups:
                if isinstance(group, (list, tuple)):
                    for span in group:
                        if isinstance(span, (list, tuple)) and len(span) == 2:
                            flattened.append(span)
                        else:
                            self.logger.warning(f"Unexpected span format in group: {span}")
                else:
                    self.logger.warning(f"Unexpected group format: {group}")
            return flattened
        
        def _extract_span_length(span):
            """Extract span length from [start, end] pair."""
            if isinstance(span, (tuple, list)) and len(span) == 2:
                try:
                    start, end = span[0], span[1]
                    # Convert to int if they're tensors or other numeric types
                    if hasattr(start, 'item'):
                        start = start.item()
                    if hasattr(end, 'item'):
                        end = end.item()
                    return max(0, int(end) - int(start))  # Ensure non-negative
                except Exception as e:
                    self.logger.warning(f"Error extracting span length from {span}: {e}")
                    return 0
            else:
                self.logger.warning(f"Invalid span format: {span} (type: {type(span)}), using length 0")
                return 0
        
        try:
            # Flatten the nested structures
            flat_teacher_spans = _flatten_spans(teacher_spans)
            flat_student_spans = _flatten_spans(student_spans)
            
            self.logger.debug(f"🔍 FLATTENED SPANS: teacher={len(flat_teacher_spans)} spans, student={len(flat_student_spans)} spans")
            self.logger.debug(f"🔍 Teacher spans sample: {flat_teacher_spans[:3] if flat_teacher_spans else 'none'}")
            self.logger.debug(f"🔍 Student spans sample: {flat_student_spans[:3] if flat_student_spans else 'none'}")
            
            # Calculate token counts from flattened spans
            total_teacher_tokens = sum(_extract_span_length(span) for span in flat_teacher_spans)
            total_student_tokens = sum(_extract_span_length(span) for span in flat_student_spans)
            total_assistant_tokens = total_teacher_tokens + total_student_tokens
            
            self.logger.debug(f"🔍 SPAN TOKENS: teacher={total_teacher_tokens}, student={total_student_tokens}, total={total_assistant_tokens}")
            
            if total_assistant_tokens == 0:
                self.logger.warning("No assistant tokens found in spans - falling back to student-only mode")
                return 0.0, total_llm_loss + coord_loss_total
        except Exception as e:
            self.logger.error(f"❌ Error computing span lengths: {e}")
            self.logger.error(f"   teacher_spans: {teacher_spans}")
            self.logger.error(f"   student_spans: {student_spans}")
            # Fallback to student-only mode on any error
            self.logger.warning("Falling back to student-only mode due to span parsing error")
            return 0.0, total_llm_loss + coord_loss_total
        
        # Proportional loss allocation based on token counts
        teacher_ratio = total_teacher_tokens / total_assistant_tokens
        student_ratio = total_student_tokens / total_assistant_tokens
        
        # Teacher loss: only proportional LLM loss (no coordinate losses)
        teacher_lm_loss = total_llm_loss * teacher_ratio
        
        # Student loss: proportional LLM loss + all coordinate losses
        # (coordinate losses only apply to student responses since only students do detection)
        student_lm_loss = (total_llm_loss * student_ratio) + coord_loss_total
        
        self.logger.debug(f"Span-based loss computation:")
        self.logger.debug(f"  Teacher tokens: {total_teacher_tokens}, Student tokens: {total_student_tokens}")
        self.logger.debug(f"  Teacher ratio: {teacher_ratio:.3f}, Student ratio: {student_ratio:.3f}")
        self.logger.debug(f"  Teacher loss: {teacher_lm_loss:.6f}, Student loss: {student_lm_loss:.6f}")
        
        return teacher_lm_loss, student_lm_loss
