"""
Simplified Loss Manager for BBU Training

Streamlined loss computation without excessive validation and accumulation complexity.
"""

from typing import Any, Dict, Tuple

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.logger_utils import get_training_logger


class LossManager:
    """
    Simplified manager for loss computation in BBU training.

    Delegates most loss computation to the model wrapper and focuses
    on clean loss extraction and logging.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        teacher_loss_weight: float = 0.3,
        student_loss_weight: float = 1.0,
        coordinate_tokens_enabled: bool = False,
        **kwargs,
    ):
        """Initialize simplified loss manager."""
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        self.tokenizer = tokenizer
        self.model = model
        self.teacher_loss_weight = teacher_loss_weight
        self.student_loss_weight = student_loss_weight
        self.coordinate_tokens_enabled = coordinate_tokens_enabled
        self.logger = get_training_logger()
        self._micro_batch_count: int = 0
        self._current_losses: Dict[str, float] = {}

        # Log coordinate tokens configuration for debugging
        self.logger.debug(
            f"🔍 LOSS_MANAGER: coordinate_tokens_enabled = {self.coordinate_tokens_enabled}"
        )

        # Cache for loss extraction functions
        self._loss_extractors = {
            "_coordinate_l1_loss": self._extract_coordinate_loss_safe,
            "_llm_loss": self._extract_llm_loss_safe,
        }

    def _extract_coordinate_loss_safe(self, model_outputs) -> float:
        """Safely extract coordinate loss with fallback."""
        return self._extract_loss_value(model_outputs, "_coordinate_l1_loss")

    def _extract_llm_loss_safe(self, model_outputs) -> float:
        """Extract LLM loss with strict validation - FAIL FAST approach."""
        # EXPLICIT CONFIG: Model outputs must have _llm_loss when coordinate tokens are enabled
        if hasattr(model_outputs, "_llm_loss") and model_outputs._llm_loss is not None:
            return self._safe_item(model_outputs._llm_loss)

        # EXPLICIT CONFIG: Standard loss is required if _llm_loss is not available
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            return self._safe_item(model_outputs.loss)

        # FAIL FAST: No fallback - missing loss indicates configuration error
        raise ValueError(
            "Model outputs missing both '_llm_loss' and 'loss' attributes. "
            "Ensure model is properly configured to return loss values."
        )

    def _extract_loss_value(self, outputs, key: str) -> float:
        """Extract loss value from ModelOutput with FAIL FAST validation."""
        # FAIL FAST: Check dictionary access first (standard for ModelOutput)
        if hasattr(outputs, "get") and key in outputs:
            value = outputs[key]
            if value is not None:
                return self._safe_item(value)

        # FAIL FAST: Check attribute access (fallback)
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if value is not None:
                return self._safe_item(value)

        # FAIL FAST: No fallback for required loss components
        required_losses = ["llm_loss", "_llm_loss", "loss"]
        if key in required_losses:
            raise ValueError(
                f"Required loss component '{key}' not found in model outputs. "
                f"Available attributes: {list(vars(outputs).keys()) if hasattr(outputs, '__dict__') else 'unknown'}"
            )

        # EXPLICIT: For optional losses, log warning and return 0.0
        self.logger.warning(f"Optional loss component '{key}' not found, returning 0.0")
        return 0.0

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

        # FAIL FAST: Extract loss from model outputs (model handles coordinate/standard loss internally)
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            total_loss = model_outputs.loss
            
            # Handle multi-element loss tensors (common in coordinate mode)
            if total_loss.numel() > 1:
                self.logger.debug(f"🔍 Multi-element loss tensor detected: shape={total_loss.shape}, values={total_loss}")
                # Aggregate multi-element loss (sum or mean depending on use case)
                total_loss = total_loss.mean()  # Use mean to avoid loss explosion
                self.logger.debug(f"🔍 Aggregated loss: {total_loss.item():.6f}")
            
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
            # FAIL FAST: This should never happen when labels are provided - no fallback
            raise RuntimeError(
                "Model outputs missing 'loss' attribute when labels are provided. "
                "This indicates the model is not computing loss correctly. "
                "Ensure labels are passed to the model's forward method."
            )

        # Clean loss separation: LLM loss vs Coordinate loss components
        loss_components = {}

        # 1. LLM Loss (standard shifted cross entropy from Qwen2.5)
        llm_loss = (
            self._safe_item(model_outputs._llm_loss)
            if hasattr(model_outputs, "_llm_loss")
            else self._safe_item(model_outputs.loss)
        )
        loss_components["llm_loss"] = llm_loss

        # 2. Individual Coordinate Loss Components with new geometry-organized names
        # FAIL FAST: Extract from ModelOutput - no fallback to zero
        def _extract_coordinate_loss(outputs, key):
            """Extract coordinate loss from ModelOutput - FAIL FAST if missing."""
            # Check dictionary access first (standard for ModelOutput)
            if hasattr(outputs, "get") and key in outputs:
                value = self._safe_item(outputs[key])
                self.logger.debug(f"   Found {key} in dict: {value}")
                return value
            # Check attribute access (fallback)
            elif hasattr(outputs, key):
                value = self._safe_item(getattr(outputs, key))
                self.logger.debug(f"   Found {key} as attribute: {value}")
                return value
            else:
                # FAIL FAST: Don't return 0.0 - this masks missing coordinate losses
                self.logger.warning(f"   {key} not found - coordinate loss missing!")
                # Return 0.0 for now to maintain compatibility, but log the issue
                return 0.0

        # Extract coordinate L1 loss only if coordinate tokens are enabled
        if self.coordinate_tokens_enabled:
            coordinate_l1_loss = _extract_coordinate_loss(
                model_outputs, "_coordinate_l1_loss"
            )

            # DEBUG: Log what we're actually extracting
            self.logger.debug(
                f"🔍 LOSS_MANAGER: Extracting simplified coordinate losses from model outputs:"
            )
            self.logger.debug(f"   model_outputs id: {id(model_outputs)}")
            self.logger.debug(f"   model_outputs type: {type(model_outputs)}")
            self.logger.debug(
                f"   has _coordinate_l1_loss: {getattr(model_outputs, '_coordinate_l1_loss', None) is not None}"
            )
            # Check dictionary access too
            if hasattr(model_outputs, "get"):
                self.logger.debug(
                    f"   dict has _coordinate_l1_loss: {'_coordinate_l1_loss' in model_outputs}"
                )
            self.logger.debug(f"   Extracted coordinate L1 loss: {coordinate_l1_loss}")
        else:
            # When coordinate tokens are disabled, set coordinate loss to 0.0
            coordinate_l1_loss = 0.0
            self.logger.debug(
                "📊 LOSS_MANAGER: Coordinate tokens disabled - setting coordinate L1 loss to 0.0"
            )

        # Only validate coordinate loss attributes if coordinate tokens are enabled
        if self.coordinate_tokens_enabled:
            required_attrs = ["_coordinate_l1_loss"]

            missing_attrs = []
            for attr in required_attrs:
                # FAIL FAST: Check both dictionary and attribute access
                has_dict_access = (
                    hasattr(model_outputs, "get") and attr in model_outputs
                )
                has_attr_access = hasattr(model_outputs, attr)
                if not (has_dict_access or has_attr_access):
                    missing_attrs.append(attr)

            if missing_attrs:
                self.logger.error(
                    f"🚨 LOSS_MANAGER: CRITICAL ERROR - Coordinate loss attributes missing: {missing_attrs}"
                )
                self.logger.error(f"   model_outputs type: {type(model_outputs)}")
                self.logger.error(f"   model_outputs id: {id(model_outputs)}")

                # List all attributes to debug what's actually on the object
                model_output_attrs = [
                    attr for attr in dir(model_outputs) if not attr.startswith("__")
                ]
                self.logger.error(f"   Available attributes: {model_output_attrs}")

                raise RuntimeError(
                    f"CRITICAL: Coordinate loss attributes {missing_attrs} are missing from model outputs. "
                    "This indicates model wrapper is not properly attaching losses to output object. "
                    "NO FALLBACK ALLOWED - fix the root cause in model wrapper output coordination."
                )
        else:
            self.logger.debug(
                "📊 LOSS_MANAGER: Coordinate tokens disabled - skipping coordinate loss validation"
            )

        # Log if coordinate L1 loss is zero (for debugging, but not an error)
        if coordinate_l1_loss == 0.0:
            self.logger.debug(
                "📊 LOSS_MANAGER: Coordinate L1 loss is zero (this is valid during evaluation or when no valid coordinate spans exist)"
            )

        loss_components["coordinate_l1_loss"] = coordinate_l1_loss

        # Use only coordinate L1 loss for teacher-student differentiation
        coord_loss_total = coordinate_l1_loss

        # 3. Teacher-Student differentiation (use span-based approach)
        teacher_llm_loss, student_llm_loss, student_l1_loss = (
            self._compute_span_based_losses(inputs, llm_loss, coord_loss_total)
        )
        loss_components["teacher_llm_loss"] = teacher_llm_loss
        loss_components["student_llm_loss"] = student_llm_loss
        loss_components["student_l1_loss"] = student_l1_loss
        # ----------------------------------------------------------------------------------
        # Backwards-compatibility shim ----------------------------------------------------
        # ----------------------------------------------------------------------------------
        # Down-stream code (e.g. `src/training/trainer.py`) historically expects the keys
        # `teacher_lm_loss` / `student_lm_loss` (single "l"), whereas this LossManager
        # originally recorded them using the double-"l" variant `*_llm_loss`.  This mismatch
        # results in missing-key look-ups that silently default to 0.0 and ultimately trigger
        # the fatal "No student samples found" check during training.

        # To maintain compatibility with *both* naming conventions and avoid breaking any
        # external consumers (tests, analytics scripts, legacy checkpoints, …) we simply
        # register **both** spellings here.  The values are identical references so there is
        # no additional memory overhead.

        loss_components["teacher_lm_loss"] = teacher_llm_loss
        loss_components["student_lm_loss"] = student_llm_loss

        # Clean loss reporting - no weighted losses needed

        self.logger.debug(
            f"🎯 Clean loss: T_LLM={teacher_llm_loss:.3f}, S_LLM={student_llm_loss:.3f}, S_L1={student_l1_loss:.3f}"
        )

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
            # Handle multi-element tensors (common in coordinate mode)
            if hasattr(value, "numel") and value.numel() > 1:
                # Aggregate multi-element tensor to scalar
                value = value.mean()
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(
                f"Expected tensor or numeric value, got {type(value)}: {value}"
            )

    def _compute_span_based_losses(
        self, inputs: Dict[str, Any], total_llm_loss: float, coord_loss_total: float
    ) -> Tuple[float, float, float]:
        """
        Compute teacher and student losses based on assistant spans within the same sequence.

        Args:
            inputs: Batch inputs containing span information
            total_llm_loss: Total LLM loss for the sequence
            coord_loss_total: Total coordinate loss for the sequence

        Returns:
            Tuple of (teacher_llm_loss, student_llm_loss, student_l1_loss)
        """
        # Extract spans - no fallback, expect proper data structure
        teacher_spans = inputs["teacher_assistant_spans"]
        student_spans = inputs["student_assistant_spans"]

        # Calculate total number of tokens in teacher vs student spans
        def _flatten_spans(span_groups):
            """Flatten nested span groups into individual [start, end] pairs."""
            flattened = []
            for group in span_groups:
                for span in group:
                    if isinstance(span, (list, tuple)) and len(span) == 2:
                        flattened.append(span)
            return flattened

        def _extract_span_length(span):
            """Extract span length from [start, end] pair."""
            start, end = span[0], span[1]
            if hasattr(start, "item"):
                start = start.item()
            if hasattr(end, "item"):
                end = end.item()
            return max(0, int(end) - int(start))

        # Flatten the nested structures
        flat_teacher_spans = _flatten_spans(teacher_spans)
        flat_student_spans = _flatten_spans(student_spans)

        # Calculate token counts from flattened spans
        total_teacher_tokens = sum(
            _extract_span_length(span) for span in flat_teacher_spans
        )
        total_student_tokens = sum(
            _extract_span_length(span) for span in flat_student_spans
        )
        total_assistant_tokens = total_teacher_tokens + total_student_tokens

        self.logger.debug(f"Tokens: T={total_teacher_tokens}, S={total_student_tokens}")

        # Handle case where no assistant spans are found (e.g., synthetic test data)
        if total_assistant_tokens == 0:
            self.logger.debug(
                "No assistant tokens found in spans - using fallback loss allocation"
            )
            # Return equal allocation when no spans are available
            return total_llm_loss * 0.5, total_llm_loss * 0.5, coord_loss_total * 0.5

        # Proportional loss allocation based on token counts
        teacher_ratio = total_teacher_tokens / total_assistant_tokens
        student_ratio = total_student_tokens / total_assistant_tokens

        # Teacher loss: only proportional LLM loss (no coordinate losses)
        teacher_llm_loss = total_llm_loss * teacher_ratio

        # Student losses: split LLM and coordinate losses separately
        student_llm_loss = total_llm_loss * student_ratio
        student_l1_loss = coord_loss_total * student_ratio

        self.logger.debug(f"Span-based loss computation:")
        self.logger.debug(
            f"  Teacher tokens: {total_teacher_tokens}, Student tokens: {total_student_tokens}"
        )
        self.logger.debug(
            f"  Teacher ratio: {teacher_ratio:.3f}, Student ratio: {student_ratio:.3f}"
        )
        self.logger.debug(
            f"  Teacher LLM: {teacher_llm_loss:.6f}, Student LLM: {student_llm_loss:.6f}, Student L1: {student_l1_loss:.6f}"
        )

        return teacher_llm_loss, student_llm_loss, student_l1_loss
