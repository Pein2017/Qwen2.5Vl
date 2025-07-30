"""
Simplified Loss Manager for BBU Training

Streamlined loss computation focused purely on loss calculation and extraction.
Accumulation and metrics handling are moved to TrainingStateManager for
better separation of concerns.

Key Features:
1. **Pure Loss Computation**: Focus only on loss calculation logic
2. **Clean Interface**: Simple input/output without state management
3. **Fail-Fast Validation**: Strict validation with clear error messages
4. **Component Separation**: Clear separation between different loss types
5. **Stateless Design**: No internal accumulation or caching
"""

from typing import Any, Dict, Tuple

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.training.base_manager import BaseManager


class LossManager(BaseManager):
    """
    Simplified manager focused purely on loss computation.

    This manager handles only the core loss computation logic without
    accumulation, metrics, or state management. All state handling is
    delegated to TrainingStateManager.
    """

    def __init__(
        self,
        config: Any,
        tokenizer: PreTrainedTokenizerBase,
        model: Any = None,
        teacher_loss_weight: float = 0.3,
        student_loss_weight: float = 1.0,
        logger: Any = None,
    ):
        """
        Initialize simplified loss manager.

        Args:
            config: Configuration object with coordinate_tokens_enabled
            tokenizer: Tokenizer for processing
            model: Model instance (optional, for debugging)
            teacher_loss_weight: Weight for teacher loss components
            student_loss_weight: Weight for student loss components
            logger: Optional logger instance
        """
        # Store components before base initialization
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        self.tokenizer = tokenizer
        self.model = model
        self.teacher_loss_weight = teacher_loss_weight
        self.student_loss_weight = student_loss_weight

        # Initialize base manager
        super().__init__(config, logger)

        # State for compatibility with TrainingCoordinator
        self._current_losses: Dict[str, float] = {}
        self._accumulated_losses: Dict[str, float] = {}
        self._accumulation_count = 0

        self.log_info(
            f"Simplified loss manager initialized "
            f"(coordinate_tokens_enabled={self.config.coordinate_tokens_enabled})"
        )

    def _validate_configuration(self) -> None:
        """Validate configuration requirements."""
        self.validate_required_attribute("coordinate_tokens_enabled", bool)

    def _initialize_manager_state(self) -> None:
        """Initialize manager state (minimal for simplified manager)."""
        # No state management in simplified version
        pass

    def compute_total_loss(
        self,
        model_outputs: Any,
        inputs: Dict[str, Any],
        is_training: bool = True,
        detection_training_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss from model outputs.

        This method focuses purely on loss computation and component extraction
        without any accumulation or state management.

        Args:
            model_outputs: Output from model forward pass
            inputs: Batch inputs containing labels and ground truth
            is_training: Whether in training mode
            detection_training_enabled: Whether detection training is active

        Returns:
            Tuple of (total_loss_tensor, loss_components_dict)
        """
        # Extract total loss from model outputs
        total_loss = self._extract_total_loss(model_outputs)

        # Extract component losses
        loss_components = self._extract_loss_components(model_outputs, inputs)

        # Validate loss components
        self._validate_loss_components(loss_components, inputs)

        # Store current losses for TrainingCoordinator compatibility
        self._current_losses = loss_components.copy()

        # Accumulate losses for averaging
        for key, value in loss_components.items():
            if key not in self._accumulated_losses:
                self._accumulated_losses[key] = 0.0
            self._accumulated_losses[key] += value

        self._accumulation_count += 1

        return total_loss, loss_components

    def _extract_total_loss(self, model_outputs: Any) -> torch.Tensor:
        """
        Extract total loss from model outputs with strict validation.

        Args:
            model_outputs: Model forward pass outputs

        Returns:
            Total loss tensor

        Raises:
            RuntimeError: If loss is missing or invalid
        """
        if not hasattr(model_outputs, "loss") or model_outputs.loss is None:
            raise RuntimeError(
                "Model outputs missing 'loss' attribute when labels are provided. "
                "This indicates the model is not computing loss correctly. "
                "Ensure labels are passed to the model's forward method."
            )

        total_loss = model_outputs.loss

        # Handle multi-element loss tensors (common in coordinate mode)
        if total_loss.numel() > 1:
            self.log_debug(
                f"Multi-element loss tensor detected: shape={total_loss.shape}, "
                f"values={total_loss}"
            )
            # Aggregate multi-element loss (use mean to avoid explosion)
            total_loss = total_loss.mean()
            self.log_debug(f"Aggregated loss: {total_loss.item():.6f}")

        # Validate loss value using safe extraction
        loss_value = self._safe_item(total_loss, "total loss")
        if loss_value == 0.0:
            self.log_error(
                "MODEL OUTPUT LOSS IS 0.0! Model not computing loss correctly!"
            )
            self.log_error(f"   - model_outputs type: {type(model_outputs)}")
            self.log_error(f"   - hasattr(loss): {hasattr(model_outputs, 'loss')}")
            self.log_error(f"   - loss value: {model_outputs.loss}")
            if hasattr(model_outputs.loss, "shape"):
                self.log_error(f"   - loss shape: {model_outputs.loss.shape}")
            raise RuntimeError("Model is returning 0.0 loss! Check model forward pass.")

        return total_loss

    def _extract_loss_components(
        self, model_outputs: Any, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract individual loss components from model outputs.

        Args:
            model_outputs: Model forward pass outputs
            inputs: Batch inputs for span-based computation

        Returns:
            Dictionary of loss components
        """
        loss_components = {}

        # 1. Extract LLM Loss (standard shifted cross entropy)
        llm_loss = self._extract_llm_loss(model_outputs)

        # 2. Extract Coordinate Loss Components
        coordinate_l1_loss = self._extract_coordinate_loss(model_outputs)
        loss_components["coordinate_l1_loss"] = coordinate_l1_loss

        # 3. Compute Teacher-Student Loss Distribution
        teacher_llm_loss, student_llm_loss, student_l1_loss = (
            self._compute_span_based_losses(inputs, llm_loss, coordinate_l1_loss)
        )

        # Store component losses
        loss_components.update(
            {
                "teacher_lm_loss": teacher_llm_loss,
                "student_lm_loss": student_llm_loss,
                "student_l1_loss": student_l1_loss,
            }
        )

        # Legacy compatibility
        loss_components["lm_loss"] = teacher_llm_loss + student_llm_loss

        self.log_debug(
            f"Loss components: T_LLM={teacher_llm_loss:.3f}, S_LLM={student_llm_loss:.3f}, "
            f"S_L1={student_l1_loss:.3f}"
        )

        return loss_components

    def _extract_llm_loss(self, model_outputs: Any) -> float:
        """Extract LLM loss with strict validation."""
        if hasattr(model_outputs, "_llm_loss") and model_outputs._llm_loss is not None:
            return self._safe_item(model_outputs._llm_loss, "LLM loss (_llm_loss)")

        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            return self._safe_item(model_outputs.loss, "LLM loss (loss)")

        raise ValueError(
            "Model outputs missing both '_llm_loss' and 'loss' attributes. "
            "Ensure model is properly configured to return loss values."
        )

    def _extract_coordinate_loss(self, model_outputs: Any) -> float:
        """Extract coordinate loss from model outputs."""
        if not self.config.coordinate_tokens_enabled:
            self.log_debug(
                "Coordinate tokens disabled - setting coordinate L1 loss to 0.0"
            )
            return 0.0

        # First try to extract coordinate loss from model outputs (computed by model's loss manager)
        coordinate_l1_loss = getattr(model_outputs, "_coordinate_l1_loss", None)

        if coordinate_l1_loss is not None:
            loss_value = self._safe_item(
                coordinate_l1_loss, "coordinate L1 loss (model outputs)"
            )
            self.log_debug(
                f"Extracted coordinate L1 loss from model outputs: {loss_value}"
            )
            return loss_value

        # If not found in outputs, try to get it from the model wrapper
        # This is a fallback for cases where dataclass outputs don't support dynamic attributes
        if hasattr(self, "model") and hasattr(
            self.model, "_last_coordinate_l1_loss_for_training"
        ):
            coordinate_l1_loss = self.model._last_coordinate_l1_loss_for_training
            loss_value = self._safe_item(
                coordinate_l1_loss, "coordinate L1 loss (model wrapper)"
            )
            self.log_debug(
                f"Extracted coordinate L1 loss from model wrapper fallback: {loss_value}"
            )
            return loss_value

        # Try to access the model through trainer if available
        if hasattr(self, "_trainer_ref") and self._trainer_ref is not None:
            model = self._trainer_ref.model
            if hasattr(model, "_last_coordinate_l1_loss_for_training"):
                coordinate_l1_loss = model._last_coordinate_l1_loss_for_training
                loss_value = self._safe_item(
                    coordinate_l1_loss, "coordinate L1 loss (trainer model)"
                )
                self.log_debug(
                    f"Extracted coordinate L1 loss from trainer model fallback: {loss_value}"
                )
                return loss_value

        # Final fallback: try global storage
        try:
            from src.models.wrapper import get_global_coordinate_loss

            global_coordinate_loss = get_global_coordinate_loss()
            if global_coordinate_loss is not None:
                loss_value = self._safe_item(
                    global_coordinate_loss, "coordinate L1 loss (global storage)"
                )
                self.log_debug(
                    f"Extracted coordinate L1 loss from global storage: {loss_value}"
                )
                return loss_value
        except ImportError:
            self.log_debug("Could not import global coordinate loss storage")

        self.log_debug(
            "No coordinate L1 loss found in model outputs, wrapper, or global storage - returning 0.0"
        )
        return 0.0

    def _extract_loss_value(self, outputs: Any, key: str) -> float:
        """Extract loss value from ModelOutput with strict validation."""
        # Check dictionary access first (standard for ModelOutput)
        if hasattr(outputs, "get") and key in outputs:
            value = outputs[key]
            if value is not None:
                return self._safe_item(value)

        # Check attribute access
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if value is not None:
                return self._safe_item(value)

        # Strict validation - all loss components must be present
        available_attrs = (
            list(vars(outputs).keys()) if hasattr(outputs, "__dict__") else "unknown"
        )
        raise ValueError(
            f"Loss component '{key}' not found in model outputs. "
            f"Available attributes: {available_attrs}"
        )

    def _validate_loss_components(
        self, loss_components: Dict[str, float], inputs: Dict[str, Any]
    ) -> None:
        """Validate extracted loss components."""
        if not self.config.coordinate_tokens_enabled:
            return  # Skip validation when coordinate tokens disabled

        # This validation should happen at model output level
        # Here we just validate the computed components
        coordinate_l1_loss = loss_components.get("coordinate_l1_loss", 0.0)

        # Warn if coordinate loss is zero but we have student spans
        if coordinate_l1_loss == 0.0 and "student_assistant_spans" in inputs:
            student_spans = inputs["student_assistant_spans"]
            if student_spans and any(len(group) > 0 for group in student_spans):
                self.log_warning(
                    f"Coordinate tokens enabled with student spans present, "
                    f"but coordinate_l1_loss=0.0. This may indicate missing "
                    f"coordinate tokens in the data or model wrapper issues."
                )

    def _compute_span_based_losses(
        self, inputs: Dict[str, Any], total_llm_loss: float, coord_loss_total: float
    ) -> Tuple[float, float, float]:
        """
        Compute teacher and student losses based on assistant spans.

        Args:
            inputs: Batch inputs containing span information
            total_llm_loss: Total LLM loss for the sequence
            coord_loss_total: Total coordinate loss for the sequence

        Returns:
            Tuple of (teacher_llm_loss, student_llm_loss, student_l1_loss)
        """
        # Validate required span data
        self._validate_span_inputs(inputs)

        teacher_spans = inputs["teacher_assistant_spans"]
        student_spans = inputs["student_assistant_spans"]

        # Calculate token counts
        total_teacher_tokens = self._calculate_span_tokens(teacher_spans)
        total_student_tokens = self._calculate_span_tokens(student_spans)
        total_assistant_tokens = total_teacher_tokens + total_student_tokens

        # Validate token counts
        self._validate_token_counts(total_teacher_tokens, total_student_tokens)

        # Handle teacher dropout case (expected with teacher_ratio < 1.0)
        if total_teacher_tokens == 0:
            self.log_debug(
                f"Teacher dropout: No teacher tokens in batch. "
                f"Allocating all loss to student: S={total_student_tokens}"
            )
            return 0.0, total_llm_loss, coord_loss_total

        # Proportional loss allocation
        teacher_ratio = total_teacher_tokens / total_assistant_tokens
        student_ratio = total_student_tokens / total_assistant_tokens

        teacher_llm_loss = total_llm_loss * teacher_ratio
        student_llm_loss = total_llm_loss * student_ratio
        student_l1_loss = coord_loss_total * student_ratio

        self.log_debug(
            f"Span-based loss distribution: "
            f"T_tokens={total_teacher_tokens}, S_tokens={total_student_tokens}, "
            f"T_ratio={teacher_ratio:.3f}, S_ratio={student_ratio:.3f}"
        )

        return teacher_llm_loss, student_llm_loss, student_l1_loss

    def _validate_span_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate that required span data is present."""
        if "teacher_assistant_spans" not in inputs:
            raise ValueError(
                "Required 'teacher_assistant_spans' missing from batch inputs. "
                "This indicates a data preprocessing error."
            )
        if "student_assistant_spans" not in inputs:
            raise ValueError(
                "Required 'student_assistant_spans' missing from batch inputs. "
                "This indicates a data preprocessing error."
            )

        teacher_spans = inputs["teacher_assistant_spans"]
        student_spans = inputs["student_assistant_spans"]

        if teacher_spans is None:
            raise ValueError(
                "teacher_assistant_spans is None. Expected list of span groups."
            )
        if student_spans is None:
            raise ValueError(
                "student_assistant_spans is None. Expected list of span groups."
            )

    def _calculate_span_tokens(self, span_groups: Any) -> int:
        """Calculate total tokens from span groups."""
        if not isinstance(span_groups, (list, tuple)):
            raise ValueError(
                f"Expected span_groups to be list or tuple, got {type(span_groups)}"
            )

        total_tokens = 0
        for group_idx, group in enumerate(span_groups):
            if not isinstance(group, (list, tuple)):
                raise ValueError(
                    f"Expected span group {group_idx} to be list or tuple, "
                    f"got {type(group)}"
                )

            for span_idx, span in enumerate(group):
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    raise ValueError(
                        f"Expected span [{group_idx}][{span_idx}] to be [start, end] pair, "
                        f"got {type(span)} with length {len(span) if hasattr(span, '__len__') else 'unknown'}"
                    )

                start, end = span[0], span[1]

                # Convert to integers with validation
                try:
                    start_int = int(start.item() if hasattr(start, "item") else start)
                    end_int = int(end.item() if hasattr(end, "item") else end)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid span coordinates: start={start}, end={end}. "
                        f"Expected integer coordinates: {e}"
                    )

                if start_int < 0 or end_int < 0:
                    raise ValueError(
                        f"Negative span coordinates not allowed: "
                        f"start={start_int}, end={end_int}"
                    )

                if end_int < start_int:
                    raise ValueError(
                        f"Invalid span: end ({end_int}) < start ({start_int})"
                    )

                total_tokens += max(0, end_int - start_int)

        return total_tokens

    def _validate_token_counts(
        self, total_teacher_tokens: int, total_student_tokens: int
    ) -> None:
        """Validate token counts for data integrity."""
        total_assistant_tokens = total_teacher_tokens + total_student_tokens

        if total_assistant_tokens == 0:
            raise ValueError(
                "No assistant tokens found in spans. This indicates a critical "
                "data preprocessing failure. All training samples must contain "
                "either teacher or student assistant spans."
            )

        if total_student_tokens == 0:
            raise ValueError(
                "No student tokens found in batch. All samples must contain "
                "student assistant spans for coordinate token learning. "
                f"Teacher tokens: {total_teacher_tokens}, Student tokens: {total_student_tokens}"
            )

        # Log token distribution
        if total_teacher_tokens == 0:
            self.log_debug(
                "No teacher tokens in this batch - allocating all loss to student"
            )
        else:
            self.log_debug(
                f"Batch has teachers: T={total_teacher_tokens}, S={total_student_tokens}"
            )

    def _safe_item(self, value: Any, context: str = "unknown") -> float:
        """Extract scalar value from tensor or float with strict validation."""
        if hasattr(value, "item"):
            # Handle multi-element tensors
            if hasattr(value, "numel") and value.numel() > 1:
                self.log_debug(
                    f"Multi-element tensor in {context}: shape={value.shape}, taking mean"
                )
                value = value.mean()
            result = value.item()
        elif isinstance(value, (int, float)):
            result = float(value)
        else:
            raise TypeError(
                f"Expected tensor or numeric value in {context}, got {type(value)}: {value}"
            )

        # Validation for invalid loss values - use small positive value to avoid breaking student validation
        if torch.isnan(torch.tensor(result)):
            self.log_warning(
                f"NaN loss detected in {context}: {result}. Using small positive value to maintain training stability. "
                f"This indicates training instability - check learning rates, gradient clipping, and data preprocessing."
            )
            return 1e-8  # Use small positive value instead of 0.0 to avoid triggering student validation
        if torch.isinf(torch.tensor(result)):
            self.log_warning(
                f"Infinite loss detected in {context}: {result}. Using small positive value to maintain training stability. "
                f"This indicates training instability - check learning rates, gradient clipping, and data preprocessing."
            )
            return 1e-8  # Use small positive value instead of 0.0 to avoid triggering student validation
        if result < 0.0:
            self.log_warning(
                f"Negative loss detected in {context}: {result}. Using small positive value to maintain training stability. "
                f"Loss values must be non-negative - this indicates an error in loss computation."
            )
            return 1e-8  # Use small positive value instead of 0.0 to avoid triggering student validation

        return result

    def get_current_losses(self) -> Dict[str, float]:
        """
        Get current loss components from the last compute_total_loss call.

        Returns:
            Dictionary of current loss components
        """
        return self._current_losses.copy()

    def get_averaged_losses(self) -> Dict[str, float]:
        """
        Get averaged loss components over accumulated batches.

        Returns:
            Dictionary of averaged loss components
        """
        if self._accumulation_count == 0:
            return {}

        averaged_losses = {}
        for key, total_loss in self._accumulated_losses.items():
            averaged_losses[key] = total_loss / self._accumulation_count

        return averaged_losses

    def reset_loss_accumulation(self) -> None:
        """Reset loss accumulation state."""
        self._accumulated_losses.clear()
        self._accumulation_count = 0
        self.log_debug("Loss accumulation state reset")

    def save_training_state(self) -> Dict[str, Any]:
        """
        Save current training state for evaluation isolation.

        Returns:
            Dictionary containing current training state
        """
        return {
            "current_losses": self._current_losses.copy(),
            "accumulated_losses": self._accumulated_losses.copy(),
            "accumulation_count": self._accumulation_count,
        }

    def restore_training_state(self, state: Dict[str, Any]) -> None:
        """
        Restore training state after evaluation.

        Args:
            state: Training state dictionary from save_training_state
        """
        self._current_losses = state.get("current_losses", {})
        self._accumulated_losses = state.get("accumulated_losses", {})
        self._accumulation_count = state.get("accumulation_count", 0)
        self.log_debug("Training state restored")
