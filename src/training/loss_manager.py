"""
Simplified Loss Manager for BBU Training

Streamlined loss computation without excessive validation and accumulation complexity.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from src.config import config
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
        if hasattr(model_outputs, 'loss') and model_outputs.loss is not None:
            total_loss = model_outputs.loss
            self.logger.info(f"🔍 TRAINING LOSS CHECK: model_outputs.loss = {total_loss.item():.6f}")
            if total_loss.item() == 0.0:
                self.logger.error(f"🚨 MODEL OUTPUT LOSS IS 0.0! This means the model is not computing loss correctly!")
                self.logger.error(f"   - model_outputs type: {type(model_outputs)}")
                self.logger.error(f"   - hasattr(loss): {hasattr(model_outputs, 'loss')}")
                self.logger.error(f"   - loss value: {model_outputs.loss}")
                self.logger.error(f"   - loss shape: {model_outputs.loss.shape if hasattr(model_outputs.loss, 'shape') else 'no shape'}")
                raise RuntimeError(f"Model is returning 0.0 loss! Check model forward pass.")
        else:
            # EXPOSE ERROR: This should never happen in training!
            labels = inputs.get("labels")
            self.logger.error(f"🚨 LOSS COMPUTATION FALLBACK TRIGGERED!")
            self.logger.error(f"   - model_outputs.loss is None: {not hasattr(model_outputs, 'loss') or model_outputs.loss is None}")
            self.logger.error(f"   - labels present: {labels is not None}")
            if labels is not None:
                self.logger.error(f"   - labels shape: {labels.shape}")
                total_loss = F.cross_entropy(
                    model_outputs.logits.view(-1, model_outputs.logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                self.logger.error(f"   - Computed fallback loss: {total_loss.item():.6f}")
            else:
                self.logger.error(f"   - NO LABELS FOUND - Setting loss to 0.0 (THIS IS WRONG!)")
                total_loss = torch.tensor(0.0, device=model_outputs.logits.device)
                raise RuntimeError("Training loss is 0.0 because no labels found in inputs! This means data loading is broken.")

        # Clean loss separation: LLM loss vs Coordinate loss components
        loss_components = {}
        
        # 1. LLM Loss (standard shifted cross entropy from Qwen2.5)
        llm_loss = self._safe_item(model_outputs._regular_loss) if hasattr(model_outputs, '_regular_loss') else self._safe_item(model_outputs.loss)
        loss_components["llm_loss"] = llm_loss
        
        # 2. Individual Coordinate Loss Components (no duplicates or summations)
        focal_loss = self._safe_item(model_outputs._focal_loss) if hasattr(model_outputs, '_focal_loss') else 0.0
        l1_loss = self._safe_item(model_outputs._l1_loss) if hasattr(model_outputs, '_l1_loss') else 0.0
        giou_loss = self._safe_item(model_outputs._giou_loss) if hasattr(model_outputs, '_giou_loss') else 0.0
        
        loss_components["focal_loss"] = focal_loss
        loss_components["l1_loss"] = l1_loss  
        loss_components["giou_loss"] = giou_loss
        
        # Compute total coordinate loss for teacher-student differentiation
        coord_loss_total = focal_loss + l1_loss + giou_loss
        
        # 3. Teacher-Student differentiation (determine from sample type)
        is_teacher_sample = self._is_teacher_sample(inputs)
        if is_teacher_sample:
            # Teacher samples: only LLM loss with coordinate tokens inserted
            loss_components["teacher_lm_loss"] = llm_loss
            loss_components["student_lm_loss"] = 0.0
        else:
            # Student samples: both LLM + coordinate losses
            loss_components["teacher_lm_loss"] = 0.0
            loss_components["student_lm_loss"] = llm_loss + coord_loss_total
        
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
        if hasattr(value, 'item'):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(f"Expected tensor or numeric value, got {type(value)}: {value}")
    
    def _is_teacher_sample(self, inputs: Dict[str, Any]) -> bool:
        """
        Determine if the current sample is a teacher sample.
        
        Teacher samples are typically identified by having longer descriptions
        or specific markers in the input data.
        
        Args:
            inputs: Batch inputs containing labels and metadata
            
        Returns:
            True if this is a teacher sample, False if student sample
        """
        # Method 1: Check if this is a teacher sample based on input_ids length
        # Teacher samples tend to have longer, more descriptive text
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if hasattr(input_ids, 'shape'):
                seq_length = input_ids.shape[-1]
                # Heuristic: Teacher samples are typically longer
                # This is a simple approach - could be improved with explicit marking
                return seq_length > 1500  # Adjust threshold based on your data
        
        # Method 2: Check for explicit teacher marker in batch
        if "is_teacher" in inputs:
            return inputs["is_teacher"]
        
        # Method 3: Check metadata or other indicators
        if "teacher_assistant_spans" in inputs:
            # If teacher_assistant_spans exist, this might be a teacher sample
            spans = inputs["teacher_assistant_spans"]
            return spans is not None and len(spans) > 0
        
        # Default: assume student sample if we can't determine
        return False