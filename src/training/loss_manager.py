"""
Loss Manager for Multi-Component BBU Training

This module extracts and centralizes all loss computation logic from the monolithic
BBUTrainer class. It handles:

- Language modeling loss computation and NaN detection
- Teacher-student loss splitting and tracking
- Detection loss integration and coordination
- Loss accumulation across gradient accumulation steps
- Component-wise loss averaging and logging
- Ground truth object extraction and validation

Key Features:
- Clean separation of loss logic from training orchestration
- Robust loss accumulation with proper averaging
- Component-wise loss tracking for detailed monitoring
- Integration with detection system when enabled
- Proper teacher-student loss attribution
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase

from src.config import config
from src.detection.detection_loss import DetectionLoss
from src.logger_utils import get_training_logger
from src.utils.schema import GroundTruthObject
from src.utils.utils import IGNORE_INDEX


class LossManager:
    """
    Centralized manager for all loss computation in BBU training.
    
    Handles multi-component loss computation including language modeling,
    detection, and teacher-student learning with proper accumulation
    and averaging across gradient accumulation steps.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        detection_enabled: bool = True,
        **detection_config
    ):
        """
        Initialize loss manager with detection configuration.
        
        Args:
            tokenizer: Tokenizer for caption loss computation
            detection_enabled: Whether to enable detection loss
            **detection_config: Detection loss configuration parameters
        """
        self.tokenizer = tokenizer
        self.detection_enabled = detection_enabled
        self.logger = get_training_logger()
        
        # Initialize detection loss if enabled
        self.detection_loss = None
        if detection_enabled:
            self.detection_loss = DetectionLoss(
                tokenizer=tokenizer,
                **detection_config
            )
            self.logger.info("✅ Detection loss initialized")
        
        # Current loss components (single forward pass)
        self._current_lm_loss: float = 0.0
        self._current_teacher_lm_loss: float = 0.0
        self._current_student_lm_loss: float = 0.0
        self._current_bbox_loss: float = 0.0
        self._current_caption_loss: float = 0.0
        self._current_objectness_loss: float = 0.0
        self._current_bbox_l1_loss: float = 0.0
        self._current_bbox_giou_loss: float = 0.0
        
        # Loss accumulators for gradient accumulation
        self._accumulated_lm_loss: float = 0.0
        self._accumulated_teacher_lm_loss: float = 0.0
        self._accumulated_student_lm_loss: float = 0.0
        self._accumulated_bbox_l1_loss: float = 0.0
        self._accumulated_bbox_giou_loss: float = 0.0
        self._accumulated_caption_loss: float = 0.0
        self._accumulated_objectness_loss: float = 0.0
        
        # Micro-batch counter for proper averaging
        self._micro_batch_count: int = 0
        
        # Teacher performance tracking
        self._teacher_performance_stats = {
            'student_with_teacher_count': 0,
            'student_without_teacher_count': 0,
            'student_with_teacher_loss_sum': 0.0,
            'student_without_teacher_loss_sum': 0.0,
            'log_interval': 50  # Log every 50 batches with teacher-student data
        }
    
    def compute_total_loss(
        self,
        model_outputs: Any,
        inputs: Dict[str, Any],
        is_training: bool = True,
        detection_training_enabled: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including LM and detection components.
        
        Args:
            model_outputs: Output from model forward pass
            inputs: Batch inputs containing labels and ground truth
            is_training: Whether in training mode (affects accumulation)
            detection_training_enabled: Whether detection training is active
            
        Returns:
            Tuple of (total_loss_tensor, loss_components_dict)
        """
        if is_training:
            self._micro_batch_count += 1
        
        # Extract ground truth objects
        ground_truth_objects = self._extract_ground_truth_objects(inputs)
        
        # 1. Compute language modeling loss
        lm_loss = self._compute_language_modeling_loss(model_outputs, inputs)
        
        # 2. Compute teacher-student loss split
        teacher_loss, student_loss = self._compute_teacher_student_losses(
            model_outputs.logits, inputs.get("labels"), inputs
        )
        
        # 3. Compute detection loss if enabled
        detection_loss_tensor = torch.tensor(0.0, device=lm_loss.device)
        detection_components = {}
        
        if (
            self.detection_enabled
            and detection_training_enabled
            and ground_truth_objects
            and any(len(gt_list) > 0 for gt_list in ground_truth_objects)
        ):
            detection_loss_tensor, detection_components = self._compute_detection_loss(
                model_outputs, inputs, ground_truth_objects
            )
        
        # 4. Combine total loss for backpropagation WITH teacher-student weighting
        # Get teacher-student weights from config
        teacher_weight = getattr(config, 'teacher_loss_weight', 0.3)
        student_weight = getattr(config, 'student_loss_weight', 1.0)
        
        # Apply weights and combine ALL losses for backpropagation
        weighted_teacher_loss = teacher_weight * teacher_loss
        weighted_student_loss = student_weight * student_loss
        
        total_loss = lm_loss + detection_loss_tensor + weighted_teacher_loss + weighted_student_loss
        
        # 5. Update current loss components
        self._update_current_losses(lm_loss, teacher_loss, student_loss, detection_components)
        
        # 6. Accumulate losses if training
        if is_training:
            self._accumulate_losses()
        
        # 7. Prepare loss components dictionary
        loss_components = {
            "lm_loss": self._current_lm_loss,
            "teacher_lm_loss": self._current_teacher_lm_loss,
            "student_lm_loss": self._current_student_lm_loss,
            **detection_components
        }
        
        return total_loss, loss_components
    
    def _compute_language_modeling_loss(
        self, 
        model_outputs: Any, 
        inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute language modeling loss with NaN detection."""
        lm_loss = model_outputs.loss
        
        # NaN detection and logging
        if torch.isnan(lm_loss):
            self.logger.error("❌ NaN detected in language modeling loss!")
            # Return a small positive loss to continue training
            lm_loss = torch.tensor(1e-6, device=lm_loss.device, requires_grad=True)
        
        return lm_loss
    
    def _compute_teacher_student_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Compute separate losses for teacher and student spans.
        
        Args:
            logits: Model output logits (B, S, V)
            labels: Target labels (B, S)
            inputs: Batch inputs containing span information
            
        Returns:
            Tuple of (teacher_loss, student_loss) as float values
        """
        if logits is None or labels is None:
            return 0.0, 0.0
        
        # Get teacher and student spans from inputs
        teacher_spans = inputs.get("teacher_assistant_spans", [])
        student_spans = inputs.get("student_assistant_spans", [])
        
        if not teacher_spans and not student_spans:
            return 0.0, 0.0
        
        # Apply shifting for next-token prediction (align logits and labels)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size, shifted_seq_len, vocab_size = shift_logits.shape
        
        # Flatten tensors for easier indexing
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # Collect indices for teacher and student spans
        teacher_indices = []
        student_indices = []
        
        for batch_idx in range(batch_size):
            # Process teacher spans
            if batch_idx < len(teacher_spans):
                for start, end in teacher_spans[batch_idx]:
                    for pos in range(start, end):
                        shifted_pos = pos - 1  # Adjust for shifting
                        if 0 <= shifted_pos < shifted_seq_len:
                            flat_idx = batch_idx * shifted_seq_len + shifted_pos
                            if flat_labels[flat_idx] != IGNORE_INDEX:
                                teacher_indices.append(flat_idx)
            
            # Process student spans
            if batch_idx < len(student_spans):
                for start, end in student_spans[batch_idx]:
                    for pos in range(start, end):
                        shifted_pos = pos - 1  # Adjust for shifting
                        if 0 <= shifted_pos < shifted_seq_len:
                            flat_idx = batch_idx * shifted_seq_len + shifted_pos
                            if flat_labels[flat_idx] != IGNORE_INDEX:
                                student_indices.append(flat_idx)
        
        # Compute separate losses AS TENSORS (maintain gradients!)
        teacher_loss_tensor = torch.tensor(0.0, device=logits.device, requires_grad=True)
        student_loss_tensor = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        if teacher_indices:
            teacher_indices = torch.tensor(teacher_indices, device=logits.device)
            teacher_logits = flat_logits[teacher_indices]
            teacher_labels = flat_labels[teacher_indices]
            teacher_loss_tensor = F.cross_entropy(teacher_logits, teacher_labels, reduction="mean")
        
        if student_indices:
            student_indices = torch.tensor(student_indices, device=logits.device)
            student_logits = flat_logits[student_indices]
            student_labels = flat_labels[student_indices]
            student_loss_tensor = F.cross_entropy(student_logits, student_labels, reduction="mean")
        
        # Track student performance with/without teacher for analysis
        self._track_student_performance_by_teacher(
            teacher_spans, student_spans, teacher_loss_tensor, student_loss_tensor
        )
        
        return teacher_loss_tensor, student_loss_tensor
    
    def _track_student_performance_by_teacher(
        self,
        teacher_spans: List[List[Tuple[int, int]]],
        student_spans: List[List[Tuple[int, int]]],
        teacher_loss: torch.Tensor,
        student_loss: torch.Tensor
    ):
        """
        Track student performance separately for samples with/without teachers.
        
        This helps analyze whether teacher guidance is improving student learning.
        """
        # Only track when student spans are present
        if not student_spans or len(student_spans) == 0:
            return
        
        # Check if teachers are present in this batch
        has_teacher = bool(teacher_spans and any(spans for spans in teacher_spans))
        
        # Convert loss tensors to float for tracking (detach from graph for logging)
        student_loss_value = student_loss.detach().item() if student_loss.requires_grad else student_loss.item()
        
        # Update statistics
        stats = self._teacher_performance_stats
        if has_teacher:
            stats['student_with_teacher_count'] += 1
            stats['student_with_teacher_loss_sum'] += student_loss_value
        else:
            stats['student_without_teacher_count'] += 1
            stats['student_without_teacher_loss_sum'] += student_loss_value
        
        # Log comparison periodically
        total_tracked = stats['student_with_teacher_count'] + stats['student_without_teacher_count']
        if total_tracked > 0 and total_tracked % stats['log_interval'] == 0:
            # Calculate average losses
            avg_with_teacher = (
                stats['student_with_teacher_loss_sum'] / stats['student_with_teacher_count']
                if stats['student_with_teacher_count'] > 0 else 0.0
            )
            avg_without_teacher = (
                stats['student_without_teacher_loss_sum'] / stats['student_without_teacher_count']
                if stats['student_without_teacher_count'] > 0 else 0.0
            )
            
            # Calculate teacher effectiveness
            improvement = (avg_without_teacher - avg_with_teacher) if avg_without_teacher > 0 else 0.0
            improvement_pct = (improvement / avg_without_teacher * 100) if avg_without_teacher > 0 else 0.0
            
            self.logger.info(
                f"📊 Teacher Effectiveness Analysis (after {total_tracked} teacher-student batches):"
            )
            self.logger.info(
                f"   🎯 Student loss WITH teacher: {avg_with_teacher:.4f} "
                f"({stats['student_with_teacher_count']} batches)"
            )
            self.logger.info(
                f"   🎯 Student loss WITHOUT teacher: {avg_without_teacher:.4f} "
                f"({stats['student_without_teacher_count']} batches)"
            )
            if improvement > 0:
                self.logger.info(
                    f"   ✅ Teacher improves student loss by {improvement:.4f} ({improvement_pct:.1f}%)"
                )
            elif improvement < 0:
                self.logger.info(
                    f"   ⚠️  Teacher guidance shows higher loss by {abs(improvement):.4f} ({abs(improvement_pct):.1f}%)"
                )
            else:
                self.logger.info("   ➖ No significant difference observed")
    
    def _compute_detection_loss(
        self,
        model_outputs: Any,
        inputs: Dict[str, Any],
        ground_truth_objects: List[List[GroundTruthObject]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute detection loss using the detection head."""
        if self.detection_loss is None:
            return torch.tensor(0.0), {}
        
        # Get vision features from model outputs
        vision_feats = self._extract_vision_features(model_outputs, inputs)
        
        # Run detection head
        detection_outputs = model_outputs.model.detection_head(
            hidden_states=model_outputs.hidden_states[-1],
            attention_mask=inputs.get("attention_mask"),
            vision_feats=vision_feats,
            ground_truth_objects=ground_truth_objects
        )
        
        # Compute detection loss
        detection_loss_dict = self.detection_loss(detection_outputs, ground_truth_objects)
        
        # Extract individual components
        detection_components = {
            "bbox_l1_loss": detection_loss_dict.get("bbox_l1_loss", 0.0),
            "bbox_giou_loss": detection_loss_dict.get("bbox_giou_loss", 0.0),
            "caption_loss": detection_loss_dict.get("caption_loss", 0.0),
            "objectness_loss": detection_loss_dict.get("objectness_loss", 0.0),
        }
        
        return detection_loss_dict["total_loss"], detection_components
    
    def _extract_vision_features(
        self, 
        model_outputs: Any, 
        inputs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Extract vision features from model outputs for detection head."""
        # This is a simplified version - in practice, you'd extract 
        # appropriate vision features from the model's vision encoder
        # For now, return None to indicate no vision features
        return None
    
    def _extract_ground_truth_objects(
        self, 
        inputs: Dict[str, Any]
    ) -> List[List[GroundTruthObject]]:
        """Extract ground truth objects from batch inputs."""
        return inputs.get("ground_truth_objects", [])
    
    def _update_current_losses(
        self,
        lm_loss: torch.Tensor,
        teacher_loss: torch.Tensor,
        student_loss: torch.Tensor,
        detection_components: Dict[str, float]
    ):
        """Update current loss components."""
        self._current_lm_loss = lm_loss.item()
        self._current_teacher_lm_loss = teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else float(teacher_loss)
        self._current_student_lm_loss = student_loss.item() if isinstance(student_loss, torch.Tensor) else float(student_loss)
        
        # Update detection losses
        self._current_bbox_l1_loss = detection_components.get("bbox_l1_loss", 0.0)
        self._current_bbox_giou_loss = detection_components.get("bbox_giou_loss", 0.0)
        self._current_caption_loss = detection_components.get("caption_loss", 0.0)
        self._current_objectness_loss = detection_components.get("objectness_loss", 0.0)
        self._current_bbox_loss = self._current_bbox_l1_loss + self._current_bbox_giou_loss
    
    def _accumulate_losses(self):
        """Accumulate current losses to accumulators."""
        self._accumulated_lm_loss += self._current_lm_loss
        self._accumulated_teacher_lm_loss += self._current_teacher_lm_loss
        self._accumulated_student_lm_loss += self._current_student_lm_loss
        self._accumulated_bbox_l1_loss += self._current_bbox_l1_loss
        self._accumulated_bbox_giou_loss += self._current_bbox_giou_loss
        self._accumulated_caption_loss += self._current_caption_loss
        self._accumulated_objectness_loss += self._current_objectness_loss
    
    def get_averaged_losses(self) -> Dict[str, float]:
        """
        Get averaged loss components and reset accumulators.
        
        Returns:
            Dictionary of averaged loss components
        """
        num_micro_batches = max(1, self._micro_batch_count)
        
        # Compute averages
        averaged_losses = {
            "lm_loss": self._accumulated_lm_loss / num_micro_batches,
            "teacher_lm_loss": self._accumulated_teacher_lm_loss / num_micro_batches,
            "student_lm_loss": self._accumulated_student_lm_loss / num_micro_batches,
            "bbox_l1_loss": self._accumulated_bbox_l1_loss / num_micro_batches,
            "bbox_giou_loss": self._accumulated_bbox_giou_loss / num_micro_batches,
            "caption_loss": self._accumulated_caption_loss / num_micro_batches,
            "objectness_loss": self._accumulated_objectness_loss / num_micro_batches,
        }
        
        # Add derived loss components
        averaged_losses["bbox_loss"] = (
            averaged_losses["bbox_l1_loss"] + averaged_losses["bbox_giou_loss"]
        )
        averaged_losses["total_loss"] = (
            averaged_losses["lm_loss"] 
            + averaged_losses["bbox_loss"]
            + averaged_losses["caption_loss"] 
            + averaged_losses["objectness_loss"]
        )
        
        # Reset accumulators
        self._reset_accumulators()
        
        return averaged_losses
    
    def _reset_accumulators(self):
        """Reset all loss accumulators."""
        self._accumulated_lm_loss = 0.0
        self._accumulated_teacher_lm_loss = 0.0
        self._accumulated_student_lm_loss = 0.0
        self._accumulated_bbox_l1_loss = 0.0
        self._accumulated_bbox_giou_loss = 0.0
        self._accumulated_caption_loss = 0.0
        self._accumulated_objectness_loss = 0.0
        self._micro_batch_count = 0
    
    def save_training_state(self) -> Dict[str, Any]:
        """Save current training state for evaluation isolation."""
        return {
            "accumulated_lm_loss": self._accumulated_lm_loss,
            "accumulated_teacher_lm_loss": self._accumulated_teacher_lm_loss,
            "accumulated_student_lm_loss": self._accumulated_student_lm_loss,
            "accumulated_bbox_l1_loss": self._accumulated_bbox_l1_loss,
            "accumulated_bbox_giou_loss": self._accumulated_bbox_giou_loss,
            "accumulated_caption_loss": self._accumulated_caption_loss,
            "accumulated_objectness_loss": self._accumulated_objectness_loss,
            "micro_batch_count": self._micro_batch_count,
        }
    
    def restore_training_state(self, state: Dict[str, Any]):
        """Restore training state after evaluation."""
        self._accumulated_lm_loss = state["accumulated_lm_loss"]
        self._accumulated_teacher_lm_loss = state["accumulated_teacher_lm_loss"]
        self._accumulated_student_lm_loss = state["accumulated_student_lm_loss"]
        self._accumulated_bbox_l1_loss = state["accumulated_bbox_l1_loss"]
        self._accumulated_bbox_giou_loss = state["accumulated_bbox_giou_loss"]
        self._accumulated_caption_loss = state["accumulated_caption_loss"]
        self._accumulated_objectness_loss = state["accumulated_objectness_loss"]
        self._micro_batch_count = state["micro_batch_count"]
    
    def get_current_losses(self) -> Dict[str, float]:
        """Get current loss components (single forward pass)."""
        return {
            "lm_loss": self._current_lm_loss,
            "teacher_lm_loss": self._current_teacher_lm_loss,
            "student_lm_loss": self._current_student_lm_loss,
            "bbox_l1_loss": self._current_bbox_l1_loss,
            "bbox_giou_loss": self._current_bbox_giou_loss,
            "caption_loss": self._current_caption_loss,
            "objectness_loss": self._current_objectness_loss,
            "bbox_loss": self._current_bbox_loss,
        }