import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from typing import Any, Dict, List, Tuple, Union

from transformers import PreTrainedTokenizerBase

from src.logger_utils import get_detection_logger
from src.schema import (
    DetectionPredictions,
    LossDictType,
)  # typing
from src.utils import IGNORE_INDEX

logger = get_detection_logger()


class DetectionLoss(nn.Module):
    """
    Detection loss for open vocabulary dense object captioning.

    Implements Hungarian matching with multi-task loss combining:
    - Bounding box regression (L1 + GIoU)
    - Object presence classification
    - Caption generation (language modeling)
    """

    def __init__(
        self,
        bbox_weight: float,
        giou_weight: float,
        objectness_weight: float,
        caption_weight: float,
        focal_loss_gamma: float,
        focal_loss_alpha: float,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.objectness_weight = objectness_weight
        self.caption_weight = caption_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        # Tokenizer for caption processing
        self.tokenizer = tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is required for caption loss computation")
        self.ignore_index = IGNORE_INDEX

    def forward(
        self,
        pred_outputs: Union[DetectionPredictions, Dict[str, torch.Tensor]],
        ground_truth_objects: List[List[Dict[str, Any]]],
    ) -> LossDictType:
        """
        Compute detection loss using Hungarian matching.

        Args:
            pred_outputs: {
                "pred_boxes": (B, N, 4),           # Normalized [0,1] coordinates
                "pred_objectness": (B, N),          # Object confidence scores (pre-sigmoid)
                "caption_logits": (B, N, max_len, vocab_size)  # Caption token logits
            }
            ground_truth_objects: List[List[Dict]] - per batch, per object

        Returns:
            dict: Detailed detection loss components
        """
        if isinstance(pred_outputs, DetectionPredictions):
            preds = pred_outputs
        else:
            # Legacy dict – wrap into dataclass for validation
            preds = DetectionPredictions(**pred_outputs)

        pred_boxes = preds.pred_boxes
        pred_objectness = preds.pred_objectness
        caption_logits = preds.caption_logits

        batch_size = pred_boxes.shape[0]
        num_gt_total = 0
        device = pred_boxes.device

        # Wrapper to upcast inputs to float32 for stable loss computation
        def upcast_and_execute(loss_fn, *args):
            # Move to a new function to ensure memory is released after execution
            def execute(*casted_args):
                return loss_fn(*casted_args)

            # Only upcast tensor arguments, leave others untouched
            casted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    casted_args.append(arg.float())
                else:
                    casted_args.append(arg)

            return execute(*casted_args)

        def _compute_caption_loss(logits, targets):
            """Compute cross-entropy loss with the correct dimensions."""
            # Ensure targets are Long type for cross-entropy
            if targets.dtype != torch.long:
                targets = targets.long()
            return F.cross_entropy(
                logits.permute(0, 2, 1),  # (N, vocab_size, max_len)
                targets,  # (N, max_len)
                ignore_index=self.ignore_index,
            )

        # Track loss components for debugging (separate bbox components)
        total_l1_loss = 0.0
        total_giou_loss = 0.0
        total_caption_loss = 0.0
        total_objectness_loss = 0.0

        for b in range(batch_size):
            gt_objects_sample = ground_truth_objects[b]
            num_gt_sample = len(gt_objects_sample)
            num_gt_total += num_gt_sample
            num_queries = pred_boxes.shape[1]
            max_caption_len = caption_logits.shape[2]

            # Hungarian matching for this sample
            matched_pred_idx, matched_gt_idx = self._hungarian_match(
                pred_boxes[b],  # (N, 4)
                pred_objectness[b],  # (N,)
                gt_objects_sample,  # List[Dict]
            )

            # --- Caption Loss (for all queries) ---
            # Create target tensor for all queries, init with ignore_index
            target_captions = torch.full(
                (num_queries, max_caption_len),
                self.ignore_index,
                dtype=torch.long,
                device=device,
            )

            # 1. Targets for matched queries (ground truth descriptions)
            if len(matched_gt_idx) > 0:
                gt_texts = [gt_objects_sample[i]["desc"] for i in matched_gt_idx]
                tokenized_captions = self.tokenizer(
                    gt_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_caption_len,
                    return_tensors="pt",
                ).input_ids.to(device)

                target_captions[matched_pred_idx] = tokenized_captions

            # 2. Target for unmatched queries ("no object" / EOS token)
            unmatched_pred_idx = [
                i for i in range(num_queries) if i not in matched_pred_idx
            ]
            if len(unmatched_pred_idx) > 0:
                target_captions[unmatched_pred_idx, 0] = self.tokenizer.eos_token_id

            # Compute cross-entropy loss for all queries in float32
            caption_loss = upcast_and_execute(
                _compute_caption_loss, caption_logits[b], target_captions
            )
            total_caption_loss += caption_loss
            # --- End Caption Loss ---

            # Bbox regression loss for matched objects
            if len(matched_pred_idx) > 0:
                l1_loss_val, giou_loss_val = upcast_and_execute(
                    self._bbox_loss,
                    pred_boxes[b][matched_pred_idx],
                    gt_objects_sample,
                    matched_gt_idx,
                )
                total_l1_loss += l1_loss_val
                total_giou_loss += giou_loss_val

            # Objectness loss for all predictions
            objectness_loss = upcast_and_execute(
                self._objectness_loss,
                pred_objectness[b],  # (N,)
                matched_pred_idx,  # Indices of matched predictions
            )
            total_objectness_loss += objectness_loss

        # ------------------------------------------------------------------
        # Loss normalisation strategy
        # Stabler normalisation: divide by (num_gt + ε·num_queries) so images
        # with few objects are not overweighted and zero-GT images are still
        # well-defined.
        epsilon = 1e-3
        denom = num_gt_total + epsilon * batch_size * pred_boxes.shape[1]

        final_l1_loss = total_l1_loss / denom
        final_giou_loss = total_giou_loss / denom

        # Caption loss – average across batch (caption_loss is already token &
        # query averaged inside _compute_caption_loss).
        final_caption_loss = (
            total_caption_loss / batch_size if batch_size > 0 else total_caption_loss
        )

        # Objectness loss is averaged by batch size as before
        final_objectness_loss = (
            total_objectness_loss / batch_size
            if batch_size > 0
            else total_objectness_loss
        )

        # Apply final weights
        weighted_bbox_loss = (
            self.bbox_weight * final_l1_loss + self.giou_weight * final_giou_loss
        )
        weighted_caption_loss = self.caption_weight * final_caption_loss
        weighted_objectness_loss = self.objectness_weight * final_objectness_loss

        # Total loss for backpropagation
        final_loss = (
            weighted_bbox_loss + weighted_caption_loss + weighted_objectness_loss
        )

        # Return detailed loss components for enhanced logging
        loss_components = {
            "total_loss": final_loss,
            "bbox_loss": weighted_bbox_loss,
            "bbox_l1_loss": final_l1_loss,
            "bbox_giou_loss": final_giou_loss,
            "caption_loss": weighted_caption_loss,
            "objectness_loss": weighted_objectness_loss,
        }

        # Store individual loss components for backward compatibility (unweighted)
        self.last_bbox_loss = final_l1_loss + final_giou_loss
        self.last_l1_loss = final_l1_loss
        self.last_giou_loss = final_giou_loss
        self.last_caption_loss = final_caption_loss
        self.last_objectness_loss = final_objectness_loss

        # Debug logging every few steps
        if hasattr(self, "_debug_counter"):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 1 == 0:
            logger.debug(f"🔍 DETECTION LOSS DEBUG:")
            logger.debug(f"   Bbox loss: {final_l1_loss + final_giou_loss:.6f}")
            logger.debug(f"   Caption loss: {final_caption_loss:.6f}")
            logger.debug(f"   Objectness loss: {final_objectness_loss:.6f}")
            logger.debug(
                f"   Final loss: {final_loss.item() if hasattr(final_loss, 'item') else final_loss:.6f}"
            )
            logger.debug(
                f"   Weights: bbox={self.bbox_weight}, giou={self.giou_weight}, caption={self.caption_weight}, objectness={self.objectness_weight}"
            )

            # Add debugging for predictions vs ground truth
            if len(ground_truth_objects) > 0 and len(ground_truth_objects[0]) > 0:
                # Show first batch, first GT object
                gt_box = ground_truth_objects[0][0]["box"]
                pred_box = pred_boxes[0, 0].detach().cpu().tolist()
                pred_obj = torch.sigmoid(pred_objectness[0, 0]).item()

                logger.debug(f"   Sample GT box: {gt_box}")
                logger.debug(
                    f"   Sample pred box: {pred_box}"
                )  # Show actual float values
                logger.debug(f"   Sample pred objectness: {pred_obj:.3f}")
                logger.debug(
                    f"   Num GT objects in batch: {[len(gt) for gt in ground_truth_objects]}"
                )

        return loss_components

    def _hungarian_match(
        self,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
        gt_objects: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int]]:
        """Hungarian matching using bbox + objectness costs"""
        if len(gt_objects) == 0:
            return [], []

        # Convert GT to tensors - use "box" key from your data format
        gt_boxes = torch.stack(
            [self._box_to_tensor(obj["box"], pred_boxes.device) for obj in gt_objects]
        )  # (M, 4)

        # GT boxes are in pixel coordinates, need to normalize them
        # Note: Your ground truth extraction should handle this normalization
        # For now, assume they're already normalized by extract_ground_truth_from_sample

        # Cost matrix computation
        N, M = pred_boxes.shape[0], gt_boxes.shape[0]
        cost_matrix = torch.zeros(N, M, device=pred_boxes.device)

        # Use weights for cost calculation to balance components
        bbox_weight = self.bbox_weight
        giou_weight = self.giou_weight
        objectness_weight = self.objectness_weight

        for i in range(N):
            for j in range(M):
                # Bbox cost (L1 + GIoU)
                l1_cost = F.l1_loss(pred_boxes[i], gt_boxes[j], reduction="sum")
                giou_cost = (
                    1
                    - self._compute_giou(
                        pred_boxes[i : i + 1], gt_boxes[j : j + 1]
                    ).squeeze()
                )

                # Objectness cost (encourage high confidence for matched objects)
                # Apply sigmoid here since detection head outputs raw logits
                objectness_cost = 1 - torch.sigmoid(pred_objectness[i])

                # Weighted sum of costs
                cost_matrix[i, j] = (
                    bbox_weight * l1_cost
                    + giou_weight * giou_cost
                    + objectness_weight * objectness_cost
                )

        # Hungarian assignment
        if SCIPY_AVAILABLE:
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            return pred_idx.tolist(), gt_idx.tolist()
        else:
            raise ValueError("scipy is not available, please install it")

    def _bbox_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_objects: List[Dict[str, Any]],
        gt_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return separate L1 and GIoU losses"""
        gt_boxes = torch.stack(
            [
                self._box_to_tensor(gt_objects[i]["box"], pred_boxes.device)
                for i in gt_indices
            ]
        )

        # L1 regression loss
        l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")

        # GIoU loss (converted to positive loss)
        giou_loss = (1 - torch.diag(self._compute_giou(pred_boxes, gt_boxes))).mean()

        return l1_loss, giou_loss

    def _objectness_loss(
        self,
        pred_objectness: torch.Tensor,
        matched_pred_idx: List[int],
    ) -> torch.Tensor:
        """
        Focal loss for object presence to handle class imbalance.
        """
        num_queries = pred_objectness.shape[0]
        device = pred_objectness.device

        # Create target tensor: 1 for matched queries, 0 for others
        target_objectness = torch.zeros(num_queries, device=device)
        if len(matched_pred_idx) > 0:
            target_objectness[matched_pred_idx] = 1.0

        # Compute BCE loss without reduction to apply focal scaling
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_objectness, target_objectness, reduction="none"
        )

        # Compute probabilities and p_t for focal loss
        p = torch.sigmoid(pred_objectness)
        p_t = p * target_objectness + (1 - p) * (1 - target_objectness)
        modulating_factor = (1.0 - p_t) ** self.focal_loss_gamma

        # Compute alpha factor
        alpha_t = self.focal_loss_alpha * target_objectness + (
            1 - self.focal_loss_alpha
        ) * (1 - target_objectness)

        focal_loss = alpha_t * modulating_factor * bce_loss

        return focal_loss.mean()

    def _compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized Intersection over Union (GIoU) between two sets of boxes.
        """
        # boxes format: [x1, y1, x2, y2] normalized to [0,1]

        # Ensure boxes are valid (x1 < x2, y1 < y2)
        boxes1 = torch.stack(
            [
                torch.min(boxes1[:, 0], boxes1[:, 2]),  # x1
                torch.min(boxes1[:, 1], boxes1[:, 3]),  # y1
                torch.max(boxes1[:, 0], boxes1[:, 2]),  # x2
                torch.max(boxes1[:, 1], boxes1[:, 3]),  # y2
            ],
            dim=1,
        )

        boxes2 = torch.stack(
            [
                torch.min(boxes2[:, 0], boxes2[:, 2]),  # x1
                torch.min(boxes2[:, 1], boxes2[:, 3]),  # y1
                torch.max(boxes2[:, 0], boxes2[:, 2]),  # x2
                torch.max(boxes2[:, 1], boxes2[:, 3]),  # y2
            ],
            dim=1,
        )

        # Calculate intersection
        x1_inter = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1_inter = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2_inter = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2_inter = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(
            y2_inter - y1_inter, min=0
        )

        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area

        # IoU with numerical stability
        iou = inter_area / torch.clamp(union_area, min=1e-7)

        # Enclosing box for GIoU
        x1_enc = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1_enc = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2_enc = torch.max(boxes1[:, 2], boxes2[:, 2])
        y2_enc = torch.max(boxes1[:, 3], boxes2[:, 3])

        enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

        # GIoU with numerical stability and proper bounds
        giou = iou - (enc_area - union_area) / torch.clamp(enc_area, min=1e-7)

        # Clamp GIoU to valid range [-1, 1]
        giou = torch.clamp(giou, min=-1.0, max=1.0)

        return giou

    def _box_to_tensor(self, box: Any, device: torch.device) -> torch.Tensor:
        """Convert a box (list/tuple/Tensor) to a float32 Tensor on *device*.

        This utility prevents the common ``torch.tensor(existing_tensor)`` anti-pattern
        that triggers the *copy construct* warning from PyTorch by forwarding
        Tensors via ``.to`` instead of re-wrapping them.
        """
        if isinstance(box, torch.Tensor):
            # Only device / dtype cast – no new allocation if already correct.
            return box.to(device=device, dtype=torch.float32)
        # Assume sequence of 4 coordinates; let any error surfacing here be explicit.
        return torch.tensor(box, dtype=torch.float32, device=device)
