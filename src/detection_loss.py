import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.logger_utils import get_detection_logger
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
        tokenizer,
    ):
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

    def forward(self, pred_outputs, ground_truth_objects):
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
        pred_boxes = pred_outputs["pred_boxes"]  # (B, N, 4)
        pred_objectness = pred_outputs["pred_objectness"]  # (B, N)
        caption_logits = pred_outputs["caption_logits"]  # (B, N, max_len, vocab_size)

        batch_size = pred_boxes.shape[0]
        num_gt_total = 0
        device = pred_boxes.device

        # Track loss components for debugging
        total_bbox_loss = 0.0
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

            # Compute cross-entropy loss for all queries
            caption_loss = F.cross_entropy(
                caption_logits[b].permute(0, 2, 1),  # (N, vocab_size, max_len)
                target_captions,  # (N, max_len)
                ignore_index=self.ignore_index,
            )
            total_caption_loss += caption_loss
            # --- End Caption Loss ---

            # Bbox regression loss for matched objects
            if len(matched_pred_idx) > 0:
                bbox_loss = self._bbox_loss(
                    pred_boxes[b][matched_pred_idx],
                    gt_objects_sample,
                    matched_gt_idx,
                )
                total_bbox_loss += bbox_loss

            # Objectness loss for all predictions
            objectness_loss = self._objectness_loss(
                pred_objectness[b],  # (N,)
                matched_pred_idx,  # Indices of matched predictions
            )
            total_objectness_loss += objectness_loss

        # Normalize by total number of ground truth objects in the batch
        # This is the standard DETR approach for stable loss scaling.
        if num_gt_total > 0:
            final_bbox_loss = total_bbox_loss / num_gt_total
            final_caption_loss = total_caption_loss / num_gt_total
        else:
            final_bbox_loss = total_bbox_loss  # Avoid division by zero
            final_caption_loss = total_caption_loss

        # Objectness loss is averaged by batch size
        final_objectness_loss = (
            total_objectness_loss / batch_size
            if batch_size > 0
            else total_objectness_loss
        )

        # Apply final weights
        weighted_bbox_loss = self.bbox_weight * final_bbox_loss
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
            "caption_loss": weighted_caption_loss,
            "objectness_loss": weighted_objectness_loss,
        }

        # Store individual loss components for backward compatibility (unweighted)
        self.last_bbox_loss = final_bbox_loss
        self.last_caption_loss = final_caption_loss
        self.last_objectness_loss = final_objectness_loss

        # Debug logging every few steps
        if hasattr(self, "_debug_counter"):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 1 == 0:
            logger.debug(f"🔍 DETECTION LOSS DEBUG:")
            logger.debug(f"   Bbox loss: {final_bbox_loss:.6f}")
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
                pred_box = pred_outputs["pred_boxes"][0, 0].detach().cpu().tolist()
                pred_obj = torch.sigmoid(pred_outputs["pred_objectness"][0, 0]).item()

                logger.debug(f"   Sample GT box: {gt_box}")
                logger.debug(
                    f"   Sample pred box: {pred_box}"
                )  # Show actual float values
                logger.debug(f"   Sample pred objectness: {pred_obj:.3f}")
                logger.debug(
                    f"   Num GT objects in batch: {[len(gt) for gt in ground_truth_objects]}"
                )

        return loss_components

    def _hungarian_match(self, pred_boxes, pred_objectness, gt_objects):
        """Hungarian matching using bbox + objectness costs"""
        if len(gt_objects) == 0:
            return [], []

        # Convert GT to tensors - use "box" key from your data format
        gt_boxes = torch.stack(
            [
                torch.tensor(obj["box"], dtype=torch.float32, device=pred_boxes.device)
                for obj in gt_objects
            ]
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

    def _bbox_loss(self, pred_boxes, gt_objects, gt_indices):
        """L1 + GIoU loss for matched boxes"""
        gt_boxes = torch.stack(
            [
                torch.tensor(
                    gt_objects[i]["box"], dtype=torch.float32, device=pred_boxes.device
                )
                for i in gt_indices
            ]
        )

        # L1 regression loss
        l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")

        # GIoU loss
        giou_loss = (
            1 - self._compute_giou(pred_boxes, gt_boxes).diag()
        ).mean()  # Use diag for matched pairs

        return l1_loss + giou_loss

    def _objectness_loss(self, pred_objectness, matched_pred_idx):
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

    def _compute_giou(self, boxes1, boxes2):
        """Compute GIoU between two sets of boxes"""
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
