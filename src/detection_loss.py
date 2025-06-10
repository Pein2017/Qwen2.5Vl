import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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
        bbox_weight=5.0,
        giou_weight=2.0,
        objectness_weight=1.0,
        caption_weight=0.1,
        tokenizer=None,
    ):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.objectness_weight = objectness_weight
        self.caption_weight = caption_weight

        # Tokenizer for caption processing
        self.tokenizer = tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is required for caption loss computation")

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
        total_loss = 0.0

        # Track loss components for debugging
        total_bbox_loss = 0.0
        total_caption_loss = 0.0
        total_objectness_loss = 0.0

        for b in range(batch_size):
            # Hungarian matching for this sample
            matched_pred_idx, matched_gt_idx = self._hungarian_match(
                pred_boxes[b],  # (N, 4)
                pred_objectness[b],  # (N,)
                ground_truth_objects[b],  # List[Dict]
            )

            # Bbox regression loss for matched objects
            bbox_loss = 0.0
            caption_loss = 0.0
            if len(matched_pred_idx) > 0:
                bbox_loss = self._bbox_loss(
                    pred_boxes[b][matched_pred_idx],
                    ground_truth_objects[b],
                    matched_gt_idx,
                )

                # Caption generation loss for matched objects
                caption_loss = self._caption_loss(
                    caption_logits[b][
                        matched_pred_idx
                    ],  # (num_matched, max_len, vocab_size)
                    ground_truth_objects[b],
                    matched_gt_idx,
                )

                # Ensure losses are non-negative
                bbox_loss = torch.clamp(bbox_loss, min=0.0)
                caption_loss = torch.clamp(caption_loss, min=0.0)

                # Accumulate unweighted losses for individual tracking
                total_bbox_loss += (
                    bbox_loss.item() if hasattr(bbox_loss, "item") else bbox_loss
                )
                total_caption_loss += (
                    caption_loss.item()
                    if hasattr(caption_loss, "item")
                    else caption_loss
                )

                # Add weighted losses to total (for backward compatibility)
                sample_loss = (
                    self.bbox_weight * bbox_loss + self.caption_weight * caption_loss
                )
                total_loss += sample_loss

            # Objectness loss for all predictions
            objectness_loss = self._objectness_loss(
                pred_objectness[b],  # (N,)
                matched_pred_idx,  # Indices of matched predictions
                len(ground_truth_objects[b]),  # Number of GT objects
            )

            # Ensure objectness loss is non-negative
            objectness_loss = torch.clamp(objectness_loss, min=0.0)

            total_loss += self.objectness_weight * objectness_loss
            total_objectness_loss += (
                objectness_loss.item()
                if hasattr(objectness_loss, "item")
                else objectness_loss
            )

        # Average over batch
        final_loss = total_loss / batch_size if batch_size > 0 else total_loss

        # Ensure final loss is non-negative
        final_loss = torch.clamp(final_loss, min=0.0)

        # Return detailed loss components for enhanced logging (already weighted)
        loss_components = {
            "total_loss": final_loss,
            "bbox_loss": (total_bbox_loss / batch_size * self.bbox_weight)
            if batch_size > 0
            else 0.0,
            "caption_loss": (total_caption_loss / batch_size * self.caption_weight)
            if batch_size > 0
            else 0.0,
            "objectness_loss": (
                total_objectness_loss / batch_size * self.objectness_weight
            )
            if batch_size > 0
            else 0.0,
        }

        # Store individual loss components for backward compatibility (weighted)
        self.last_bbox_loss = loss_components["bbox_loss"]
        self.last_caption_loss = loss_components["caption_loss"]
        self.last_objectness_loss = loss_components["objectness_loss"]

        # Debug logging every few steps
        if hasattr(self, "_debug_counter"):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 1 == 0:
            print(f"🔍 DETECTION LOSS DEBUG:")
            print(f"   Bbox loss (weighted): {loss_components['bbox_loss']:.6f}")
            print(f"   Caption loss (weighted): {loss_components['caption_loss']:.6f}")
            print(
                f"   Objectness loss (weighted): {loss_components['objectness_loss']:.6f}"
            )
            print(
                f"   Final loss: {final_loss.item() if hasattr(final_loss, 'item') else final_loss:.6f}"
            )
            print(
                f"   Weights: bbox={self.bbox_weight}, caption={self.caption_weight}, objectness={self.objectness_weight}"
            )

            # Add debugging for predictions vs ground truth
            if len(ground_truth_objects) > 0 and len(ground_truth_objects[0]) > 0:
                # Show first batch, first GT object
                gt_box = ground_truth_objects[0][0]["box"]
                pred_box = pred_outputs["pred_boxes"][0, 0].detach().cpu().tolist()
                pred_obj = torch.sigmoid(pred_outputs["pred_objectness"][0, 0]).item()

                print(f"   Sample GT box: {gt_box}")
                print(f"   Sample pred box: {pred_box}")  # Show actual float values
                print(f"   Sample pred objectness: {pred_obj:.3f}")
                print(
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

        for i in range(N):
            for j in range(M):
                # Bbox cost (L1 + GIoU) - reduced scaling
                bbox_cost = F.l1_loss(pred_boxes[i], gt_boxes[j], reduction="sum")
                giou_cost = (
                    1
                    - self._compute_giou(
                        pred_boxes[i : i + 1], gt_boxes[j : j + 1]
                    ).item()
                )

                # Objectness cost (encourage high confidence for matched objects)
                # Apply sigmoid here since detection head outputs raw logits
                objectness_cost = (1 - torch.sigmoid(pred_objectness[i])).item()

                cost_matrix[i, j] = bbox_cost + giou_cost + objectness_cost

        # Hungarian assignment
        if SCIPY_AVAILABLE:
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            return pred_idx.tolist(), gt_idx.tolist()
        else:
            raise ValueError("scipy is not available, please install it")

    def _bbox_loss(self, pred_boxes, gt_objects, gt_indices):
        """L1 + GIoU loss for matched boxes"""
        # Extract normalized boxes from GT - use "box" key from your data format
        gt_boxes = torch.stack(
            [
                torch.tensor(
                    gt_objects[i]["box"], dtype=torch.float32, device=pred_boxes.device
                )
                for i in gt_indices
            ]
        )

        # GT boxes should already be normalized by extract_ground_truth_from_sample

        # L1 loss (always positive)
        l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")

        # GIoU loss (ensure positive)
        giou = self._compute_giou(pred_boxes, gt_boxes)
        giou_loss = (1 - giou).mean()

        # Ensure GIoU loss is positive (since GIoU can be negative, 1-GIoU can be > 2)
        giou_loss = torch.clamp(giou_loss, min=0.0, max=2.0)

        total_bbox_loss = l1_loss + giou_loss

        # Final safeguard
        total_bbox_loss = torch.clamp(total_bbox_loss, min=0.0)

        return total_bbox_loss

    def _objectness_loss(self, pred_objectness, matched_pred_idx, num_gt_objects):
        """Binary classification loss for object presence"""
        # Create objectness targets
        objectness_targets = torch.zeros_like(pred_objectness)
        if len(matched_pred_idx) > 0:
            objectness_targets[matched_pred_idx] = 1.0

        # Binary cross entropy loss with logits (detection head outputs raw logits)
        return F.binary_cross_entropy_with_logits(pred_objectness, objectness_targets)

    def _caption_loss(self, caption_logits, gt_objects, gt_indices):
        """Improved language modeling loss for caption generation with proper next-token prediction"""
        # caption_logits: (num_matched, max_len, vocab_size)
        # gt_objects: List[Dict] with "desc" field from your data format
        # gt_indices: indices into gt_objects for matched predictions

        if len(gt_indices) == 0:
            return torch.tensor(0.0, device=caption_logits.device)

        total_caption_loss = 0.0
        num_valid_captions = 0

        for i, gt_idx in enumerate(gt_indices):
            # Get ground truth description - use "desc" key from your data format
            gt_description = gt_objects[gt_idx]["desc"]

            # Tokenize the description
            gt_tokens = self.tokenizer.encode(
                gt_description, add_special_tokens=False, return_tensors="pt"
            ).to(caption_logits.device)  # (1, seq_len)

            # Get prediction logits for this object
            pred_logits = caption_logits[i]  # (max_len, vocab_size)

            # Compute next-token prediction loss
            if gt_tokens.shape[1] > 0:
                max_len = pred_logits.shape[0]
                gt_seq_len = min(gt_tokens.shape[1], max_len)

                if gt_seq_len > 0:
                    if gt_seq_len > 1:
                        # Use logits from positions 0 to seq_len-2 to predict tokens 1 to seq_len-1
                        pred_logits_for_loss = pred_logits[
                            : gt_seq_len - 1
                        ]  # (seq_len-1, vocab_size)
                        target_tokens = gt_tokens[0, 1:gt_seq_len]  # (seq_len-1,)

                        caption_loss = F.cross_entropy(
                            pred_logits_for_loss,
                            target_tokens,
                            reduction="mean",
                        )

                        # Ensure caption loss is non-negative
                        caption_loss = torch.clamp(caption_loss, min=0.0)

                        total_caption_loss += caption_loss
                        num_valid_captions += 1
                    else:
                        # Single token case - predict the first token
                        caption_loss = F.cross_entropy(
                            pred_logits[0:1],  # (1, vocab_size)
                            gt_tokens[0, 0:1],  # (1,)
                            reduction="mean",
                        )

                        # Ensure caption loss is non-negative
                        caption_loss = torch.clamp(caption_loss, min=0.0)

                        total_caption_loss += caption_loss
                        num_valid_captions += 1

        if num_valid_captions > 0:
            final_caption_loss = total_caption_loss / num_valid_captions
            # Final safeguard
            final_caption_loss = torch.clamp(final_caption_loss, min=0.0)
            return final_caption_loss
        else:
            return torch.tensor(0.0, device=caption_logits.device)

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
