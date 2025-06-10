# ! As reference only, don't delete this file
"""
Simplified Object Detection Loss for Qwen2.5-VL training.
Decoupled design focusing only on detection loss calculation.

Key Features:
- 🎯 Single Responsibility: Only handles object detection loss
- 🔄 Clean Integration: Works with standard model.forward() output
- ⚡ Memory Efficient: No embedding reuse complexity
- 📊 Robust Parsing: Handles Qwen2.5-VL special tokens and JSON fallback

Usage:
    # Initialize detection loss
    detection_loss = ObjectDetectionLoss(
        bbox_weight=0.6,
        giou_weight=0.4,
        class_weight=0.3,
        repetition_penalty=1.1
    )

    # In trainer compute_loss method:
    outputs = model(**inputs)
    lm_loss = outputs.loss

    detection_loss_value = detection_loss.compute(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        ground_truth_objects=inputs.get('ground_truth_objects')
    )

    total_loss = lm_loss + detection_loss_value
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Using greedy matching.")

# Import monitoring classes
from src.logger_utils import get_loss_logger
from src.response_parser import ResponseParser
from src.training.monitor import TokenStats, TrainingMonitor

logger = get_loss_logger()


class HungarianMatcher:
    """Hungarian matching for optimal object assignment."""

    def __init__(self, parser: ResponseParser):
        self.parser = parser

    def match(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Match predicted and ground truth objects."""
        if not pred_objects or not gt_objects:
            return [], []

        if SCIPY_AVAILABLE:
            return self._hungarian_match(pred_objects, gt_objects)
        else:
            return self._greedy_match(pred_objects, gt_objects)

    def _hungarian_match(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Hungarian algorithm matching."""
        cost_matrix = self._compute_cost_matrix(pred_objects, gt_objects)
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matched_pred = [pred_objects[i] for i in pred_indices]
        matched_gt = [gt_objects[i] for i in gt_indices]

        return matched_pred, matched_gt

    def _greedy_match(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Greedy matching as fallback."""
        matched_pred = []
        matched_gt = []
        used_gt_indices = set()

        for pred_obj in pred_objects:
            best_cost = float("inf")
            best_gt_idx = -1

            for j, gt_obj in enumerate(gt_objects):
                if j in used_gt_indices:
                    continue

                cost = self._compute_cost(pred_obj, gt_obj)
                if cost < best_cost:
                    best_cost = cost
                    best_gt_idx = j

            if best_gt_idx != -1 and best_cost < 0.8:  # Threshold for valid match
                matched_pred.append(pred_obj)
                matched_gt.append(gt_objects[best_gt_idx])
                used_gt_indices.add(best_gt_idx)

        return matched_pred, matched_gt

    def _compute_cost_matrix(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> torch.Tensor:
        """Compute cost matrix for Hungarian matching."""
        cost_matrix = torch.zeros(len(pred_objects), len(gt_objects))

        for i, pred_obj in enumerate(pred_objects):
            for j, gt_obj in enumerate(gt_objects):
                cost = self._compute_cost(pred_obj, gt_obj)
                cost_matrix[i, j] = cost

        return cost_matrix.numpy()

    def _compute_cost(self, pred_obj: Dict, gt_obj: Dict) -> float:
        """Compute matching cost between two objects."""
        # Bbox IoU cost
        pred_bbox = torch.tensor(pred_obj["bbox"], dtype=torch.float32).unsqueeze(0)
        gt_bbox = torch.tensor(gt_obj["bbox"], dtype=torch.float32).unsqueeze(0)

        iou = self._compute_iou(pred_bbox, gt_bbox).item()
        bbox_cost = 1.0 - iou

        # Semantic similarity cost
        pred_desc = pred_obj.get("description", "")
        gt_desc = gt_obj.get("description", "")
        similarity = self.parser.calculate_semantic_similarity(pred_desc, gt_desc)
        semantic_cost = 1.0 - similarity

        # Combined cost (weighted average)
        return 0.7 * bbox_cost + 0.3 * semantic_cost

    def _compute_iou(
        self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU between bounding boxes."""
        # Calculate intersection
        x1_inter = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
        y1_inter = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
        x2_inter = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
        y2_inter = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(
            y2_inter - y1_inter, min=0
        )

        # Calculate union
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (
            pred_boxes[..., 3] - pred_boxes[..., 1]
        )
        gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (
            gt_boxes[..., 3] - gt_boxes[..., 1]
        )
        union_area = pred_area + gt_area - inter_area

        # Calculate IoU
        return inter_area / (union_area + 1e-7)


class ObjectDetectionLoss(nn.Module):
    """
    Simplified Object Detection Loss for Qwen2.5-VL.

    Focuses solely on detection loss calculation with clean integration.
    """

    def __init__(
        self,
        bbox_weight: float = 0.6,
        giou_weight: float = 0.4,
        class_weight: float = 0.3,
        max_generation_length: int = 512,
        hungarian_matching: bool = True,
        enable_monitoring: bool = False,
        monitor: Optional[TrainingMonitor] = None,
        repetition_penalty: float = 1.1,
    ):
        super().__init__()

        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.class_weight = class_weight
        self.max_generation_length = max_generation_length
        self.enable_monitoring = enable_monitoring
        self.repetition_penalty = repetition_penalty

        # Initialize components
        self.parser = ResponseParser()
        self.matcher = HungarianMatcher(self.parser) if hungarian_matching else None
        self.monitor = monitor

        logger.info(f"🎯 ObjectDetectionLoss initialized:")
        logger.info(f"   bbox_weight: {bbox_weight}")
        logger.info(f"   giou_weight: {giou_weight}")
        logger.info(f"   class_weight: {class_weight}")
        logger.info(f"   hungarian_matching: {hungarian_matching}")
        logger.info(f"   monitoring_enabled: {enable_monitoring}")
        logger.info(f"   repetition_penalty: {repetition_penalty}")

    def compute(
        self,
        model: nn.Module,
        tokenizer,
        inputs: Dict,
        input_parts: Optional[List[torch.Tensor]] = None,
        ground_truth_texts: Optional[List[str]] = None,
        batch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute object detection loss.

        Args:
            model: The model instance
            tokenizer: Tokenizer for text processing
            inputs: Model inputs dictionary
            input_parts: List of input token sequences (before generation)
            ground_truth_texts: List of ground truth text responses
            batch_idx: Current batch index for monitoring

        Returns:
            Detection loss tensor
        """
        device = next(model.parameters()).device

        # Parse ground truth texts to extract objects
        if input_parts is not None and ground_truth_texts is not None:
            ground_truth_objects = []
            token_stats = TokenStats() if self.enable_monitoring else None

            for i, gt_text in enumerate(ground_truth_texts):
                logger.info(
                    f"🎯 Ground Truth Sample {i}: Text (length={len(gt_text)}):"
                )
                logger.info(
                    f"   Content: {repr(gt_text[:300])}{'...' if len(gt_text) > 300 else ''}"
                )
                gt_objects = self.parser.parse_response(gt_text, i)
                ground_truth_objects.append(gt_objects)

                # Collect token statistics for monitoring
                if self.enable_monitoring and token_stats:
                    gt_tokens = tokenizer.encode(gt_text, add_special_tokens=False)
                    gt_token_length = len(gt_tokens)

                    input_token_length = (
                        input_parts[i].size(0) if i < len(input_parts) else 0
                    )

                    token_stats.add_sample(
                        input_length=input_token_length, gt_length=gt_token_length
                    )
        else:
            # Fallback to old interface
            ground_truth_objects = inputs.get("ground_truth_objects")
            token_stats = None

        # Validate we have ground truth objects
        if ground_truth_objects is None or not ground_truth_objects:
            raise ValueError("No ground truth objects provided")

        # Generate predictions
        logger.info(f"🎯 Starting prediction generation for batch...")
        if input_parts is not None:
            logger.info(f"   Using input_parts method with {len(input_parts)} samples")
            predicted_objects_batch, predicted_texts = (
                self._generate_predictions_from_inputs(
                    model, tokenizer, inputs, input_parts, token_stats
                )
            )
        else:
            logger.info(f"   Using standard method")
            predicted_objects_batch, predicted_texts = self._generate_predictions(
                model, tokenizer, inputs, token_stats
            )

        logger.info(f"✅ Prediction generation completed:")
        logger.info(f"   Generated {len(predicted_texts)} predictions")
        logger.info(
            f"   Total predicted objects: {sum(len(objs) for objs in predicted_objects_batch)}"
        )
        logger.info(
            f"   Total ground truth objects: {sum(len(objs) for objs in ground_truth_objects)}"
        )

        # Monitor batch if enabled
        if self.enable_monitoring and self.monitor and batch_idx is not None:
            self.monitor.log_batch_analysis(
                batch_idx=batch_idx,
                predicted_objects_batch=predicted_objects_batch,
                ground_truth_objects_batch=ground_truth_objects,
                predicted_texts=predicted_texts,
                ground_truth_texts=ground_truth_texts,
                token_stats=token_stats,
                input_parts=input_parts,
            )

        # Compute losses
        return self._compute_batch_losses(
            predicted_objects_batch, ground_truth_objects, device
        )

    def _generate_predictions(
        self,
        model: nn.Module,
        tokenizer,
        inputs: Dict,
        token_stats: Optional[TokenStats] = None,
    ) -> Tuple[List[List[Dict]], List[str]]:
        """Generate predictions for the batch using proper input preparation."""
        from src.utils import prepare_inputs_for_generate

        # Set model to eval mode for inference
        original_training_mode = model.training
        model.eval()

        predicted_objects_batch = []
        predicted_texts = []

        with torch.no_grad():
            # Prepare inputs for generation (handles shape validation and extraction)
            generation_inputs, prompt_end_indices = prepare_inputs_for_generate(inputs)
            batch_size = generation_inputs["input_ids"].size(0)

            for i in range(batch_size):
                # Extract single sample from generation inputs
                sample_inputs = self._extract_sample_for_generation(
                    generation_inputs, i
                )

                # Generate response using prepared inputs
                generated_text = self._generate_single_response_prepared(
                    model,
                    tokenizer,
                    sample_inputs,
                    sample_index=i,
                    token_stats=token_stats,
                )

                predicted_texts.append(generated_text)

                # Parse response
                if generated_text:
                    predicted_objects = self.parser.parse_response(generated_text, i)
                    predicted_objects_batch.append(predicted_objects)
                else:
                    logger.warning(f"⚠️ Sample {i}: Empty generated text")
                    predicted_objects_batch.append([])

        # Restore training mode
        model.train(original_training_mode)

        return predicted_objects_batch, predicted_texts

    def _extract_sample_for_generation(
        self, generation_inputs: Dict, index: int
    ) -> Dict:
        """Extract single sample from prepared generation inputs."""
        sample = {}

        # Handle standard inputs
        for key, value in generation_inputs.items():
            if key not in ["pixel_values", "image_grid_thw", "image_counts_per_sample"]:
                if isinstance(value, torch.Tensor):
                    sample[key] = value[index : index + 1]
                elif isinstance(value, list) and len(value) > index:
                    sample[key] = [value[index]]
                else:
                    sample[key] = value

        # Handle visual inputs using image_counts_per_sample from generation_inputs
        if (
            "pixel_values" in generation_inputs
            and "image_grid_thw" in generation_inputs
        ):
            pixel_values = generation_inputs["pixel_values"]
            image_grid_thw = generation_inputs["image_grid_thw"]

            # Use image_counts_per_sample from generation_inputs
            if "image_counts_per_sample" in generation_inputs:
                image_counts = generation_inputs["image_counts_per_sample"]

                if index < len(image_counts) and image_counts[index] > 0:
                    start_idx = sum(image_counts[:index])
                    end_idx = start_idx + image_counts[index]

                    if (
                        start_idx < pixel_values.shape[0]
                        and end_idx <= pixel_values.shape[0]
                    ):
                        sample["pixel_values"] = pixel_values[start_idx:end_idx]
                        sample["image_grid_thw"] = image_grid_thw[start_idx:end_idx]
                    else:
                        sample["pixel_values"] = None
                        sample["image_grid_thw"] = None
                else:
                    sample["pixel_values"] = None
                    sample["image_grid_thw"] = None
            else:
                # Fallback: assume equal distribution
                total_images = pixel_values.shape[0] if pixel_values is not None else 0
                batch_size = generation_inputs["input_ids"].size(0)
                images_per_sample = total_images // batch_size

                if images_per_sample > 0:
                    start_idx = index * images_per_sample
                    end_idx = start_idx + images_per_sample
                    sample["pixel_values"] = pixel_values[start_idx:end_idx]
                    sample["image_grid_thw"] = image_grid_thw[start_idx:end_idx]
                else:
                    sample["pixel_values"] = None
                    sample["image_grid_thw"] = None
        else:
            sample["pixel_values"] = None
            sample["image_grid_thw"] = None

        return sample

    def _generate_single_response_prepared(
        self,
        model: nn.Module,
        tokenizer,
        sample_inputs: Dict,
        sample_index: int = 0,
        token_stats: Optional[TokenStats] = None,
    ) -> str:
        """Generate single response from prepared sample inputs."""
        # Build generation inputs following official pattern
        generation_inputs = {
            "input_ids": sample_inputs["input_ids"],
            "max_new_tokens": self.max_generation_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,  # Add repeat penalty to prevent repetitive outputs
        }

        # Add attention mask if present
        if "attention_mask" in sample_inputs:
            generation_inputs["attention_mask"] = sample_inputs["attention_mask"]

        # Add vision inputs if they exist
        if (
            "pixel_values" in sample_inputs
            and sample_inputs["pixel_values"] is not None
            and sample_inputs["pixel_values"].shape[0] > 0
        ):
            generation_inputs["pixel_values"] = sample_inputs["pixel_values"]

        if (
            "image_grid_thw" in sample_inputs
            and sample_inputs["image_grid_thw"] is not None
            and sample_inputs["image_grid_thw"].shape[0] > 0
        ):
            generation_inputs["image_grid_thw"] = sample_inputs["image_grid_thw"]

        # Validate vision input consistency
        has_pixel_values = "pixel_values" in generation_inputs
        has_grid_thw = "image_grid_thw" in generation_inputs

        if has_pixel_values != has_grid_thw:
            raise ValueError(
                f"Inconsistent vision inputs: pixel_values={has_pixel_values}, "
                f"image_grid_thw={has_grid_thw}. Both must be present or both must be absent."
            )

        # Get prompt length for token statistics
        prompt_length = sample_inputs["input_ids"].size(1)

        # Log generation inputs for debugging
        logger.debug(f"🚀 Sample {sample_index}: Starting generation with inputs:")
        logger.debug(f"   input_ids shape: {generation_inputs['input_ids'].shape}")
        logger.debug(f"   max_new_tokens: {generation_inputs['max_new_tokens']}")
        logger.debug(f"   has_pixel_values: {'pixel_values' in generation_inputs}")
        logger.debug(f"   has_image_grid_thw: {'image_grid_thw' in generation_inputs}")

        # Generate - let any errors bubble up
        # Disable tqdm progress bar for generation
        outputs = model.generate(**generation_inputs)

        logger.debug(
            f"✅ Sample {sample_index}: Generation completed, output shape: {outputs.shape}"
        )

        # Extract generated part and calculate lengths
        generated_ids = outputs[:, prompt_length:]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Log raw generated text
        logger.info(
            f"🎯 Sample {sample_index}: Generated text (length={len(generated_text)}):"
        )
        logger.info(
            f"   Content: {repr(generated_text[:300])}{'...' if len(generated_text) > 300 else ''}"
        )

        # Update token statistics if monitoring is enabled
        if self.enable_monitoring and token_stats and self.monitor:
            actual_generated_tokens = generated_ids.size(1)
            total_output_length = outputs.size(1)
            parsed_objects = self.parser.parse_response(generated_text, sample_index)

            # Update token stats for this sample
            if sample_index < len(token_stats.prompt_lengths):
                token_stats.prompt_lengths[sample_index] = prompt_length
                token_stats.generated_lengths[sample_index] = actual_generated_tokens
                token_stats.total_output_lengths[sample_index] = total_output_length

            # Log generation analysis
            self.monitor.log_generation_analysis(
                sample_idx=sample_index,
                prompt_length=prompt_length,
                max_new_tokens=self.max_generation_length,
                actual_generated_tokens=actual_generated_tokens,
                total_output_length=total_output_length,
                generated_text=generated_text,
                parsed_objects_count=len(parsed_objects),
            )
        else:
            # Parse objects even when monitoring is disabled for logging purposes
            parsed_objects = self.parser.parse_response(generated_text, sample_index)

        return generated_text.strip()

    def _generate_predictions_from_inputs(
        self,
        model: nn.Module,
        tokenizer,
        inputs: Dict,
        input_parts: List[torch.Tensor],
        token_stats: Optional[TokenStats] = None,
    ) -> Tuple[List[List[Dict]], List[str]]:
        """Generate predictions from extracted input parts using proper preparation."""
        from src.utils import prepare_inputs_for_generate

        # Set model to eval mode for inference
        original_training_mode = model.training
        model.eval()

        predicted_objects_batch = []
        predicted_texts = []

        with torch.no_grad():
            # Create mock inputs for generation preparation
            batch_size = len(input_parts)
            max_input_length = max(part.size(0) for part in input_parts)

            # Create padded input_ids tensor
            mock_input_ids = torch.full(
                (batch_size, max_input_length),
                tokenizer.pad_token_id,
                dtype=input_parts[0].dtype,
                device=input_parts[0].device,
            )

            prompt_end_indices = []
            for i, input_part in enumerate(input_parts):
                seq_len = input_part.size(0)
                mock_input_ids[i, :seq_len] = input_part
                prompt_end_indices.append(seq_len)

            # Create mock inputs dict with visual components
            mock_inputs = {"input_ids": mock_input_ids}

            # Add visual inputs from original inputs
            for key in ["pixel_values", "image_grid_thw", "image_counts_per_sample"]:
                if key in inputs and inputs[key] is not None:
                    mock_inputs[key] = inputs[key]

            # Prepare for generation
            generation_inputs, _ = prepare_inputs_for_generate(
                mock_inputs, prompt_end_indices
            )

            for i in range(batch_size):
                # Extract single sample
                sample_inputs = self._extract_sample_for_generation(
                    generation_inputs, i
                )

                # Generate response
                generated_text = self._generate_single_response_prepared(
                    model,
                    tokenizer,
                    sample_inputs,
                    sample_index=i,
                    token_stats=token_stats,
                )

                predicted_texts.append(generated_text)

                # Parse response
                predicted_objects = self.parser.parse_response(generated_text, i)
                predicted_objects_batch.append(predicted_objects)

        # Restore training mode
        model.train(original_training_mode)

        return predicted_objects_batch, predicted_texts

    def _compute_batch_losses(
        self,
        predicted_objects_batch: List[List[Dict]],
        ground_truth_objects_batch: List[List[Dict]],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute detection losses for the batch."""
        batch_losses = []

        for pred_objects, gt_objects in zip(
            predicted_objects_batch, ground_truth_objects_batch
        ):
            if not pred_objects:
                logger.warning("No predictions but have GT")
                continue

            # Match objects
            if self.matcher:
                matched_pred, matched_gt = self.matcher.match(pred_objects, gt_objects)
            else:
                min_len = min(len(pred_objects), len(gt_objects))
                matched_pred = pred_objects[:min_len]
                matched_gt = gt_objects[:min_len]

            if not matched_pred or not matched_gt:
                batch_losses.append(
                    torch.tensor(1.0, device=device, requires_grad=True)
                )
                continue

            # Compute individual losses
            sample_loss = self._compute_sample_loss(matched_pred, matched_gt, device)
            batch_losses.append(sample_loss)

        # Return average loss across batch
        if batch_losses:
            return torch.stack(batch_losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_sample_loss(
        self, matched_pred: List[Dict], matched_gt: List[Dict], device: torch.device
    ) -> torch.Tensor:
        """Compute loss for a single sample."""
        # Convert to tensors
        pred_boxes = torch.tensor(
            [obj["bbox"] for obj in matched_pred],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        gt_boxes = torch.tensor(
            [obj["bbox"] for obj in matched_gt], dtype=torch.float32, device=device
        )

        # Compute bbox L1 loss
        bbox_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")

        # Compute GIoU loss
        giou_loss = self._compute_giou_loss(pred_boxes, gt_boxes)

        # Compute semantic classification loss
        class_loss = self._compute_semantic_loss(matched_pred, matched_gt, device)

        # Combine losses
        total_loss = (
            self.bbox_weight * bbox_loss
            + self.giou_weight * giou_loss
            + self.class_weight * class_loss
        )

        return total_loss

    def _compute_giou_loss(
        self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute GIoU loss."""
        # Calculate intersection
        x1_inter = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
        y1_inter = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
        x2_inter = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
        y2_inter = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(
            y2_inter - y1_inter, min=0
        )

        # Calculate union
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (
            pred_boxes[..., 3] - pred_boxes[..., 1]
        )
        gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (
            gt_boxes[..., 3] - gt_boxes[..., 1]
        )
        union_area = pred_area + gt_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)

        # Calculate enclosing box
        x1_enclosing = torch.min(pred_boxes[..., 0], gt_boxes[..., 0])
        y1_enclosing = torch.min(pred_boxes[..., 1], gt_boxes[..., 1])
        x2_enclosing = torch.max(pred_boxes[..., 2], gt_boxes[..., 2])
        y2_enclosing = torch.max(pred_boxes[..., 3], gt_boxes[..., 3])

        enclosing_area = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)

        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)

        # Return GIoU loss (1 - GIoU)
        return 1 - giou.mean()

    def _compute_semantic_loss(
        self, matched_pred: List[Dict], matched_gt: List[Dict], device: torch.device
    ) -> torch.Tensor:
        """Compute semantic classification loss."""
        similarities = []

        for pred_obj, gt_obj in zip(matched_pred, matched_gt):
            pred_desc = pred_obj.get("description", "")
            gt_desc = gt_obj.get("description", "")
            similarity = self.parser.calculate_semantic_similarity(pred_desc, gt_desc)
            similarities.append(similarity)

        if not similarities:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert to loss (1 - similarity)
        similarity_tensor = torch.tensor(
            similarities, device=device, dtype=torch.float32
        )
        return 1.0 - similarity_tensor.mean()
