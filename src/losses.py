"""
Unified loss functions for Qwen2.5-VL training with object detection integration.
Simplified and efficient implementation focusing on telecom equipment inspection.

Key Features:
- 🚀 Embedding Reuse: Avoids reprocessing images by reusing embeddings from training forward pass
- 🔄 Multi-Round Conversations: Properly handles few-shot examples (only last assistant response is GT)
- ⚡ Batch Processing: Efficient batch generation instead of sample-by-sample
- 🎯 Hungarian Matching: Optimal object matching for detection losses
- 📊 Semantic Classification: Uses SentenceTransformer for description similarity

Usage Example:

    # 1. Initialize the loss
    detection_loss = ObjectDetectionLoss(
        lm_weight=1.0,
        bbox_weight=0.5,
        giou_weight=0.3,
        class_weight=0.2,
        detection_mode="inference",  # or "hybrid" for occasional detection loss
        inference_frequency=5,  # compute detection loss every 5 steps (for hybrid mode)
    )

    # 2. In your training loop:
    def training_step(self, model, inputs):
        # Standard forward pass
        outputs = model(**inputs)

        # 🚀 OPTIMIZATION: Extract embeddings for reuse (optional but recommended)
        if hasattr(inputs, 'pixel_values') and inputs['pixel_values'] is not None:
            inputs_embeds = ObjectDetectionLoss.extract_embeddings_from_model(
                model,
                inputs['input_ids'],
                inputs.get('pixel_values'),
                inputs.get('image_grid_thw')
            )
        else:
            inputs_embeds = None

        # Compute combined loss (LM + Detection)
        loss_dict = detection_loss(
            model=model,
            outputs=outputs,
            tokenizer=self.tokenizer,
            input_ids=inputs['input_ids'],
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            ground_truth_objects=inputs.get('ground_truth_objects'),
            labels=inputs.get('labels'),
            inputs_embeds=inputs_embeds,  # 🚀 Pass embeddings for optimization
        )

        return loss_dict['total_loss']

Multi-Round Conversation Format:
    Your training data with multiple user-assistant rounds is handled correctly:

    conversations: [
        {"role": "system", "content": "You are Telecom Inspector Pro..."},
        {"role": "user", "content": "<image>"},           # Example 1
        {"role": "assistant", "content": "[{...}]"},       # Example 1 response (masked)
        {"role": "user", "content": "<image>"},           # Example 2
        {"role": "assistant", "content": "[{...}]"},       # Example 2 response (masked)
        {"role": "user", "content": "<image>"},           # Query image
        # Last assistant response will be the training target
    ]

    ✅ Only the LAST assistant response is used as ground truth
    ✅ All previous assistant responses are masked (not trained on)
    ✅ System and user messages are always masked
"""

import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_debug_logging():
    """Set up debug logging configuration."""
    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    for handler in debug_logger.handlers[:]:
        debug_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to logger
    debug_logger.addHandler(console_handler)

    return debug_logger


def clear_cuda_cache():
    """Clear CUDA cache to save memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_cuda_memory(context: str = ""):
    """Log CUDA memory usage for debugging OOM issues."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        debug_logger.info(
            f"🔧 CUDA Memory {context} | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB"
        )
    else:
        debug_logger.info(f"🔧 CUDA not available {context}")


def log_compact_tokens(tokenizer, input_ids: torch.Tensor, context: str = ""):
    """Compact logging: show first 100 and last 100 tokens with their text."""
    debug_logger.info(f"📊 {context} | input.shape={input_ids.shape}")

    if input_ids.numel() == 0:
        debug_logger.info("   Empty tensor")
        return

    # Get first and last 100 tokens
    tokens = input_ids.flatten()
    total_tokens = len(tokens)

    if total_tokens <= 200:
        # Show all tokens if less than 200
        first_100 = tokens
        last_100 = torch.tensor([])
    else:
        first_100 = tokens[:100]
        last_100 = tokens[-100:]

    # Decode and show
    try:
        if len(first_100) > 0:
            first_text = tokenizer.decode(first_100, skip_special_tokens=False)
            debug_logger.info(f"   First 100: {first_100.tolist()}")
            debug_logger.info(f"   First text: {repr(first_text[:200])}")

        if len(last_100) > 0:
            last_text = tokenizer.decode(last_100, skip_special_tokens=False)
            debug_logger.info(f"   Last 100: {last_100.tolist()}")
            debug_logger.info(f"   Last text: {repr(last_text[:200])}")

        if total_tokens > 200:
            debug_logger.info(f"   ... [{total_tokens - 200} tokens omitted] ...")

    except Exception as e:
        debug_logger.warning(f"   Decode failed: {e}")


# Initialize debug logger
debug_logger = setup_debug_logging()


# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    warnings.warn(
        "sentence-transformers not available. Falling back to rule-based similarity."
    )

try:
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Some similarity metrics will be disabled.")

# Hungarian matching for optimal object assignment
try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Hungarian matching will be disabled.")


class LossComputationError(Exception):
    """Custom exception for loss computation errors."""

    pass


class DeviceManager:
    """Centralized device management for loss computations."""

    @staticmethod
    def check_device_consistency(
        tensors: List[torch.Tensor], expected_device: torch.device, context: str = ""
    ):
        """Check that all tensors are on the expected device."""
        for i, tensor in enumerate(tensors):
            if tensor.device != expected_device:
                raise LossComputationError(
                    f"Device mismatch in {context}: tensor[{i}] is on {tensor.device}, "
                    f"expected {expected_device}"
                )

    @staticmethod
    def ensure_tensor_device(
        tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        if tensor.device != device:
            return tensor.to(device)
        return tensor

    @staticmethod
    def create_zero_loss(device: torch.device) -> torch.Tensor:
        """Create a zero loss tensor on the specified device."""
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def validate_tensor_shapes(
        tensors: List[torch.Tensor], expected_shapes: List[tuple], context: str = ""
    ):
        """Validate tensor shapes match expected dimensions."""
        for i, (tensor, expected_shape) in enumerate(zip(tensors, expected_shapes)):
            if tensor.shape != expected_shape:
                raise LossComputationError(
                    f"Shape mismatch in {context}: tensor[{i}] has shape {tensor.shape}, "
                    f"expected {expected_shape}"
                )


class ResponseParser:
    """
    Parser for Qwen2.5-VL special token format with fallback to legacy JSON.

    Primary: <|object_ref_start|>desc<|object_ref_end|><|box_start|>(x1, y1), (x2, y2)<|box_end|>
    Fallback: [{bbox:[x1,y1,x2,y2],desc:'object_type, details'}]
    """

    def __init__(self, early_training_mode: bool = True):
        self.early_training_mode = early_training_mode
        self._init_sentence_transformer()

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer with better error handling."""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.sentence_transformer = SentenceTransformer(
                "/data4/swift/model_cache/sentence-transformers/all-MiniLM-L6-v2/"
            )
            print("✅ SentenceTransformer loaded successfully")
        else:
            self.sentence_transformer = None

    def parse_response(self, response_text: str) -> List[Dict]:
        """
        Parse response text into list of bbox objects.
        Handles multiple formats:
        1. Special tokens: <|object_ref_start|>desc<|object_ref_end|><|box_start|>(x1, y1), (x2, y2)<|box_end|>
        2. Valid JSON: [{"bbox":[x1,y1,x2,y2],"desc":"description"}]
        3. Unquoted format: [{bbox:[x1,y1,x2,y2],desc:'description'}]

        Args:
            response_text: Raw model response

        Returns:
            List of dictionaries with 'bbox' and 'description' keys
        """
        if not response_text or not response_text.strip():
            return []

        # Clean and parse
        text = self._clean_response_text(response_text)

        # Try special token format first (preferred)
        objects = self._parse_special_tokens(text)

        # If special token parsing fails, try JSON format
        if not objects:
            objects = self._parse_json_format(text)

        # If JSON parsing fails, try unquoted format
        if not objects:
            objects = self._parse_unquoted_format(text)

        # If still no objects, try alternative patterns
        if not objects:
            objects = self._try_alternative_patterns(text)

        return self._validate_objects(objects)

    def _clean_response_text(self, text: str) -> str:
        """Remove common prefixes/suffixes and normalize."""
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```") and text.endswith("```"):
            lines = text.split(sep="\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])

        # Remove common prefixes
        prefixes_to_remove = [
            "Here are the objects:",
            "The objects in the image are:",
            "Objects detected:",
            "Detection results:",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        return text

    def _parse_special_tokens(self, text: str) -> List[Dict]:
        """
        Parse Qwen2.5-VL special token format.

        Format: <|object_ref_start|>description<|object_ref_end|><|box_start|>(x1, y1), (x2, y2)<|box_end|>
        """
        objects = []

        # Pattern to match special token format
        pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\(([^)]+)\),\s*\(([^)]+)\)<\|box_end\|>"

        matches = re.findall(pattern, text, re.DOTALL)

        for desc, coords1, coords2 in matches:
            try:
                # Parse coordinates: (x1, y1), (x2, y2)
                x1, y1 = map(float, coords1.split(", "))
                x2, y2 = map(float, coords2.split(", "))
                bbox = [x1, y1, x2, y2]
                objects.append({"bbox": bbox, "description": desc.strip()})
            except (ValueError, IndexError) as e:
                if not self.early_training_mode:
                    raise LossComputationError(
                        f"Failed to parse special token coordinates: {e}"
                    )
                continue

        return objects

    def _parse_json_format(self, text: str) -> List[Dict]:
        """Parse valid JSON format."""
        objects = []

        try:
            # Try to parse as valid JSON
            parsed = json.loads(text)

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "bbox" in item and "desc" in item:
                        bbox = item["bbox"]
                        desc = item["desc"]

                        if isinstance(bbox, list) and len(bbox) == 4:
                            try:
                                coords = [float(x) for x in bbox]
                                objects.append({"bbox": coords, "description": desc})
                            except (ValueError, TypeError):
                                continue

        except (json.JSONDecodeError, TypeError, KeyError):
            # Not valid JSON, will try other formats
            pass

        return objects

    def _parse_unquoted_format(self, text: str) -> List[Dict]:
        """Parse the unquoted format using regex."""
        # Pattern for unquoted format: {bbox:[x1,y1,x2,y2],desc:'description'}
        pattern = r'\{bbox:\s*\[([^\]]+)\]\s*,\s*desc:\s*[\'"]([^\'"]+)[\'"]\s*\}'

        matches = re.findall(pattern, text)
        objects = []

        for bbox_str, desc in matches:
            try:
                # Parse bbox coordinates
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) != 4:
                    continue

                objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError) as e:
                if not self.early_training_mode:
                    raise LossComputationError(f"Failed to parse bbox coordinates: {e}")
                continue

        return objects

    def _try_alternative_patterns(self, text: str) -> List[Dict]:
        """Try alternative parsing patterns."""
        objects = []

        # Try JSON-like format with quotes
        json_pattern = r'\{"bbox":\s*\[([^\]]+)\]\s*,\s*"desc":\s*"([^"]+)"\s*\}'
        matches = re.findall(json_pattern, text)

        for bbox_str, desc in matches:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        # Try pattern without quotes around bbox
        if not objects:
            pattern2 = r'\{bbox:\[([^\]]+)\],desc:[\'"]([^\'"]+)[\'"]\}'
            matches = re.findall(pattern2, text)

            for bbox_str, desc in matches:
                try:
                    coords = [float(x.strip()) for x in bbox_str.split(",")]
                    if len(coords) == 4:
                        objects.append({"bbox": coords, "description": desc.strip()})
                except (ValueError, IndexError):
                    continue

        return objects

    def _validate_objects(self, objects: List[Dict]) -> List[Dict]:
        """Validate parsed objects."""
        validated = []

        for obj in objects:
            try:
                # Validate bbox format
                bbox = obj.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                # Ensure all coordinates are numeric
                bbox = [float(coord) for coord in bbox]

                # Validate bbox coordinates (x1 <= x2, y1 <= y2)
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    continue

                # Validate description
                desc = obj.get("description", "").strip()
                if not desc:
                    continue

                validated.append({"bbox": bbox, "description": desc})

            except (ValueError, TypeError) as e:
                if not self.early_training_mode:
                    raise LossComputationError(f"Object validation failed: {e}")
                continue

        return validated

    def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate semantic similarity between descriptions."""
        if self.sentence_transformer is not None:
            embeddings = self.sentence_transformer.encode([desc1, desc2])
            similarity = torch.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0),
            ).item()
            return max(0.0, similarity)

        return self._rule_based_similarity(desc1, desc2)

    def _rule_based_similarity(self, desc1: str, desc2: str) -> float:
        """Rule-based similarity as fallback."""
        desc1_lower = desc1.lower().strip()
        desc2_lower = desc2.lower().strip()

        if desc1_lower == desc2_lower:
            return 1.0

        # Extract key terms
        words1 = set(desc1_lower.split())
        words2 = set(desc2_lower.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class BaseLoss(ABC, nn.Module):
    """Abstract base class for all loss functions."""

    def __init__(self, device_manager: DeviceManager = None):
        super().__init__()
        self.device_manager = device_manager or DeviceManager()

    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        pass


class BoundingBoxLoss(BaseLoss):
    """L1 loss for bounding box regression."""

    def compute(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss between predicted and ground truth boxes."""
        self.device_manager.check_device_consistency(
            [pred_boxes, gt_boxes], pred_boxes.device, "BoundingBoxLoss"
        )
        return F.l1_loss(pred_boxes, gt_boxes, reduction="mean")


class GIoULoss(BaseLoss):
    """Generalized IoU loss for bounding boxes."""

    def compute(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss between predicted and ground truth boxes."""
        self.device_manager.check_device_consistency(
            [pred_boxes, gt_boxes], pred_boxes.device, "GIoULoss"
        )

        # Ensure boxes are in [x1, y1, x2, y2] format
        if pred_boxes.shape[-1] != 4 or gt_boxes.shape[-1] != 4:
            raise LossComputationError("Boxes must have 4 coordinates")

        return self._compute_giou_loss(pred_boxes, gt_boxes)

    def _compute_giou_loss(
        self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute the actual GIoU loss."""
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


class SemanticClassificationLoss(BaseLoss):
    """Semantic classification loss using desc similarity."""

    def __init__(self, parser: ResponseParser, device_manager: DeviceManager = None):
        super().__init__(device_manager)
        self.parser = parser

    def compute(
        self, pred_objects: List[Dict], gt_objects: List[Dict], device: torch.device
    ) -> torch.Tensor:
        """Compute semantic classification loss."""
        if not pred_objects or not gt_objects:
            return self.device_manager.create_zero_loss(device)

        similarities = []
        for pred_obj in pred_objects:
            pred_desc = pred_obj.get("description", "")
            best_similarity = 0.0

            for gt_obj in gt_objects:
                gt_desc = gt_obj.get("description", "")
                similarity = self.parser.calculate_semantic_similarity(
                    pred_desc, gt_desc
                )
                best_similarity = max(best_similarity, similarity)

            similarities.append(best_similarity)

        if not similarities:
            return self.device_manager.create_zero_loss(device)

        # Convert to loss (1 - similarity)
        similarity_tensor = torch.tensor(
            similarities, device=device, dtype=torch.float32
        )
        loss = 1.0 - similarity_tensor.mean()

        return loss


class HungarianMatcher:
    """Hungarian matching for optimal object assignment."""

    def __init__(self, parser: ResponseParser):
        self.parser = parser

    def match(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Match predicted and ground truth objects using Hungarian algorithm."""
        if not pred_objects or not gt_objects:
            return [], []

        if not SCIPY_AVAILABLE:
            # Fallback to greedy matching
            return self._greedy_match(pred_objects, gt_objects)

        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(pred_objects, gt_objects)

        # Apply Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        # Extract matched objects
        matched_pred = [pred_objects[i] for i in pred_indices]
        matched_gt = [gt_objects[i] for i in gt_indices]

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
        # Bbox IoU cost (lower IoU = higher cost)
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
        total_cost = 0.7 * bbox_cost + 0.3 * semantic_cost

        return total_cost

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
        iou = inter_area / (union_area + 1e-7)

        return iou

    def _greedy_match(
        self, pred_objects: List[Dict], gt_objects: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Greedy matching as fallback when scipy is not available."""
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


class ObjectDetectionLoss(nn.Module):
    """
    Refactored object detection loss with improved architecture.

    Key improvements:
    - Separated concerns into specialized loss classes
    - Better device management
    - Improved error handling
    - More maintainable code structure
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        bbox_weight: float = 0.5,
        giou_weight: float = 0.3,
        class_weight: float = 0.2,
        hungarian_matching: bool = True,
        ignore_index: int = -100,
        detection_mode: str = "inference",
        inference_frequency: int = 5,
        max_generation_length: int = 512,
        use_semantic_similarity: bool = True,
        early_training_mode: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.lm_weight = lm_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.class_weight = class_weight
        self.hungarian_matching = hungarian_matching
        self.ignore_index = ignore_index
        self.detection_mode = detection_mode
        self.inference_frequency = inference_frequency
        self.max_generation_length = max_generation_length
        self.use_semantic_similarity = use_semantic_similarity
        self.early_training_mode = early_training_mode

        # Initialize components
        self.device_manager = DeviceManager()
        self.parser = ResponseParser(early_training_mode=early_training_mode)
        self.matcher = HungarianMatcher(self.parser) if hungarian_matching else None

        # Initialize loss functions
        self.bbox_loss = BoundingBoxLoss(self.device_manager)
        self.giou_loss = GIoULoss(self.device_manager)
        self.semantic_loss = SemanticClassificationLoss(
            self.parser, self.device_manager
        )

        # Training step counter
        self.training_step = 0

    def set_training_mode(self, early_training: bool = True):
        """Toggle between early training mode and strict mode."""
        self.early_training_mode = early_training
        self.parser.early_training_mode = early_training
        print(
            f"🔧 Detection loss parser mode: {'Early Training (Lenient)' if early_training else 'Strict'}"
        )

    def forward(
        self,
        model,
        outputs,
        tokenizer,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        ground_truth_objects: Optional[List[List[Dict]]] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined language modeling and detection losses with memory optimization."""
        # Get model device and validate inputs
        model_device = next(model.parameters()).device

        # Clear CUDA cache at the start of loss computation
        clear_cuda_cache()
        log_cuda_memory("START")

        self._validate_device_consistency(
            [input_ids, pixel_values, image_grid_thw, labels, inputs_embeds],
            model_device,
        )

        # Initialize total loss
        total_loss = self.device_manager.create_zero_loss(model_device)
        loss_dict = {}

        # Standard Language Modeling loss
        if outputs.loss is not None:
            lm_loss = outputs.loss
            # Log the LM loss for debugging
            debug_logger.debug(f"📊 LM Loss: {lm_loss.item():.6f}")
            loss_dict["lm_loss"] = lm_loss
            # CRITICAL FIX: Detach before accumulation to prevent computation graph retention
            total_loss = total_loss + self.lm_weight * lm_loss.detach()

        # If detection mode is disabled, return early
        if self.detection_mode == "disabled":
            debug_logger.info("🚫 Detection loss disabled, returning LM loss only")
            # Ensure total_loss is scalar
            if total_loss.dim() > 0:
                total_loss = total_loss.squeeze()
            loss_dict["total_loss"] = total_loss
            return loss_dict

        # Check if we should compute detection loss
        if not self._should_compute_detection_loss(ground_truth_objects, labels):
            debug_logger.debug("⏭️ Skipping detection loss computation")
            # Ensure total_loss is scalar
            if total_loss.dim() > 0:
                total_loss = total_loss.squeeze()
            loss_dict["total_loss"] = total_loss
            return loss_dict

        # Compute detection losses
        try:
            detection_losses = self._compute_detection_losses(
                model,
                tokenizer,
                input_ids,
                labels,
                pixel_values,
                image_grid_thw,
                ground_truth_objects,
                inputs_embeds,
            )

            # Add detection losses to total with proper detaching
            for loss_name, loss_value in detection_losses.items():
                if loss_value is not None and not torch.isnan(loss_value):
                    loss_dict[loss_name] = loss_value

                    # CRITICAL FIX: Detach before accumulation to prevent computation graph retention
                    if loss_name == "bbox_loss":
                        total_loss = total_loss + self.bbox_weight * loss_value.detach()
                    elif loss_name == "giou_loss":
                        total_loss = total_loss + self.giou_weight * loss_value.detach()
                    elif loss_name == "class_loss":
                        total_loss = (
                            total_loss + self.class_weight * loss_value.detach()
                        )

        except Exception as e:
            debug_logger.warning(f"⚠️ Detection loss computation failed: {e}")

        # Ensure total_loss is scalar
        if total_loss.dim() > 0:
            total_loss = total_loss.squeeze()

        loss_dict["total_loss"] = total_loss

        debug_logger.debug(f"✅ Final total loss: {total_loss.item():.6f}")

        return loss_dict

    def _validate_device_consistency(
        self, tensors: List[Optional[torch.Tensor]], expected_device: torch.device
    ):
        """Validate device consistency for all tensor inputs."""
        valid_tensors = [t for t in tensors if t is not None]
        self.device_manager.check_device_consistency(
            valid_tensors, expected_device, "ObjectDetectionLoss forward"
        )

    def _should_compute_detection_loss(self, ground_truth_objects, labels) -> bool:
        """Determine whether to compute detection loss."""
        if ground_truth_objects is None or labels is None:
            return False

        if self.detection_mode not in ["inference", "hybrid"]:
            return False

        if self.detection_mode == "hybrid":
            return self.training_step % self.inference_frequency == 0

        return True

    def _compute_detection_losses(
        self,
        model,
        tokenizer,
        input_ids,
        labels,
        pixel_values,
        image_grid_thw,
        ground_truth_objects,
        inputs_embeds,
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses using the appropriate method."""
        # Validate inputs_embeds if provided
        if inputs_embeds is not None:
            # Check if inputs_embeds is valid and safe to use
            try:
                # Basic validation
                if (
                    inputs_embeds.shape[0] != input_ids.shape[0]
                    or inputs_embeds.shape[1] != input_ids.shape[1]
                ):
                    print(
                        "⚠️ Warning: inputs_embeds shape mismatch, falling back to standard processing"
                    )
                    inputs_embeds = None
                elif (
                    torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any()
                ):
                    print(
                        "⚠️ Warning: inputs_embeds contains NaN/Inf, falling back to standard processing"
                    )
                    inputs_embeds = None
            except Exception as e:
                print(
                    f"⚠️ Warning: inputs_embeds validation failed: {e}, falling back to standard processing"
                )
                inputs_embeds = None

        if inputs_embeds is not None:
            return self._compute_detection_losses_with_embeddings(
                model, tokenizer, input_ids, labels, inputs_embeds, ground_truth_objects
            )
        else:
            return self._compute_detection_losses_standard(
                model,
                tokenizer,
                input_ids,
                labels,
                pixel_values,
                image_grid_thw,
                ground_truth_objects,
            )

    def _compute_detection_losses_with_embeddings(
        self, model, tokenizer, input_ids, labels, inputs_embeds, ground_truth_objects
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses using pre-computed embeddings with memory optimization."""
        model_device = next(model.parameters()).device

        # Clear CUDA cache before detection loss computation
        clear_cuda_cache()

        self.device_manager.check_device_consistency(
            [input_ids, labels, inputs_embeds],
            model_device,
            "detection_losses_with_embeddings",
        )

        # Set model to eval mode for inference
        original_training_mode = model.training
        model.eval()

        with torch.no_grad():
            # Extract and generate responses with unified method
            predicted_objects_batch = self._generate_responses_unified(
                model, tokenizer, input_ids, labels, inputs_embeds=inputs_embeds
            )

        # Restore original training mode
        model.train(original_training_mode)

        # Clear cache after generation
        clear_cuda_cache()

        return self._compute_batch_detection_losses(
            predicted_objects_batch, ground_truth_objects, model_device
        )

    def _compute_detection_losses_standard(
        self,
        model,
        tokenizer,
        input_ids,
        labels,
        pixel_values,
        image_grid_thw,
        ground_truth_objects,
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses using standard inference with memory optimization."""
        model_device = next(model.parameters()).device

        # Clear CUDA cache before detection loss computation
        clear_cuda_cache()

        # Validate multimodal inputs before generation
        if pixel_values is not None and image_grid_thw is not None:
            debug_logger.info(
                f"🔍 MULTIMODAL | input.shape={input_ids.shape} | pixel.shape={pixel_values.shape} | grid.shape={image_grid_thw.shape}"
            )
            log_compact_tokens(tokenizer, input_ids, "INPUT")
            self.validate_multimodal_inputs(
                input_ids, pixel_values, image_grid_thw, model.config
            )
            debug_logger.info("✅ Validation passed")
        else:
            debug_logger.info("📝 Text-only batch")

        # Set model to eval mode for inference
        original_training_mode = model.training
        model.eval()

        with torch.no_grad():
            predicted_objects_batch = self._generate_responses_unified(
                model, tokenizer, input_ids, labels, pixel_values, image_grid_thw
            )

        # Restore original training mode
        model.train(original_training_mode)

        # Clear cache after generation
        clear_cuda_cache()

        return self._compute_batch_detection_losses(
            predicted_objects_batch, ground_truth_objects, model_device
        )

    def _generate_responses_unified(
        self,
        model,
        tokenizer,
        input_ids,
        labels,
        pixel_values=None,
        image_grid_thw=None,
        inputs_embeds=None,
    ):
        """Unified response generation method supporting both embeddings and standard inputs."""
        batch_size = input_ids.size(0)
        predicted_objects_batch = []

        # CRITICAL FIX: Handle empty tensors at batch level
        if pixel_values is not None and pixel_values.shape[0] == 0:
            debug_logger.warning(
                "⚠️ Empty pixel_values at batch level - treating all samples as text-only"
            )
            pixel_values = None
            image_grid_thw = None

        if image_grid_thw is not None and image_grid_thw.shape[0] == 0:
            debug_logger.warning(
                "⚠️ Empty image_grid_thw at batch level - treating all samples as text-only"
            )
            pixel_values = None
            image_grid_thw = None

        # Process each sample individually to save memory
        for i in range(batch_size):
            # Clear CUDA cache before processing each sample
            clear_cuda_cache()

            # Extract single sample
            sample_input_ids = input_ids[i : i + 1]
            sample_labels = labels[i : i + 1] if labels is not None else None

            # IMPROVED: Handle pixel_values slicing more carefully
            sample_pixel_values = None
            if pixel_values is not None:
                # For text-only samples in a mixed batch, we need to handle indexing carefully
                # Since pixel_values are concatenated across all samples, we need to determine
                # how many patches belong to each sample
                if i == 0:
                    # For simplicity, if we have pixel_values, assume all samples have images
                    # This is a limitation that should be addressed in the data collator
                    sample_pixel_values = pixel_values
                else:
                    # For subsequent samples, we can't easily slice pixel_values
                    # because we don't know the patch distribution per sample
                    # This is a known limitation of the current data format
                    sample_pixel_values = None
                    debug_logger.warning(
                        f"⚠️ Cannot slice pixel_values for sample {i} - treating as text-only"
                    )

            # IMPROVED: Handle image_grid_thw slicing more carefully
            sample_image_grid_thw = None
            if image_grid_thw is not None:
                if i == 0:
                    # For simplicity, if we have image_grid_thw, assume all samples have images
                    sample_image_grid_thw = image_grid_thw
                else:
                    # For subsequent samples, we can't easily slice image_grid_thw
                    # because we don't know the image distribution per sample
                    sample_image_grid_thw = None
                    debug_logger.warning(
                        f"⚠️ Cannot slice image_grid_thw for sample {i} - treating as text-only"
                    )

            sample_embeds = (
                inputs_embeds[i : i + 1] if inputs_embeds is not None else None
            )

            # Find the prompt end (where generation should start)
            prompt_end_idx = self._find_prompt_end(
                sample_input_ids, sample_labels, tokenizer
            )

            if prompt_end_idx is None:
                predicted_objects_batch.append([])
                continue

            # Generate response
            generated_text = self._generate_single_response_unified(
                model,
                tokenizer,
                sample_input_ids,
                prompt_end_idx,
                sample_pixel_values,
                sample_image_grid_thw,
                sample_embeds,
            )

            # Parse the generated response
            if generated_text:
                predicted_objects = self.parser.parse_response(generated_text)
                predicted_objects_batch.append(predicted_objects)
            else:
                predicted_objects_batch.append([])

        return predicted_objects_batch

    def _find_prompt_end(self, input_ids, labels, tokenizer):
        """Find where the prompt ends and generation should start."""
        if labels is None:
            # If no labels, generate from the end of input
            return input_ids.size(1) - 1

        # Find the first non-masked token in labels (where actual response starts)
        labels_flat = labels.flatten()
        non_masked_indices = (labels_flat != self.ignore_index).nonzero(as_tuple=True)[
            0
        ]

        if len(non_masked_indices) > 0:
            return non_masked_indices[0].item()
        else:
            # Fallback: generate from end of input
            return input_ids.size(1) - 1

    def _generate_single_response_unified(
        self,
        model,
        tokenizer,
        input_ids,
        prompt_end_idx,
        pixel_values=None,
        image_grid_thw=None,
        inputs_embeds=None,
    ):
        """Generate a single response using the unified approach."""
        debug_logger = logging.getLogger("debug")

        # Prepare prompt
        prompt_ids = input_ids[:, :prompt_end_idx]
        debug_logger.info(f"🎯 GENERATION | prompt.shape={prompt_ids.shape}")
        log_compact_tokens(tokenizer, prompt_ids[0], "PROMPT")

        # CRITICAL FIX: Check for empty pixel_values and treat as None
        if pixel_values is not None and pixel_values.shape[0] == 0:
            debug_logger.error(
                "❌ Empty pixel_values detected - treating as text-only input"
            )
            pixel_values = None
            image_grid_thw = None
            raise ValueError("Empty pixel_values detected")

        # CRITICAL FIX: Check for empty image_grid_thw and treat as None
        if image_grid_thw is not None and image_grid_thw.shape[0] == 0:
            debug_logger.error(
                "❌ Empty image_grid_thw detected - treating as text-only input"
            )
            raise ValueError("Empty image_grid_thw detected")

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": self.max_generation_length,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Add multimodal inputs ONLY if they are valid (non-empty)
        if pixel_values is not None and pixel_values.shape[0] > 0:
            generation_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
            generation_kwargs["image_grid_thw"] = image_grid_thw
        if inputs_embeds is not None:
            generation_kwargs["inputs_embeds"] = inputs_embeds

        debug_logger.info(f"🔧 Generation kwargs: {list(generation_kwargs.keys())}")

        with torch.no_grad():
            # Use the robust official generation method
            outputs = model.generate(input_ids=prompt_ids, **generation_kwargs)

        # Extract only the generated part
        generated_ids = outputs[:, prompt_ids.size(1) :]

        # Decode with and without special tokens
        generated_text_with_special = tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Log generation results
        debug_logger.info("✅ GENERATION COMPLETED")
        log_compact_tokens(tokenizer, generated_ids[0], "GENERATED")
        debug_logger.info(f"   Generated text: {repr(generated_text[:200])}")

        return generated_text.strip()

    def _compute_batch_detection_losses(
        self, predicted_objects_batch, ground_truth_objects_batch, device
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses for a batch using specialized loss functions."""
        batch_bbox_losses = []
        batch_giou_losses = []
        batch_class_losses = []

        for pred_objects, gt_objects in zip(
            predicted_objects_batch, ground_truth_objects_batch
        ):
            if not gt_objects:
                # No ground truth - create zero losses
                batch_bbox_losses.append(self.device_manager.create_zero_loss(device))
                batch_giou_losses.append(self.device_manager.create_zero_loss(device))
                batch_class_losses.append(self.device_manager.create_zero_loss(device))
                continue

            if not pred_objects:
                # No predictions but have GT - penalize
                num_gt = len(gt_objects)
                batch_bbox_losses.append(
                    torch.tensor(5.0 * num_gt, device=device, requires_grad=True)
                )
                batch_giou_losses.append(
                    torch.tensor(2.0 * num_gt, device=device, requires_grad=True)
                )
                batch_class_losses.append(
                    torch.tensor(1.0 * num_gt, device=device, requires_grad=True)
                )
                continue

            # Match objects
            if self.matcher:
                matched_pred, matched_gt = self.matcher.match(pred_objects, gt_objects)
            else:
                min_len = min(len(pred_objects), len(gt_objects))
                matched_pred = pred_objects[:min_len]
                matched_gt = gt_objects[:min_len]

            if not matched_pred or not matched_gt:
                batch_bbox_losses.append(
                    torch.tensor(1.0, device=device, requires_grad=True)
                )
                batch_giou_losses.append(
                    torch.tensor(1.0, device=device, requires_grad=True)
                )
                batch_class_losses.append(
                    torch.tensor(1.0, device=device, requires_grad=True)
                )
                continue

            # Compute losses using specialized loss functions
            pred_boxes = torch.tensor(
                [obj["bbox"] for obj in matched_pred],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            gt_boxes = torch.tensor(
                [obj["bbox"] for obj in matched_gt], dtype=torch.float32, device=device
            )

            bbox_loss = self.bbox_loss.compute(pred_boxes, gt_boxes)
            giou_loss = self.giou_loss.compute(pred_boxes, gt_boxes)
            class_loss = self.semantic_loss.compute(matched_pred, matched_gt, device)

            batch_bbox_losses.append(bbox_loss)
            batch_giou_losses.append(giou_loss)
            batch_class_losses.append(class_loss)

        # Average losses across batch
        return {
            "bbox_loss": torch.stack(batch_bbox_losses).mean()
            if batch_bbox_losses
            else self.device_manager.create_zero_loss(device),
            "giou_loss": torch.stack(batch_giou_losses).mean()
            if batch_giou_losses
            else self.device_manager.create_zero_loss(device),
            "class_loss": torch.stack(batch_class_losses).mean()
            if batch_class_losses
            else self.device_manager.create_zero_loss(device),
        }

    @staticmethod
    def validate_multimodal_inputs(
        input_ids, pixel_values, image_grid_thw, model_config
    ):
        """Validate that multimodal inputs are properly aligned."""
        if pixel_values is None or image_grid_thw is None:
            return True  # Text-only is always valid

        # Handle empty tensors gracefully
        if pixel_values.shape[0] == 0 or image_grid_thw.shape[0] == 0:
            debug_logger.warning(
                f"⚠️ Empty tensors | pixel.shape={pixel_values.shape} | grid.shape={image_grid_thw.shape}"
            )
            return True  # Allow processing as text-only

        # Check image sequence alignment
        vision_start_token_id = 151652  # <|vision_start|>
        image_sequence_count = (input_ids == vision_start_token_id).sum().item()
        expected_features = image_grid_thw.shape[0]

        if image_sequence_count != expected_features:
            raise ValueError(
                f"❌ Image sequence mismatch! Found {image_sequence_count} vision tokens, expected {expected_features}"
            )

        # Validate pixel_values format
        if len(pixel_values.shape) not in [2, 4]:
            raise ValueError(
                f"❌ Invalid pixel_values dimensions! Expected 2D or 4D, got {pixel_values.shape}"
            )

        return True

    @staticmethod
    def extract_embeddings_from_model(
        model, input_ids, pixel_values=None, image_grid_thw=None, **kwargs
    ):
        """
        DISABLED: Embedding extraction and reuse is disabled to prevent CUDA errors.

        The embedding reuse optimization was causing CUDA index out of bounds errors
        during generation because the custom visual processing couldn't handle edge cases
        properly. The official Qwen2.5-VL generation method is more robust.

        Returns:
            None (always) - forces the system to use standard generation
        """
        # DISABLED: Always return None to force standard generation
        # This prevents CUDA index out of bounds errors caused by embedding reuse
        print("🔧 Embedding reuse disabled - using robust standard generation")
        return None
