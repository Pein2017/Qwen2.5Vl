#!/usr/bin/env python3
"""
COCO-style evaluation metrics for dense captioning/grounding tasks.

This module provides comprehensive evaluation capabilities for object detection
with open-vocabulary descriptions, specifically designed for Chinese industrial
equipment annotation tasks.

Enhanced features for open-vocabulary evaluation:
- Hierarchical label matching
- Soft semantic scoring
- Novel object detection metrics
- Multi-threshold analysis
- Fine-grained error analysis
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Configure logging early
logger = logging.getLogger(__name__)

# Required dependencies - SentenceTransformers is mandatory
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    logger.debug("✅ SentenceTransformers and dependencies available")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ SentenceTransformers is required but not available: {e}")
    logger.error("Please install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

    # Create dummy classes to avoid import errors, but they will raise runtime errors
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SentenceTransformers is required but not installed. Install with: pip install sentence-transformers"
            )

    def cosine_similarity(*args, **kwargs):
        raise ImportError(
            "scikit-learn is required but not installed. Install with: pip install scikit-learn"
        )


from tqdm import tqdm


class SemanticMatcher:
    """Handles semantic similarity computation for object labels."""

    def __init__(
        self,
        model_name: str = "/data4/swift/model_cache/sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize semantic matcher with SentenceTransformer model."""
        self.model_name = model_name
        self.model = None

        try:
            self.model = SentenceTransformer(model_name)
            self.cosine_similarity = cosine_similarity
            logger.info(f"Initialized SentenceTransformer model: {model_name}")
        except ImportError as e:
            logger.error(f"SentenceTransformer is required but not available: {e}")
            logger.error("Please install: pip install sentence-transformers")
            raise ImportError(
                "SentenceTransformer is required for semantic matching. "
                "Please install with: pip install sentence-transformers"
            ) from e

        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}

    def compute_similarity(self, desc1: str, desc2: str) -> float:
        """
        Compute semantic similarity between two descriptions.

        Args:
            desc1: First description
            desc2: Second description

        Returns:
            Similarity score between 0 and 1
        """
        return self._transformer_similarity(desc1, desc2)

    def _transformer_similarity(self, desc1: str, desc2: str) -> float:
        """Compute similarity using SentenceTransformer embeddings."""
        if self.model is None:
            raise RuntimeError("SentenceTransformer model not initialized")

        # Handle edge cases
        if not desc1.strip() or not desc2.strip():
            return 0.0
        if desc1.strip() == desc2.strip():
            return 1.0

        # Get embeddings (with caching)
        emb1 = self._get_embedding(desc1)
        emb2 = self._get_embedding(desc2)

        # Compute cosine similarity
        similarity = self.cosine_similarity([emb1], [emb2])[0][0]

        # Ensure similarity is in [0, 1] range
        return max(0.0, min(1.0, similarity))

    def _get_embedding(self, text: str):
        """Get embedding for text with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.model.encode(text)
        self._embedding_cache[text] = embedding
        return embedding

    def compute_hierarchical_similarity(
        self, desc1: str, desc2: str
    ) -> Dict[str, float]:
        """
        Compute hierarchical similarity for structured labels.

        Handles labels like "螺丝连接点/BBU安装螺丝/连接正确" by comparing
        at different levels of the hierarchy.

        Returns:
            Dict with 'exact', 'partial', 'semantic' similarity scores
        """
        # Clean and normalize
        desc1 = str(desc1).strip()
        desc2 = str(desc2).strip()

        # Exact match
        if desc1 == desc2:
            return {"exact": 1.0, "partial": 1.0, "semantic": 1.0}

        # Parse hierarchical structure
        parts1 = [p.strip() for p in desc1.split("/") if p.strip()]
        parts2 = [p.strip() for p in desc2.split("/") if p.strip()]

        # Partial match - check if any hierarchical parts match
        partial_score = 0.0
        if parts1 and parts2:
            # Check for common parts
            common_parts = set(parts1) & set(parts2)
            if common_parts:
                # Weight by position (earlier parts more important)
                for i, part1 in enumerate(parts1):
                    for j, part2 in enumerate(parts2):
                        if part1 == part2:
                            # Higher weight for matching earlier parts
                            weight = 1.0 / (1 + min(i, j))
                            partial_score = max(partial_score, weight)

        # Semantic similarity
        semantic_score = self.compute_similarity(desc1, desc2)

        # Also compute semantic similarity between individual parts
        if parts1 and parts2:
            part_similarities = []
            for part1 in parts1:
                for part2 in parts2:
                    part_similarities.append(self.compute_similarity(part1, part2))
            if part_similarities:
                # Take max similarity among all part pairs
                max_part_similarity = max(part_similarities)
                semantic_score = max(semantic_score, max_part_similarity)

        return {
            "exact": 0.0,  # Already handled above
            "partial": partial_score,
            "semantic": semantic_score,
        }

    def categorize_object(self, description: Any) -> str:  # type: ignore[override]
        """Roughly categorize an open-vocabulary label for reporting purposes.

        This is **only** used for aggregating category-wise metrics and has **no**
        impact on the IoU/semantic matching itself.  We therefore keep it very
        forgiving: if the label is not a string (e.g. list, None) we coerce it to
        string.  Unknown labels fall back to "其他" (others).
        """
        if description is None:
            return "其他"

        # Some datasets mistakenly store label as list/tuple; join them.
        if isinstance(description, (list, tuple)):
            description = " ".join(map(str, description))

        # Fallback: ensure we are dealing with a string.
        description = str(description)

        desc_lower = description.lower()

        category_patterns = {
            "螺丝": ["螺丝", "screw"],
            "BBU": ["bbu", "基带处理单元"],
            "线缆": ["线缆", "cable", "光纤"],
            "标签": ["标签", "label"],
            "机柜": ["机柜", "cabinet"],
            "挡风板": ["挡风板", "windshield"],
        }

        for category, patterns in category_patterns.items():
            if any(pattern in desc_lower for pattern in patterns):
                return category

        return "其他"

    def extract_known_vocabulary(self, ground_truths: List[List[Dict]]) -> Set[str]:
        """Extract known vocabulary from ground truth data."""
        known_descriptions = set()
        for gt_objects in ground_truths:
            for obj in gt_objects:
                if "label" in obj:
                    known_descriptions.add(str(obj["label"]).strip())
        return known_descriptions


class COCOStyleMetrics:
    """COCO-style evaluation metrics for dense captioning/grounding."""

    def __init__(
        self,
        iou_thresholds: List[float] = None,
        semantic_threshold: float = 0.7,
        enable_soft_matching: bool = True,
        enable_hierarchical: bool = True,
        enable_novel_detection: bool = True,
        use_individual_categories: bool = True,
    ):
        """
        Initialize COCO-style metrics calculator with enhanced features.

        Args:
            iou_thresholds: List of IoU thresholds for evaluation
            semantic_threshold: Threshold for semantic similarity matching
            enable_soft_matching: Use soft semantic scores instead of binary threshold
            enable_hierarchical: Enable hierarchical label matching
            enable_novel_detection: Track novel object detection performance
            use_individual_categories: If True, treat each unique label as its own category (default).
                                         If False, group labels into broad categories.
        """
        if iou_thresholds is None:
            # COCO-style thresholds: 0.5:0.05:0.95
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

        self.semantic_threshold = semantic_threshold
        self.enable_soft_matching = enable_soft_matching
        self.enable_hierarchical = enable_hierarchical
        self.enable_novel_detection = enable_novel_detection
        self.use_individual_categories = use_individual_categories

        self.semantic_matcher = SemanticMatcher()

        # Additional semantic thresholds for multi-threshold analysis
        self.semantic_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]

        # Error categorization
        self.error_categories = {
            "localization": 0,  # Good semantic match but poor localization
            "classification": 0,  # Good localization but poor semantic match
            "background": 0,  # False positive (no matching GT)
            "missed": 0,  # False negative (unmatched GT)
        }

        logger.info(
            f"Initialized COCO-style metrics with {len(self.iou_thresholds)} IoU thresholds"
        )
        logger.info(f"Semantic similarity threshold: {semantic_threshold}")
        logger.info(
            f"Enhanced features - Soft matching: {enable_soft_matching}, "
            f"Hierarchical: {enable_hierarchical}, Novel detection: {enable_novel_detection}"
        )
        logger.info(
            f"Category mode: {'Individual labels' if use_individual_categories else 'Grouped categories'}"
        )

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        if len(box1) != 4 or len(box2) != 4:
            raise ValueError(
                f"Bounding boxes must have 4 coordinates, got {len(box1)} and {len(box2)}"
            )

        # Assuming bbox format = [x_min, y_min, x_max, y_max]
        x1_1, y1_1, x2_1, y2_1 = box1  # type: ignore[misc]
        x1_2, y1_2, x2_2, y2_2 = box2  # type: ignore[misc]

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def validate_and_fix_bbox(
        self, bbox: Any, image_width: int = 1000, image_height: int = 1000
    ) -> List[float] | None:
        """
        Validate and attempt to fix bounding box coordinates from LLM output.

        Args:
            bbox: Bounding box in various possible formats
            image_width: Maximum image width for bounds checking
            image_height: Maximum image height for bounds checking

        Returns:
            Valid [x_min, y_min, x_max, y_max] bbox or None if unfixable
        """
        try:
            # Handle different input types
            if bbox is None:
                return None

            # Convert to list if needed
            if isinstance(bbox, str):
                # Try to parse string representations like "[10,20,30,40]" or "10,20,30,40"
                bbox = bbox.strip().strip("[](){}").replace(" ", "")
                coords = [float(x) for x in bbox.split(",")]
            elif isinstance(bbox, (tuple, list)):
                coords = [float(x) for x in bbox]
            else:
                return None

            # Must have exactly 4 coordinates
            if len(coords) != 4:
                return None

            x1, y1, x2, y2 = coords

            # Check for NaN or infinite values
            if not all(
                isinstance(coord, (int, float))
                and not (coord != coord or abs(coord) == float("inf"))
                for coord in [x1, y1, x2, y2]
            ):
                return None

            # Ensure coordinates are in correct order (x_min <= x_max, y_min <= y_max)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Check for zero-area boxes
            if x_min >= x_max or y_min >= y_max:
                return None

            # Clamp coordinates to image bounds
            x_min = max(0, min(x_min, image_width))
            y_min = max(0, min(y_min, image_height))
            x_max = max(0, min(x_max, image_width))
            y_max = max(0, min(y_max, image_height))

            # Final check after clamping
            if x_min >= x_max or y_min >= y_max:
                return None

            return [x_min, y_min, x_max, y_max]

        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Failed to validate bbox {bbox}: {e}")
            return None

    def validate_object(
        self,
        obj: Dict,
        obj_type: str = "object",
        sample_idx: int = -1,
        obj_idx: int = -1,
    ) -> Dict | None:
        """
        Validate and clean an object (prediction or ground truth) from LLM output.

        Args:
            obj: Object dictionary with bbox_2d and label fields
            obj_type: Type description for logging ("prediction" or "ground_truth")
            sample_idx: Sample index for logging
            obj_idx: Object index within sample for logging

        Returns:
            Cleaned object dictionary or None if invalid
        """
        if not isinstance(obj, dict):
            logger.debug(f"Sample {sample_idx}, {obj_type} {obj_idx}: Not a dictionary")
            return None

        # Check required fields
        if "bbox_2d" not in obj or "label" not in obj:
            logger.debug(
                f"Sample {sample_idx}, {obj_type} {obj_idx}: Missing bbox_2d or label field"
            )
            return None

        # Validate bounding box
        valid_bbox = self.validate_and_fix_bbox(obj["bbox_2d"])
        if valid_bbox is None:
            logger.debug(
                f"Sample {sample_idx}, {obj_type} {obj_idx}: Invalid bbox_2d: {obj['bbox_2d']}"
            )
            return None

        # Validate label
        label = obj["label"]
        if label is None or (isinstance(label, str) and not label.strip()):
            logger.debug(
                f"Sample {sample_idx}, {obj_type} {obj_idx}: Empty or None label"
            )
            return None

        # Return cleaned object
        cleaned_obj = {"bbox_2d": valid_bbox, "label": str(label).strip()}

        # Preserve any additional fields
        for key, value in obj.items():
            if key not in ["bbox_2d", "label"]:
                cleaned_obj[key] = value

        return cleaned_obj

    def get_object_category(self, description: Any) -> str:
        """
        Get the category for an object based on the categorization mode.

        Args:
            description: Object label/description

        Returns:
            Category string - either the original label (individual mode)
            or a grouped category (grouped mode)
        """
        if self.use_individual_categories:
            # Treat each unique label as its own category
            if description is None:
                return "None"

            # Handle list/tuple labels by joining them
            if isinstance(description, (list, tuple)):
                description = " ".join(map(str, description))

            # Return the cleaned label as-is
            return str(description).strip()
        else:
            # Use the existing grouped categorization logic
            return self.semantic_matcher.categorize_object(description)

    # ----------------------------------------------
    # NEW: Utility for recovering truncated JSON
    # ----------------------------------------------
    def _salvage_truncated_json_array(self, json_str: str) -> List[Dict]:
        """Attempt to salvage a partially generated JSON list.

        Some model generations are cut off due to max-token limits, leaving the
        JSON array without its closing brackets (e.g. '[{...}, {').  This helper
        trims the dangling, incomplete tail so that up-to-that-point valid
        objects are still considered during evaluation.

        Returns
        -------
        List[Dict]
            The recovered list of objects.  Returns an empty list if salvage
            fails or if the input is not recognised as a JSON array.
        """
        txt = json_str.strip()
        # Fast fail: we only handle arrays that start with '['
        if not txt.startswith("["):
            return []

        # Enhanced recovery attempts
        recovery_attempts = []

        # Try original if it already has closing bracket
        if txt.endswith("]"):
            recovery_attempts.append(txt)

        # Try adding closing bracket
        recovery_attempts.append(txt + "]")

        # Try removing trailing comma and adding bracket
        if txt.endswith(","):
            recovery_attempts.append(txt[:-1] + "]")

        # Try removing incomplete object and adding bracket
        if "{" in txt and not txt.endswith("}"):
            # Find last complete object
            last_complete_brace = txt.rfind("}")
            if last_complete_brace > 0:
                # Check if there's a comma after the last complete object
                after_brace = txt[last_complete_brace + 1 :].strip()
                if after_brace.startswith(","):
                    recovery_attempts.append(txt[: last_complete_brace + 1] + "]")
                else:
                    recovery_attempts.append(txt[: last_complete_brace + 1] + "]")

        # Try each recovery attempt
        for attempt in recovery_attempts:
            try:
                data = json.loads(attempt)
                if isinstance(data, list):
                    # Filter out any non-dict items for safety
                    valid_items = [item for item in data if isinstance(item, dict)]
                    if valid_items:
                        logger.debug(
                            f"Salvaged {len(valid_items)} objects from truncated JSON"
                        )
                        return valid_items
            except json.JSONDecodeError:
                continue

        # Fallback: iterative trimming approach
        txt = json_str.strip()
        if not txt.endswith("]"):
            txt += "]"

        while len(txt) > 2:  # minimal non-empty array is '[]'
            try:
                data = json.loads(txt)
                if isinstance(data, list):
                    valid_items = [item for item in data if isinstance(item, dict)]
                    if valid_items:
                        logger.debug(
                            f"Salvaged {len(valid_items)} objects via iterative trimming"
                        )
                        return valid_items
            except json.JSONDecodeError:
                # Remove the last item candidate: cut at the last comma
                last_comma = txt.rfind(",")
                last_bracket = txt.rfind("[")
                if last_comma == -1 or last_comma < last_bracket:
                    break
                txt = txt[:last_comma] + "]"

        return []

    def match_predictions_to_gt(
        self,
        pred_objects: List[Dict],
        gt_objects: List[Dict],
        iou_threshold: float,
        use_soft_semantic: bool = None,
    ) -> Tuple[List[Tuple[int, int, float, float, Dict]], List[int], List[int]]:
        """
        Match predictions to ground truth objects using IoU and semantic similarity.

        Enhanced with soft semantic matching and hierarchical similarity.

        Returns:
            Tuple of (matches, unmatched_pred_indices, unmatched_gt_indices)
            matches: List of (pred_idx, gt_idx, iou_score, semantic_score, similarity_details)
        """
        # Fast path: nothing to match
        if not pred_objects or not gt_objects:
            return [], list(range(len(pred_objects))), list(range(len(gt_objects)))

        use_soft = (
            self.enable_soft_matching
            if use_soft_semantic is None
            else use_soft_semantic
        )

        # Build list of all candidate pairs
        candidate_pairs: List[Tuple[float, int, int, float, Dict]] = []

        for pred_idx, pred_obj in enumerate(pred_objects):
            for gt_idx, gt_obj in enumerate(gt_objects):
                iou_score = self.calculate_iou(pred_obj["bbox_2d"], gt_obj["bbox_2d"])
                if iou_score < iou_threshold:
                    continue

                # Compute semantic similarity (with hierarchical if enabled)
                if self.enable_hierarchical:
                    similarity_details = (
                        self.semantic_matcher.compute_hierarchical_similarity(
                            pred_obj["label"], gt_obj["label"]
                        )
                    )
                    semantic_score = similarity_details["semantic"]
                else:
                    semantic_score = self.semantic_matcher.compute_similarity(
                        pred_obj["label"], gt_obj["label"]
                    )
                    similarity_details = {"semantic": semantic_score}

                # For soft matching, include all pairs; for hard matching, apply threshold
                if use_soft or semantic_score >= self.semantic_threshold:
                    # Use combined score for ranking (IoU + semantic)
                    combined_score = iou_score * 0.7 + semantic_score * 0.3
                    candidate_pairs.append(
                        (
                            combined_score,
                            pred_idx,
                            gt_idx,
                            semantic_score,
                            similarity_details,
                        )
                    )

        if not candidate_pairs:
            return [], list(range(len(pred_objects))), list(range(len(gt_objects)))

        # Select non-overlapping pairs starting from the highest combined score
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        matches: List[Tuple[int, int, float, float, Dict]] = []
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()

        for (
            combined_score,
            pred_idx,
            gt_idx,
            semantic_score,
            sim_details,
        ) in candidate_pairs:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue

            # Recalculate IoU for the match (since we used combined score for ranking)
            iou_score = self.calculate_iou(
                pred_objects[pred_idx]["bbox_2d"], gt_objects[gt_idx]["bbox_2d"]
            )

            matches.append((pred_idx, gt_idx, iou_score, semantic_score, sim_details))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

        # Remaining unmatched indices
        unmatched_pred = [i for i in range(len(pred_objects)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(gt_objects)) if i not in matched_gt]

        # Categorize errors for fine-grained analysis
        self._categorize_errors(
            pred_objects,
            gt_objects,
            matches,
            unmatched_pred,
            unmatched_gt,
            iou_threshold,
        )

        logger.debug(
            f"Found {len(matches)} matches, {len(unmatched_pred)} unmatched predictions, {len(unmatched_gt)} unmatched GT"
        )

        return matches, unmatched_pred, unmatched_gt

    def _categorize_errors(
        self,
        pred_objects: List[Dict],
        gt_objects: List[Dict],
        matches: List[Tuple[int, int, float, float, Dict]],
        unmatched_pred: List[int],
        unmatched_gt: List[int],
        iou_threshold: float,
    ):
        """Categorize errors for fine-grained analysis."""
        # Analyze matched pairs for localization vs classification errors
        for pred_idx, gt_idx, iou_score, semantic_score, _ in matches:
            if iou_score < 0.7 and semantic_score > 0.8:
                self.error_categories["localization"] += 1
            elif iou_score > 0.7 and semantic_score < 0.5:
                self.error_categories["classification"] += 1

        # Unmatched predictions are background errors
        self.error_categories["background"] += len(unmatched_pred)

        # Unmatched GT are missed detections
        self.error_categories["missed"] += len(unmatched_gt)

    def calculate_ap_ar(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        category: str = None,
    ) -> Dict[str, float]:
        """
        Calculate Precision, Recall, F1, and enhanced metrics for open-vocabulary.

        Enhanced with:
        - Multi-threshold semantic analysis
        - Soft semantic scoring option
        - Novel object detection metrics
        """
        results: Dict[str, float] = {}

        if len(all_predictions) != len(all_ground_truths):
            raise ValueError(
                f"Prediction and ground truth counts must match: {len(all_predictions)} vs {len(all_ground_truths)}"
            )

        logger.info(f"Calculating metrics for {len(all_predictions)} samples")

        # Pre-filter by category if specified
        if category:
            logger.info(f"Filtering for category: {category}")
            filtered_predictions = []
            filtered_ground_truths = []

            for preds, gts in zip(all_predictions, all_ground_truths):
                filtered_preds = [
                    p for p in preds if self.get_object_category(p["label"]) == category
                ]
                filtered_gts = [
                    g for g in gts if self.get_object_category(g["label"]) == category
                ]
                filtered_predictions.append(filtered_preds)
                filtered_ground_truths.append(filtered_gts)

            all_predictions = filtered_predictions
            all_ground_truths = filtered_ground_truths

        # Extract known vocabulary for novel detection
        known_vocabulary = None
        if self.enable_novel_detection:
            known_vocabulary = self.semantic_matcher.extract_known_vocabulary(
                all_ground_truths
            )

        # Standard metrics at different IoU thresholds
        for iou_thresh in tqdm(self.iou_thresholds, desc="Computing metrics"):
            true_positives = 0
            false_positives = 0
            total_gt = 0

            for pred_objects, gt_objects in zip(all_predictions, all_ground_truths):
                total_gt += len(gt_objects)

                matches, unmatched_pred, _ = self.match_predictions_to_gt(
                    pred_objects, gt_objects, iou_thresh
                )

                true_positives += len(matches)
                false_positives += len(unmatched_pred)

            # Precision / Recall / F1
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = true_positives / total_gt if total_gt > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[f"P@{iou_thresh:.2f}"] = precision
            results[f"R@{iou_thresh:.2f}"] = recall
            results[f"F1@{iou_thresh:.2f}"] = f1
            # Backward-compatibility aliases
            results[f"AP@{iou_thresh:.2f}"] = precision
            results[f"AR@{iou_thresh:.2f}"] = recall

        # Multi-threshold semantic analysis
        if self.enable_soft_matching:
            results.update(
                self._compute_semantic_curve(
                    all_predictions, all_ground_truths, self.iou_thresholds[0]
                )
            )

        # Novel object detection metrics
        if self.enable_novel_detection and known_vocabulary:
            results.update(
                self._compute_novel_detection_metrics(
                    all_predictions, all_ground_truths, known_vocabulary
                )
            )

        # Calculate aggregated metrics
        p_values = [v for k, v in results.items() if k.startswith("P@")]
        r_values = [v for k, v in results.items() if k.startswith("R@")]
        f1_values = [v for k, v in results.items() if k.startswith("F1@")]

        results["mAP"] = np.mean(p_values) if p_values else 0.0
        results["mAR"] = np.mean(r_values) if r_values else 0.0
        results["mF1"] = np.mean(f1_values) if f1_values else 0.0

        logger.info(
            f"Calculated mAP: {results['mAP']:.4f}, mAR: {results['mAR']:.4f}, mF1: {results['mF1']:.4f}"
        )

        return results

    def _compute_semantic_curve(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        iou_threshold: float,
    ) -> Dict[str, float]:
        """Compute precision-recall curve at different semantic thresholds."""
        results = {}

        for sem_thresh in self.semantic_thresholds:
            # Temporarily override semantic threshold
            original_threshold = self.semantic_threshold
            self.semantic_threshold = sem_thresh

            true_positives = 0
            false_positives = 0
            total_gt = 0

            for pred_objects, gt_objects in zip(all_predictions, all_ground_truths):
                total_gt += len(gt_objects)

                matches, unmatched_pred, _ = self.match_predictions_to_gt(
                    pred_objects, gt_objects, iou_threshold, use_soft_semantic=False
                )

                true_positives += len(matches)
                false_positives += len(unmatched_pred)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = true_positives / total_gt if total_gt > 0 else 0.0

            results[f"P_sem@{sem_thresh:.1f}"] = precision
            results[f"R_sem@{sem_thresh:.1f}"] = recall

            # Restore original threshold
            self.semantic_threshold = original_threshold

        # Compute area under semantic precision-recall curve
        if len(self.semantic_thresholds) > 1:
            precisions = [results[f"P_sem@{t:.1f}"] for t in self.semantic_thresholds]
            recalls = [results[f"R_sem@{t:.1f}"] for t in self.semantic_thresholds]

            # Simple trapezoidal integration
            auc = 0.0
            for i in range(len(recalls) - 1):
                auc += (
                    (recalls[i + 1] - recalls[i])
                    * (precisions[i] + precisions[i + 1])
                    / 2
                )

            results["semantic_AUC"] = auc

        return results

    def _compute_novel_detection_metrics(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        known_vocabulary: Set[str],
    ) -> Dict[str, float]:
        """Compute metrics for novel object detection."""
        results = {}

        # Separate predictions into known and novel
        known_predictions = []
        novel_predictions = []

        for pred_objects in all_predictions:
            known_preds = []
            novel_preds = []

            for pred in pred_objects:
                label = str(pred["label"]).strip()
                # Check if this is a known label (exact match or high similarity)
                is_known = False

                if label in known_vocabulary:
                    is_known = True
                else:
                    # Check semantic similarity to known vocabulary
                    max_similarity = 0.0
                    for known_label in known_vocabulary:
                        sim = self.semantic_matcher.compute_similarity(
                            label, known_label
                        )
                        max_similarity = max(max_similarity, sim)

                    if max_similarity > 0.9:  # Very high similarity threshold
                        is_known = True

                if is_known:
                    known_preds.append(pred)
                else:
                    novel_preds.append(pred)

            known_predictions.append(known_preds)
            novel_predictions.append(novel_preds)

        # Compute metrics for known objects
        if any(known_predictions):
            original_novel_flag = self.enable_novel_detection
            self.enable_novel_detection = False
            try:
                known_metrics = self.calculate_ap_ar(
                    known_predictions, all_ground_truths, category=None
                )
            finally:
                # Always restore to avoid side-effects in subsequent calls.
                self.enable_novel_detection = original_novel_flag

            results["known_mAP"] = known_metrics["mAP"]
            results["known_mAR"] = known_metrics["mAR"]

        # Count novel detections
        total_novel = sum(len(preds) for preds in novel_predictions)
        total_predictions = sum(len(preds) for preds in all_predictions)

        results["novel_detection_rate"] = (
            total_novel / total_predictions if total_predictions > 0 else 0.0
        )

        # Semantic diversity of novel predictions
        novel_labels = set()
        for preds in novel_predictions:
            for pred in preds:
                novel_labels.add(str(pred["label"]).strip())

        results["novel_vocabulary_size"] = len(novel_labels)

        return results

    def compute_average_iou_and_semantic(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        iou_threshold: float = None,
    ) -> Tuple[float, float]:
        """
        Compute average IoU and average semantic similarity for matched predictions and ground truths.
        Enhanced with hierarchical similarity details.
        """
        if iou_threshold is None:
            iou_threshold = self.iou_thresholds[0]

        ious = []
        sems = []
        hierarchical_scores = defaultdict(list)

        for pred_objects, gt_objects in zip(all_predictions, all_ground_truths):
            matches, _, _ = self.match_predictions_to_gt(
                pred_objects, gt_objects, iou_threshold
            )
            for _, _, iou_score, sem_score, sim_details in matches:
                ious.append(iou_score)
                sems.append(sem_score)

                # Track hierarchical similarity components if available
                if "partial" in sim_details:
                    hierarchical_scores["partial"].append(sim_details["partial"])

        avg_iou = float(np.mean(ious)) if ious else 0.0
        avg_semantic = float(np.mean(sems)) if sems else 0.0

        # Also compute average hierarchical scores if available
        if hierarchical_scores:
            for key, scores in hierarchical_scores.items():
                avg_score = float(np.mean(scores)) if scores else 0.0
                logger.info(f"Average {key} similarity: {avg_score:.4f}")

        return avg_iou, avg_semantic

    def evaluate_dataset(
        self,
        responses_file: str,
        output_file: str,
        use_semantic: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset with enhanced COCO-style metrics for open-vocabulary.

        Args:
            responses_file: Path to raw responses JSON with 'ground_truth' and 'result' fields
            output_file: Path to save evaluation results
            use_semantic: Optional flag to temporarily override soft semantic matching

        Returns:
            Comprehensive evaluation results with enhanced metrics
        """
        # Optionally override soft semantic matching for this evaluation run
        original_soft_flag = self.enable_soft_matching
        if use_semantic is not None:
            self.enable_soft_matching = bool(use_semantic)

        start_time = time.time()

        logger.info(f"Loading responses from {responses_file}")

        # Robust JSON loading with error recovery
        responses = self._load_responses_robust(responses_file)

        logger.info(
            f"Evaluating {len(responses)} responses with enhanced COCO-style metrics"
        )

        # Reset error categories
        self.error_categories = {k: 0 for k in self.error_categories}

        # Parse responses
        all_predictions = []
        all_ground_truths = []
        skipped_samples = []

        for i, response in enumerate(tqdm(responses, desc="Parsing responses")):
            # Require the strict schema: ground_truth + pred_result
            if "ground_truth" not in response or "pred_result" not in response:
                logger.error(
                    f"Sample {i} missing required 'ground_truth' or prediction field ('pred_result')"
                )
                skipped_samples.append(f"Sample {i}: Missing required fields")
                all_predictions.append([])
                all_ground_truths.append([])
                continue

            try:
                # Parse ground truth
                gt_raw = response["ground_truth"]
                if isinstance(gt_raw, str):
                    gt_objects = json.loads(gt_raw)
                else:
                    gt_objects = gt_raw

                # Validate GT format
                if not isinstance(gt_objects, list):
                    logger.warning(
                        f"Sample {i}: Ground truth is not a list, converting"
                    )
                    gt_objects = []
            except json.JSONDecodeError as e:
                logger.error(f"Sample {i}: Failed to parse ground_truth JSON: {e}")
                logger.error(
                    f"Sample {i}: GT content preview: {response['ground_truth'][:100]}..."
                )
                skipped_samples.append(f"Sample {i}: Malformed ground_truth JSON")
                gt_objects = []
            except Exception as e:
                logger.error(f"Sample {i}: Unexpected error parsing ground_truth: {e}")
                skipped_samples.append(f"Sample {i}: Ground truth parsing error")
                gt_objects = []

            try:
                # Parse predictions (strict schema)
                pred_raw = response["pred_result"]
                pred_objects = self._parse_pred_result_robust(pred_raw, i)

                # Validate prediction format
                if not isinstance(pred_objects, list):
                    logger.warning(f"Sample {i}: Predictions is not a list, converting")
                    pred_objects = []
            except json.JSONDecodeError:
                preview = str(response.get("pred_result", ""))[:100]
                logger.error(f"Sample {i}: Result content preview: {preview}...")
                suffix = str(response.get("pred_result", ""))[-100:]
                logger.error(f"Sample {i}: Result content suffix: ...{suffix}")
                skipped_samples.append(f"Sample {i}: Malformed result JSON")
                pred_objects = []
            except Exception as e:
                logger.error(f"Sample {i}: Unexpected error parsing result: {e}")
                skipped_samples.append(f"Sample {i}: Result parsing error")
                pred_objects = []

            # Validate object format for both GT and predictions
            try:
                # Validate and clean ground truth objects using robust validation
                valid_gt_objects = []
                invalid_gt_count = 0
                for j, obj in enumerate(gt_objects):
                    cleaned_obj = self.validate_object(obj, "ground_truth", i, j)
                    if cleaned_obj is not None:
                        valid_gt_objects.append(cleaned_obj)
                    else:
                        invalid_gt_count += 1

                if invalid_gt_count > 0:
                    logger.warning(
                        f"Sample {i}: Skipped {invalid_gt_count}/{len(gt_objects)} invalid ground truth objects"
                    )

                # Validate and clean prediction objects using robust validation
                valid_pred_objects = []
                invalid_pred_count = 0
                for j, obj in enumerate(pred_objects):
                    cleaned_obj = self.validate_object(obj, "prediction", i, j)
                    if cleaned_obj is not None:
                        valid_pred_objects.append(cleaned_obj)
                    else:
                        invalid_pred_count += 1

                if invalid_pred_count > 0:
                    logger.warning(
                        f"Sample {i}: Skipped {invalid_pred_count}/{len(pred_objects)} invalid prediction objects"
                    )

                all_predictions.append(valid_pred_objects)
                all_ground_truths.append(valid_gt_objects)

            except Exception as e:
                logger.error(f"Sample {i}: Error validating object format: {e}")
                skipped_samples.append(f"Sample {i}: Object validation error")
                all_predictions.append([])
                all_ground_truths.append([])

        # Log parsing summary
        valid_samples = len(
            [
                i
                for i, (p, g) in enumerate(zip(all_predictions, all_ground_truths))
                if p or g
            ]
        )
        logger.info(f"Parsing summary: {valid_samples}/{len(responses)} samples valid")

        if skipped_samples:
            logger.warning(f"Skipped {len(skipped_samples)} problematic samples:")
            for skip_msg in skipped_samples[:10]:  # Show first 10
                logger.warning(f"  - {skip_msg}")
            if len(skipped_samples) > 10:
                logger.warning(f"  - ... and {len(skipped_samples) - 10} more")

        if valid_samples == 0:
            raise ValueError("No valid samples found for evaluation!")

        logger.info("Computing overall metrics...")

        # Calculate overall metrics
        overall_metrics = self.calculate_ap_ar(all_predictions, all_ground_truths)

        # Compute average IoU and semantic similarity
        avg_iou, avg_sem = self.compute_average_iou_and_semantic(
            all_predictions, all_ground_truths
        )
        overall_metrics[f"avg_IoU@{self.iou_thresholds[0]:.2f}"] = avg_iou
        overall_metrics[f"avg_semantic@{self.iou_thresholds[0]:.2f}"] = avg_sem

        # Calculate per-category metrics
        categories = set()
        for gts in all_ground_truths:
            for gt in gts:
                categories.add(self.get_object_category(gt["label"]))

        category_metrics = {}
        for category in categories:
            logger.info(f"Computing metrics for category: {category}")
            category_metrics[category] = self.calculate_ap_ar(
                all_predictions, all_ground_truths, category
            )

        # Compile results
        evaluation_results = {
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
            "error_analysis": dict(self.error_categories),
            "evaluation_info": {
                "total_samples": len(responses),
                "valid_samples": valid_samples,
                "skipped_samples": len(skipped_samples),
                "total_predictions": sum(len(preds) for preds in all_predictions),
                "total_ground_truth": sum(len(gts) for gts in all_ground_truths),
                "iou_thresholds": self.iou_thresholds,
                "semantic_threshold": self.semantic_threshold,
                "semantic_thresholds_analyzed": self.semantic_thresholds,
                "enhanced_features": {
                    "soft_matching": self.enable_soft_matching,
                    "hierarchical": self.enable_hierarchical,
                    "novel_detection": self.enable_novel_detection,
                },
                "evaluation_time_seconds": time.time() - start_time,
                "skipped_sample_details": skipped_samples[:20]
                if skipped_samples
                else [],  # Include first 20 for debugging
            },
        }

        # Save results
        logger.info(f"Saving evaluation results to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        self._log_summary(evaluation_results)

        # Restore original soft-matching setting
        self.enable_soft_matching = original_soft_flag

        return evaluation_results

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log evaluation summary with enhanced metrics."""
        overall = results["overall_metrics"]
        info = results["evaluation_info"]
        errors = results.get("error_analysis", {})

        logger.info("=" * 60)
        logger.info("ENHANCED EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Samples: {info['total_samples']}")
        logger.info(f"Valid Samples: {info['valid_samples']}")
        logger.info(f"Skipped Samples: {info['skipped_samples']}")
        logger.info(f"Total Predictions: {info['total_predictions']}")
        logger.info(f"Total Ground Truth: {info['total_ground_truth']}")
        logger.info(f"Evaluation Time: {info['evaluation_time_seconds']:.2f}s")
        logger.info("-" * 40)
        logger.info("STANDARD METRICS:")
        logger.info(f"  mAP (mean Precision): {overall['mAP']:.4f}")
        logger.info(f"  mAR (mean Recall):    {overall['mAR']:.4f}")
        logger.info(f"  mF1:                  {overall['mF1']:.4f}")
        logger.info(f"  P@0.50: {overall.get('P@0.50', 0):.4f}")
        logger.info(f"  R@0.50: {overall.get('R@0.50', 0):.4f}")
        logger.info(f"  F1@0.50: {overall.get('F1@0.50', 0):.4f}")
        logger.info("-" * 40)
        logger.info("MATCHING QUALITY:")
        logger.info(f"  avg_IoU@0.50: {overall.get('avg_IoU@0.50', 0):.4f}")
        logger.info(f"  avg_semantic@0.50: {overall.get('avg_semantic@0.50', 0):.4f}")

        # Enhanced metrics
        if "semantic_AUC" in overall:
            logger.info("-" * 40)
            logger.info("SEMANTIC ANALYSIS:")
            logger.info(f"  Semantic AUC: {overall['semantic_AUC']:.4f}")
            for thresh in info.get("semantic_thresholds_analyzed", []):
                p_key = f"P_sem@{thresh:.1f}"
                r_key = f"R_sem@{thresh:.1f}"
                if p_key in overall:
                    logger.info(
                        f"  @{thresh:.1f}: P={overall[p_key]:.3f}, R={overall[r_key]:.3f}"
                    )

        if "novel_detection_rate" in overall:
            logger.info("-" * 40)
            logger.info("NOVEL DETECTION:")
            logger.info(
                f"  Novel Detection Rate: {overall['novel_detection_rate']:.3f}"
            )
            logger.info(
                f"  Novel Vocabulary Size: {overall.get('novel_vocabulary_size', 0)}"
            )
            if "known_mAP" in overall:
                logger.info(f"  Known Objects mAP: {overall['known_mAP']:.4f}")
                logger.info(f"  Known Objects mAR: {overall['known_mAR']:.4f}")

        if errors:
            logger.info("-" * 40)
            logger.info("ERROR ANALYSIS:")
            total_errors = sum(errors.values())
            for error_type, count in errors.items():
                pct = (count / total_errors * 100) if total_errors > 0 else 0
                logger.info(f"  {error_type}: {count} ({pct:.1f}%)")

        logger.info("-" * 40)
        logger.info("CATEGORY BREAKDOWN:")
        for category, metrics in results["category_metrics"].items():
            logger.info(
                f"  {category}: mAP={metrics['mAP']:.3f}, mAR={metrics['mAR']:.3f}, mF1={metrics['mF1']:.3f}"
            )

        if info["skipped_samples"] > 0:
            logger.info("-" * 40)
            logger.warning(
                f"⚠️ {info['skipped_samples']} samples were skipped due to parsing errors"
            )
            logger.warning(
                "Check the detailed logs above for specific error information"
            )

        logger.info("=" * 60)

    def _load_responses_robust(self, responses_file: str) -> List[Dict[str, Any]]:
        """
        Robustly load responses from JSON file with error recovery.

        Handles common issues:
        - Incomplete JSON arrays (missing closing bracket)
        - Malformed individual objects
        - Mixed line-delimited and array formats
        """
        try:
            # First, try standard JSON loading
            with open(responses_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Handle empty files
            if not content:
                logger.warning("Empty responses file")
                return []

            # Try direct JSON parse
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    logger.info(f"Successfully loaded {len(data)} responses")
                    return data
                else:
                    logger.warning("JSON file contains non-list data, wrapping in list")
                    return [data] if data else []
            except json.JSONDecodeError as e:
                logger.warning(f"Standard JSON parsing failed: {e}")
                logger.info("Attempting robust recovery...")

            # Attempt recovery for incomplete JSON arrays
            recovered_data = self._recover_incomplete_json_array(content)
            if recovered_data:
                logger.info(
                    f"Recovered {len(recovered_data)} responses from malformed JSON"
                )
                return recovered_data

            # Try line-by-line JSONL format as fallback
            jsonl_data = self._try_jsonl_format(responses_file)
            if jsonl_data:
                logger.info(f"Loaded {len(jsonl_data)} responses as JSONL format")
                return jsonl_data

            # Final fallback: return empty list
            logger.error("All JSON recovery attempts failed")
            return []

        except Exception as e:
            logger.error(f"Critical error loading responses file: {e}")
            return []

    def _recover_incomplete_json_array(self, content: str) -> List[Dict[str, Any]]:
        """Recover from incomplete JSON array (missing closing bracket, etc.)"""
        content = content.strip()

        # Ensure it looks like an array
        if not content.startswith("["):
            return []

        # Try to fix common issues
        fixes_to_try = [
            content,  # Original
            content + "]",  # Missing closing bracket
            content.rstrip(",") + "]",  # Trailing comma + missing bracket
            content.rstrip(","),  # Just trailing comma
        ]

        for attempt in fixes_to_try:
            try:
                data = json.loads(attempt)
                if isinstance(data, list):
                    # Filter out any non-dict items
                    valid_items = [item for item in data if isinstance(item, dict)]
                    if valid_items:
                        logger.info(
                            f"JSON recovery successful with {len(valid_items)} valid items"
                        )
                        return valid_items
            except json.JSONDecodeError:
                continue

        # More aggressive recovery: try to extract complete JSON objects
        return self._extract_json_objects_from_text(content)

    def _extract_json_objects_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Extract individual JSON objects from malformed text"""
        objects = []

        # Look for patterns like {"..."}

        # Find potential JSON object boundaries
        brace_level = 0
        start_pos = None

        for i, char in enumerate(content):
            if char == "{":
                if brace_level == 0:
                    start_pos = i
                brace_level += 1
            elif char == "}":
                brace_level -= 1
                if brace_level == 0 and start_pos is not None:
                    # Found complete object
                    obj_text = content[start_pos : i + 1]
                    try:
                        obj = json.loads(obj_text)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass  # Skip malformed objects
                    start_pos = None

        logger.info(f"Extracted {len(objects)} objects from malformed JSON")
        return objects

    def _try_jsonl_format(self, responses_file: str) -> List[Dict[str, Any]]:
        """Try to parse as JSONL (one JSON object per line)"""
        objects = []

        try:
            with open(responses_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        logger.debug(f"Skipping malformed line {line_no}")
                        continue

        except Exception as e:
            logger.debug(f"JSONL parsing failed: {e}")
            return []

        return objects

    def _parse_pred_result_robust(self, pred_raw: Any, sample_idx: int) -> List[Dict]:
        """
        Robustly parse prediction results with comprehensive error recovery.

        Handles various formats the model might output:
        - Proper JSON arrays
        - Malformed JSON (missing brackets, trailing commas)
        - String representations of lists
        - Single objects that should be wrapped in arrays
        - Mixed formats
        """
        # Handle None or empty
        if pred_raw is None:
            return []

        # If already a list, validate and return
        if isinstance(pred_raw, list):
            return [item for item in pred_raw if isinstance(item, dict)]

        # If it's a dict, wrap in list
        if isinstance(pred_raw, dict):
            return [pred_raw]

        # Convert to string for parsing
        pred_str = str(pred_raw).strip()

        if not pred_str:
            return []

        # Try standard JSON parsing first
        try:
            parsed = json.loads(pred_str)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                logger.warning(
                    f"Sample {sample_idx}: Parsed prediction is not list or dict: {type(parsed)}"
                )
                return []
        except json.JSONDecodeError:
            pass  # Continue to recovery methods

        # Try salvaging truncated JSON array
        salvaged = self._salvage_truncated_json_array(pred_str)
        if salvaged:
            logger.warning(
                f"Sample {sample_idx}: Recovered {len(salvaged)} predictions from truncated JSON"
            )
            return salvaged

        # Try to handle common malformed patterns
        recovered = self._recover_malformed_predictions(pred_str, sample_idx)
        if recovered:
            return recovered

        # Final fallback: try to extract any JSON-like objects
        extracted = self._extract_json_objects_from_text(pred_str)
        if extracted:
            logger.warning(
                f"Sample {sample_idx}: Extracted {len(extracted)} objects from malformed text"
            )
            return extracted

        logger.error(
            f"Sample {sample_idx}: Could not parse prediction result: {pred_str[:100]}..."
        )
        return []

    def _recover_malformed_predictions(
        self, pred_str: str, sample_idx: int
    ) -> List[Dict]:
        """Attempt to recover from common malformed prediction patterns."""
        # Common issues:
        # 1. Missing outer brackets: {"bbox": [1,2,3,4], "label": "test"}, {"bbox": [5,6,7,8], "label": "test2"}
        # 2. Python-style formatting instead of JSON
        # 3. Extra text before/after JSON

        recovery_attempts = []

        # Try wrapping in brackets if it looks like comma-separated objects
        if "{" in pred_str and "}" in pred_str and not pred_str.strip().startswith("["):
            recovery_attempts.append(f"[{pred_str}]")

        # Try fixing Python-style formatting (single quotes to double quotes)
        if "'" in pred_str:
            # Simple replacement (not perfect but handles many cases)
            fixed = pred_str.replace("'", '"')
            recovery_attempts.append(fixed)
            if not fixed.strip().startswith("["):
                recovery_attempts.append(f"[{fixed}]")

        # Try extracting JSON from text that might have extra content
        import re

        json_match = re.search(r"\[.*\]", pred_str, re.DOTALL)
        if json_match:
            recovery_attempts.append(json_match.group(0))

        # Try each recovery attempt
        for attempt in recovery_attempts:
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, list):
                    valid_items = [item for item in parsed if isinstance(item, dict)]
                    if valid_items:
                        logger.warning(
                            f"Sample {sample_idx}: Recovered {len(valid_items)} predictions via pattern fixing"
                        )
                        return valid_items
                elif isinstance(parsed, dict):
                    logger.warning(
                        f"Sample {sample_idx}: Recovered 1 prediction via pattern fixing"
                    )
                    return [parsed]
            except json.JSONDecodeError:
                continue

        return []
