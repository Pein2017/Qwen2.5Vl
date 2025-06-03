#!/usr/bin/env python3
"""
COCO-style metrics for multimodal object detection with open-vocabulary evaluation.

This module provides comprehensive evaluation metrics similar to COCO, but adapted
for telecommunications equipment detection with semantic description matching.

Features:
- Multi-threshold IoU evaluation (AP@0.5, AP@0.75, AP@0.5:0.95)
- Semantic similarity evaluation for open-vocabulary descriptions
- Per-category analysis for telecommunications equipment types
- COCO-style Average Precision (AP) and Average Recall (AR) metrics
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

os.environ["HF_HOME"] = "/data4/swift/model_cache/"
# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "Warning: sentence-transformers not available. Semantic similarity will be disabled."
    )

try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some similarity metrics will be disabled.")


class SemanticMatcher:
    """Handle semantic matching for open-vocabulary descriptions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic matcher with sentence transformer model."""
        self.model = None
        self.model_name = model_name

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading semantic similarity model: {model_name}")
                self.model = SentenceTransformer(model_name)
                print("✅ Semantic similarity model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load semantic model: {e}")
                self.model = None
        else:
            print(
                "⚠️ Sentence transformers not available - using rule-based matching only"
            )

    def extract_object_type(self, description: str) -> str:
        """Extract object type from structured description."""
        if isinstance(description, dict):
            return description.get("object_type", "unknown")

        if not isinstance(description, str):
            return "unknown"

        # Extract object_type from "object_type:VALUE;question:...;extra question:..." format
        match = re.search(r"object_type:([^;]+)", description)
        if match:
            return match.group(1).strip()

        # Fallback: return first part if no structured format
        return description.split(";")[0].strip()

    def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate semantic similarity between two descriptions."""
        if not self.model or not SKLEARN_AVAILABLE:
            return self._rule_based_similarity(desc1, desc2)

        try:
            # Extract object types for comparison
            type1 = self.extract_object_type(desc1)
            type2 = self.extract_object_type(desc2)

            # Get embeddings
            embeddings = self.model.encode([type1, type2])

            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return self._rule_based_similarity(desc1, desc2)

    def _rule_based_similarity(self, desc1: str, desc2: str) -> float:
        """Fallback rule-based similarity calculation."""
        type1 = self.extract_object_type(desc1).lower()
        type2 = self.extract_object_type(desc2).lower()

        # Exact match
        if type1 == type2:
            return 1.0

        # Partial match (contains)
        if type1 in type2 or type2 in type1:
            return 0.8

        # Equipment category matching
        telecom_categories = {
            "bbu": ["huawei bbu", "zte bbu", "ericsson bbu", "base band unit"],
            "cable": ["fiber cable", "non-fiber cable", "cpri connection"],
            "screw": ["install screw", "floor screw"],
            "shield": ["bbu shield"],
            "cabinet": ["cabinet"],
            "label": ["label matches"],
            "odf": ["odf connection"],
            "grounding": ["cabinet grounding", "grounding"],
        }

        for category, terms in telecom_categories.items():
            if any(term in type1 for term in terms) and any(
                term in type2 for term in terms
            ):
                return 0.6

        return 0.0


class COCOStyleMetrics:
    """COCO-style evaluation metrics for multimodal object detection."""

    def __init__(
        self, iou_thresholds: List[float] = None, semantic_threshold: float = 0.7
    ):
        """
        Initialize COCO-style metrics calculator.

        Args:
            iou_thresholds: List of IoU thresholds for evaluation
            semantic_threshold: Threshold for semantic similarity matching
        """
        if iou_thresholds is None:
            # COCO-style thresholds: 0.5:0.05:0.95
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

        self.semantic_threshold = semantic_threshold
        self.semantic_matcher = SemanticMatcher()

        # Equipment categories for per-category analysis
        self.equipment_categories = {
            "bbu": ["huawei bbu", "zte bbu", "ericsson bbu"],
            "cable": ["fiber cable", "non-fiber cable"],
            "connection": ["cpri connection", "odf connection"],
            "screw": ["install screw", "floor screw"],
            "shield": ["bbu shield"],
            "cabinet": ["cabinet"],
            "label": ["label matches"],
            "grounding": ["grounding"],
        }

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def categorize_object(self, description: str) -> str:
        """Categorize object based on description."""
        obj_type = self.semantic_matcher.extract_object_type(description).lower()

        for category, terms in self.equipment_categories.items():
            if any(term in obj_type for term in terms):
                return category

        return "other"

    def match_predictions_to_gt(
        self,
        pred_objects: List[Dict],
        gt_objects: List[Dict],
        iou_threshold: float,
        use_semantic: bool = True,
    ) -> Tuple[List[Tuple[int, int, float, float]], List[int], List[int]]:
        """
        Match predictions to ground truth objects.

        Returns:
            Tuple of (matches, unmatched_pred_indices, unmatched_gt_indices)
            matches: List of (pred_idx, gt_idx, iou_score, semantic_score)
        """
        matches = []
        matched_gt = set()
        matched_pred = set()

        # Calculate all pairwise IoU and semantic similarities
        similarities = []
        for i, pred_obj in enumerate(pred_objects):
            if "bbox_2d" not in pred_obj:
                continue

            for j, gt_obj in enumerate(gt_objects):
                if "bbox_2d" not in gt_obj:
                    continue

                # Calculate IoU
                iou = self.calculate_iou(pred_obj["bbox_2d"], gt_obj["bbox_2d"])

                # Calculate semantic similarity if enabled
                semantic_sim = 0.0
                if (
                    use_semantic
                    and "description" in pred_obj
                    and "description" in gt_obj
                ):
                    semantic_sim = self.semantic_matcher.calculate_semantic_similarity(
                        pred_obj["description"], gt_obj["description"]
                    )

                similarities.append((i, j, iou, semantic_sim))

        # Sort by combined score (IoU + semantic similarity)
        similarities.sort(
            key=lambda x: x[2] + (x[3] if use_semantic else 0), reverse=True
        )

        # Greedy matching
        for pred_idx, gt_idx, iou, semantic_sim in similarities:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue

            # Check if match meets criteria
            iou_ok = iou >= iou_threshold
            semantic_ok = not use_semantic or semantic_sim >= self.semantic_threshold

            if iou_ok and semantic_ok:
                matches.append((pred_idx, gt_idx, iou, semantic_sim))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)

        # Find unmatched indices
        unmatched_pred = [i for i in range(len(pred_objects)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(gt_objects)) if i not in matched_gt]

        return matches, unmatched_pred, unmatched_gt

    def calculate_ap_ar(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        category: str = None,
    ) -> Dict[str, float]:
        """
        Calculate Average Precision (AP) and Average Recall (AR) across all samples.

        Args:
            all_predictions: List of prediction lists (one per sample)
            all_ground_truths: List of ground truth lists (one per sample)
            category: Specific category to evaluate (None for all)

        Returns:
            Dictionary with AP and AR metrics
        """
        results = {}

        for iou_thresh in self.iou_thresholds:
            precisions = []
            recalls = []

            for pred_objects, gt_objects in zip(all_predictions, all_ground_truths):
                # Filter by category if specified
                if category:
                    pred_objects = [
                        obj
                        for obj in pred_objects
                        if self.categorize_object(obj.get("description", ""))
                        == category
                    ]
                    gt_objects = [
                        obj
                        for obj in gt_objects
                        if self.categorize_object(obj.get("description", ""))
                        == category
                    ]

                if not gt_objects:
                    continue

                matches, _, _ = self.match_predictions_to_gt(
                    pred_objects, gt_objects, iou_thresh
                )

                precision = len(matches) / len(pred_objects) if pred_objects else 0.0
                recall = len(matches) / len(gt_objects) if gt_objects else 0.0

                precisions.append(precision)
                recalls.append(recall)

            # Calculate average precision and recall
            avg_precision = np.mean(precisions) if precisions else 0.0
            avg_recall = np.mean(recalls) if recalls else 0.0

            results[f"AP@{iou_thresh:.2f}"] = avg_precision
            results[f"AR@{iou_thresh:.2f}"] = avg_recall

        # Calculate mAP (mean over IoU thresholds)
        ap_values = [results[f"AP@{thresh:.2f}"] for thresh in self.iou_thresholds]
        ar_values = [results[f"AR@{thresh:.2f}"] for thresh in self.iou_thresholds]

        results["mAP"] = np.mean(ap_values)
        results["mAR"] = np.mean(ar_values)
        results["AP@0.5"] = results.get("AP@0.50", 0.0)
        results["AP@0.75"] = results.get("AP@0.75", 0.0)

        return results

    def evaluate_dataset(
        self, responses_file: str, output_file: str, use_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset with COCO-style metrics.

        Args:
            responses_file: Path to raw responses JSON
            output_file: Path to save evaluation results
            use_semantic: Whether to use semantic similarity matching

        Returns:
            Comprehensive evaluation results
        """
        print(f"📊 Loading responses from {responses_file}")

        with open(responses_file, "r", encoding="utf-8") as f:
            responses = json.load(f)

        print(f"🔢 Evaluating {len(responses)} responses with COCO-style metrics")

        # Parse all predictions and ground truths
        from .metrics import ResponseParser

        parser = ResponseParser()

        all_predictions = []
        all_ground_truths = []

        print("📝 Parsing responses...")
        for response in tqdm(responses, desc="Parsing"):
            try:
                pred_objects = parser.parse_prediction(response["prediction"])
                gt_objects = parser.parse_prediction(response["ground_truth"])

                all_predictions.append(pred_objects)
                all_ground_truths.append(gt_objects)

            except Exception as e:
                print(f"Error parsing response: {e}")
                all_predictions.append([])
                all_ground_truths.append([])

        # Calculate overall metrics
        print("📊 Calculating overall metrics...")
        overall_metrics = self.calculate_ap_ar(all_predictions, all_ground_truths)

        # Calculate per-category metrics
        print("📋 Calculating per-category metrics...")
        category_metrics = {}
        for category in self.equipment_categories.keys():
            category_metrics[category] = self.calculate_ap_ar(
                all_predictions, all_ground_truths, category
            )

        # Calculate semantic-only metrics (if enabled)
        semantic_metrics = {}
        if use_semantic:
            print("🧠 Calculating semantic-only metrics...")
            # Evaluate with very low IoU threshold to focus on semantic matching
            temp_thresholds = self.iou_thresholds
            self.iou_thresholds = [0.1]  # Very permissive IoU
            semantic_metrics = self.calculate_ap_ar(all_predictions, all_ground_truths)
            self.iou_thresholds = temp_thresholds

        # Compile results
        results = {
            "evaluation_info": {
                "responses_file": responses_file,
                "total_samples": len(responses),
                "iou_thresholds": self.iou_thresholds,
                "semantic_threshold": self.semantic_threshold,
                "use_semantic": use_semantic,
            },
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
            "semantic_metrics": semantic_metrics,
            "equipment_categories": list(self.equipment_categories.keys()),
        }

        # Save results
        print(f"💾 Saving COCO-style evaluation results to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("=" * 80)
        print("COCO-STYLE EVALUATION SUMMARY")
        print("=" * 80)

        overall = results["overall_metrics"]
        print("📊 Overall Performance:")
        print(f"   mAP (IoU 0.5:0.95): {overall['mAP']:.4f}")
        print(f"   AP@0.5:            {overall['AP@0.5']:.4f}")
        print(f"   AP@0.75:           {overall['AP@0.75']:.4f}")
        print(f"   mAR (IoU 0.5:0.95): {overall['mAR']:.4f}")

        print("\n📋 Per-Category Performance:")
        for category, metrics in results["category_metrics"].items():
            print(
                f"   {category:12s}: mAP={metrics['mAP']:.3f}, AP@0.5={metrics['AP@0.5']:.3f}"
            )

        if results["semantic_metrics"]:
            semantic = results["semantic_metrics"]
            print("\n🧠 Semantic Matching (IoU@0.1):")
            print(f"   Semantic mAP:      {semantic['mAP']:.4f}")
            print(f"   Semantic mAR:      {semantic['mAR']:.4f}")

        print("=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="COCO-style evaluation for multimodal object detection"
    )

    parser.add_argument(
        "--responses_file",
        type=str,
        required=True,
        help="Path to JSON file with raw responses",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save COCO-style evaluation results",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.7,
        help="Threshold for semantic similarity matching",
    )
    parser.add_argument(
        "--no_semantic",
        action="store_true",
        help="Disable semantic similarity evaluation",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = COCOStyleMetrics(semantic_threshold=args.semantic_threshold)

    # Run evaluation
    evaluator.evaluate_dataset(
        responses_file=args.responses_file,
        output_file=args.output_file,
        use_semantic=not args.no_semantic,
    )


if __name__ == "__main__":
    main()
