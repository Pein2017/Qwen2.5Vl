#!/usr/bin/env python3
"""
Metrics calculation module for Qwen2.5-VL evaluation.

This module only handles parsing responses and calculating metrics.
It does not perform any model inference.

Usage:
    python eval/metrics.py \
        --responses_file raw_responses.json \
        --output_file evaluation_results.json \
        --iou_threshold 0.5
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List
from tqdm import tqdm


class ResponseParser:
    """Parse model responses to extract structured predictions."""

    @staticmethod
    def parse_prediction(prediction_text: str) -> List[Dict[str, Any]]:
        """
        Parse the model's prediction text to extract bounding boxes and labels.

        Args:
            prediction_text: Raw text output from the model or JSON string from ground truth

        Returns:
            List of dictionaries with 'bbox_2d' and 'description' keys
        """
        # If it's already a list (from ground truth), return it directly
        if isinstance(prediction_text, list):
            return prediction_text

        # If it's not a string, convert to string first
        if not isinstance(prediction_text, str):
            prediction_text = str(prediction_text)

        # Try to parse as JSON directly first (handles escaped JSON strings from ground truth)
        try:
            parsed = json.loads(prediction_text)
            if isinstance(parsed, list):
                # Validate that all items are dictionaries with required keys
                valid_objects = []
                for item in parsed:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        valid_objects.append(item)
                return valid_objects
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            # Check for actual truncation (incomplete JSON structure)
            if "[" in prediction_text and not prediction_text.rstrip().endswith("]"):
                print(
                    f"Warning: Response appears truncated - starts with '[' but doesn't end with ']'. Length: {len(prediction_text)}"
                )
                # Try to fix truncated JSON by extracting valid objects
                return ResponseParser._extract_from_truncated_json(prediction_text)
            pass

        # If JSON parsing failed, try to find JSON in the response (for model predictions)
        # Look for JSON array pattern
        json_pattern = r"\[.*?\]"
        matches = re.findall(json_pattern, prediction_text, re.DOTALL)

        if matches:
            # Try to parse the first JSON match
            for json_str in matches:
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        # Validate that all items are dictionaries
                        valid_objects = []
                        for item in parsed:
                            if isinstance(item, dict) and "bbox_2d" in item:
                                valid_objects.append(item)
                        return valid_objects
                    elif isinstance(parsed, dict):
                        return [parsed]
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found, try to extract from markdown code blocks
        code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, prediction_text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, list):
                    # Validate that all items are dictionaries
                    valid_objects = []
                    for item in parsed:
                        if isinstance(item, dict) and "bbox_2d" in item:
                            valid_objects.append(item)
                    return valid_objects
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                continue

        # If we reach here, parsing failed completely
        return []

    @staticmethod
    def _extract_from_truncated_json(prediction_text: str) -> List[Dict[str, Any]]:
        """
        Extract valid objects from truncated JSON response.

        Args:
            prediction_text: Truncated JSON string

        Returns:
            List of valid objects that could be parsed
        """
        valid_objects = []

        # Find the start of the JSON array
        start_idx = prediction_text.find("[")
        if start_idx == -1:
            return []

        # Extract the content after the opening bracket
        content = prediction_text[start_idx + 1 :]

        # Use regex to find complete object patterns
        # Look for complete {"bbox_2d":[...], "description":"..."} patterns
        object_pattern = (
            r'\{"bbox_2d":\s*\[[^\]]+\]\s*,\s*"description":\s*"[^"]*"\s*\}'
        )

        matches = re.findall(object_pattern, content)

        for match in matches:
            try:
                # Try to parse each complete object
                obj = json.loads(match)
                if isinstance(obj, dict) and "bbox_2d" in obj and "description" in obj:
                    valid_objects.append(obj)
            except json.JSONDecodeError:
                continue

        # If no complete objects found, try a more lenient approach
        if not valid_objects:
            # Look for bbox_2d patterns and try to reconstruct objects
            bbox_pattern = r'"bbox_2d":\s*\[([^\]]+)\]'
            desc_pattern = r'"description":\s*"([^"]*)"'

            bbox_matches = re.findall(bbox_pattern, content)
            desc_matches = re.findall(desc_pattern, content)

            # Try to pair bboxes with descriptions
            for i, bbox_str in enumerate(bbox_matches):
                try:
                    # Parse bbox coordinates
                    coords = [int(x.strip()) for x in bbox_str.split(",")]
                    if len(coords) == 4:
                        obj = {"bbox_2d": coords}

                        # Add description if available
                        if i < len(desc_matches):
                            obj["description"] = desc_matches[i]
                        else:
                            obj["description"] = (
                                "object_type:unknown;question:none;extra question:none"
                            )

                        valid_objects.append(obj)
                except (ValueError, IndexError):
                    continue

        if valid_objects:
            print(
                f"Extracted {len(valid_objects)} valid objects from truncated response"
            )
        else:
            print("Could not extract any valid objects from truncated response")

        return valid_objects


class MetricsCalculator:
    """Calculate evaluation metrics for object detection tasks."""

    @staticmethod
    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format

        Returns:
            IoU score between 0 and 1
        """
        # Validate input types and lengths (fail-fast)
        if not isinstance(box1, (list, tuple)) or len(box1) != 4:
            raise ValueError(
                f"box1 must be a list/tuple of 4 elements, got {type(box1)} with length {len(box1) if hasattr(box1, '__len__') else 'unknown'}"
            )

        if not isinstance(box2, (list, tuple)) or len(box2) != 4:
            raise ValueError(
                f"box2 must be a list/tuple of 4 elements, got {type(box2)} with length {len(box2) if hasattr(box2, '__len__') else 'unknown'}"
            )

        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def calculate_sample_metrics(
        pred_objects: List[Dict[str, Any]],
        gt_objects: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a single sample.

        Args:
            pred_objects: Predicted objects with bbox_2d
            gt_objects: Ground truth objects with bbox_2d
            iou_threshold: IoU threshold for considering a detection correct

        Returns:
            Dictionary with sample metrics
        """
        matches = 0
        matched_gt = set()
        matched_pred = set()

        for i, pred_obj in enumerate(pred_objects):
            if not isinstance(pred_obj, dict) or "bbox_2d" not in pred_obj:
                continue

            pred_bbox = pred_obj["bbox_2d"]
            if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_obj in enumerate(gt_objects):
                if j in matched_gt or "bbox_2d" not in gt_obj:
                    continue

                gt_bbox = gt_obj["bbox_2d"]
                if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) != 4:
                    continue

                iou = MetricsCalculator.calculate_iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                matches += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)

        # Calculate metrics
        gt_count = len(gt_objects)
        pred_count = len(pred_objects)

        precision = matches / pred_count if pred_count > 0 else 0.0
        recall = matches / gt_count if gt_count > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "matches": matches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def post_process_predictions(
        pred_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Post-process predictions to remove excessive or duplicate objects.

        Args:
            pred_objects: List of predicted objects

        Returns:
            Filtered list of objects
        """
        if not pred_objects:
            return pred_objects

        # Log original count
        original_count = len(pred_objects)

        # Remove duplicates based on bbox coordinates
        unique_objects = []
        seen_bboxes = set()

        for obj in pred_objects:
            if "bbox_2d" in obj:
                bbox_tuple = tuple(obj["bbox_2d"])
                if bbox_tuple not in seen_bboxes:
                    seen_bboxes.add(bbox_tuple)
                    unique_objects.append(obj)

        # If we have an excessive number of objects (likely repetitive generation),
        # keep only the first reasonable number
        max_reasonable_objects = 50  # Reasonable upper limit for telecom equipment
        if len(unique_objects) > max_reasonable_objects:
            print(
                f"Warning: Excessive objects detected ({len(unique_objects)}), keeping first {max_reasonable_objects}"
            )
            unique_objects = unique_objects[:max_reasonable_objects]

        # Filter out objects with invalid coordinates
        valid_objects = []
        for obj in unique_objects:
            if "bbox_2d" in obj and len(obj["bbox_2d"]) == 4:
                x1, y1, x2, y2 = obj["bbox_2d"]
                # Check for reasonable coordinates (not negative, x2 > x1, y2 > y1)
                if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                    valid_objects.append(obj)

        filtered_count = len(valid_objects)
        if filtered_count != original_count:
            print(
                f"Post-processing: {original_count} -> {filtered_count} objects "
                f"(removed {original_count - filtered_count} duplicates/invalid)"
            )

        return valid_objects


def evaluate_responses(
    responses_file: str, output_file: str, iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate raw responses and calculate metrics.

    Args:
        responses_file: Path to JSON file with raw responses
        output_file: Path to save evaluation results
        iou_threshold: IoU threshold for considering a detection correct

    Returns:
        Dictionary with evaluation results
    """
    print(f"📊 Loading responses from {responses_file}")

    with open(responses_file, "r", encoding="utf-8") as f:
        responses = json.load(f)

    print(f"🔢 Evaluating {len(responses)} responses")

    parser = ResponseParser()
    calculator = MetricsCalculator()

    results = []
    total_gt = 0
    total_pred = 0
    total_matches = 0
    valid_samples = 0
    error_count = 0

    # Create progress bar
    progress_bar = tqdm(
        enumerate(responses),
        total=len(responses),
        desc="📊 Evaluating responses",
        unit="sample",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for i, response in progress_bar:
        try:
            # Parse ground truth and prediction
            gt_objects = parser.parse_prediction(response["ground_truth"])
            pred_objects = parser.parse_prediction(response["prediction"])

            # Post-process predictions
            pred_objects = calculator.post_process_predictions(pred_objects)

            # Calculate metrics
            sample_metrics = calculator.calculate_sample_metrics(
                pred_objects, gt_objects, iou_threshold
            )

            # Combine results
            result = {
                "sample_id": response["sample_id"],
                "image_path": response["image_path"],
                "prediction_text": response["prediction"],
                "gt_response": response["ground_truth"],
                **sample_metrics,
            }

            results.append(result)

            total_gt += sample_metrics["gt_count"]
            total_pred += sample_metrics["pred_count"]
            total_matches += sample_metrics["matches"]
            valid_samples += 1

            # Update progress bar with current stats
            progress_bar.set_postfix(
                {
                    "✅": valid_samples,
                    "❌": error_count,
                    "P": f"{total_matches / total_pred:.2f}"
                    if total_pred > 0
                    else "0.00",
                    "R": f"{total_matches / total_gt:.2f}" if total_gt > 0 else "0.00",
                    "Current": os.path.basename(response.get("image_path", "unknown")),
                }
            )

        except Exception as e:
            error_count += 1
            progress_bar.set_postfix(
                {
                    "✅": valid_samples,
                    "❌": error_count,
                    "Error": str(e)[:15] + "..." if len(str(e)) > 15 else str(e),
                }
            )
            continue

    progress_bar.close()

    # Calculate overall metrics
    overall_precision = total_matches / total_pred if total_pred > 0 else 0.0
    overall_recall = total_matches / total_gt if total_gt > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    # Calculate per-sample averages
    avg_precision = (
        sum(r["precision"] for r in results) / len(results) if results else 0.0
    )
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

    # Create summary
    summary = {
        "evaluation_info": {
            "responses_file": responses_file,
            "iou_threshold": iou_threshold,
            "total_responses": len(responses),
            "valid_samples": valid_samples,
        },
        "dataset_stats": {
            "total_gt_objects": total_gt,
            "total_pred_objects": total_pred,
            "total_matches": total_matches,
        },
        "overall_metrics": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
        "average_metrics": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        },
        "detailed_results": results,
    }

    # Save results
    print(f"💾 Saving evaluation results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total responses: {len(responses)}")
    print(f"Valid samples: {valid_samples}")
    print(f"Total GT objects: {total_gt}")
    print(f"Total predicted objects: {total_pred}")
    print(f"Total matches (IoU >= {iou_threshold}): {total_matches}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print("=" * 60)

    return summary


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate raw responses and calculate metrics"
    )

    # Required arguments
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
        help="Path to save evaluation results",
    )

    # Optional arguments
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a detection correct",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.responses_file):
        raise FileNotFoundError(f"Responses file does not exist: {args.responses_file}")

    # Run evaluation
    evaluate_responses(
        responses_file=args.responses_file,
        output_file=args.output_file,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
