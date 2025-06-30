"""
Evaluation utilities for Qwen2.5-VL dense captioning/grounding tasks.

This module contains utilities for evaluation, specifically focused on
logging and basic file operations for the evaluation pipeline.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List


class EvaluationLogger:
    """Simple logger for evaluation tasks."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_name: str = None,
        verbose: bool = True,
    ):
        self.log_dir = log_dir
        self.verbose = verbose

        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename if not provided
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"evaluation_{timestamp}.log"

        self.log_file = os.path.join(log_dir, log_name)

        # Setup logger
        self.logger = logging.getLogger(f"eval_{id(self)}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message: str, level: str = "info"):
        """Log a message at the specified level."""
        if level.lower() == "debug":
            self.logger.debug(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


def load_responses_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate responses file for evaluation.

    Args:
        file_path: Path to responses JSON file

    Returns:
        List of response dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Responses file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            responses = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to decode {file_path}: {e.msg}", e.doc, e.pos
            )

    if not isinstance(responses, list):
        raise ValueError(f"Expected list of responses, got {type(responses)}")

    if not responses:
        logging.warning("Responses file is empty, returning empty list.")
        return []

    processed_responses = []
    for i, item in enumerate(responses):
        if not isinstance(item, dict):
            logging.warning(f"Skipping non-dict item at index {i}")
            continue

        required_fields = ["ground_truth", "result"]
        if not all(field in item for field in required_fields):
            logging.warning(f"Skipping item at index {i} due to missing fields")
            continue

        try:
            # Parse nested JSON strings
            item["ground_truth"] = json.loads(item["ground_truth"])
            item["result"] = json.loads(item["result"])
            processed_responses.append(item)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(
                f"Skipping item at index {i} due to parsing error in 'ground_truth' or 'result': {e}"
            )
            continue

    if not processed_responses:
        raise ValueError("No valid responses found after parsing.")

    return processed_responses


def validate_object_format(obj: Dict[str, Any]) -> None:
    """
    Validate object format for dense captioning/grounding.

    Args:
        obj: Object dictionary to validate

    Raises:
        ValueError: If object format is invalid
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object to be dict, got {type(obj)}")

    required_fields = ["bbox_2d", "label"]
    for field in required_fields:
        if field not in obj:
            raise ValueError(f"Object missing required field: {field}")

    bbox = obj["bbox_2d"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox_2d must be list/tuple of 4 numbers, got {bbox}")

    label = obj["label"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"label must be non-empty string, got {label}")


def save_evaluation_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_file: Path to save results
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def format_evaluation_summary(results: Dict[str, Any]) -> str:
    """
    Format evaluation results into a readable summary.

    Args:
        results: Evaluation results dictionary

    Returns:
        Formatted summary string
    """
    overall = results.get("overall_metrics", {})
    info = results.get("evaluation_info", {})
    category_metrics = results.get("category_metrics", {})

    summary_lines = [
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60,
        f"Samples: {info.get('total_samples', 0)}",
        f"Total Predictions: {info.get('total_predictions', 0)}",
        f"Total Ground Truth: {info.get('total_ground_truth', 0)}",
        f"Evaluation Time: {info.get('evaluation_time_seconds', 0):.2f}s",
        "-" * 40,
        f"mAP: {overall.get('mAP', 0):.4f}",
        f"mAR: {overall.get('mAR', 0):.4f}",
        f"AP@0.50: {overall.get('AP@0.50', 0):.4f}",
        f"AP@0.75: {overall.get('AP@0.75', 0):.4f}",
        "-" * 40,
    ]

    # Add category breakdown
    if category_metrics:
        summary_lines.append("Category Breakdown:")
        for category, metrics in category_metrics.items():
            summary_lines.append(
                f"  {category}: mAP={metrics.get('mAP', 0):.3f}, mAR={metrics.get('mAR', 0):.3f}"
            )

    summary_lines.append("=" * 60)

    return "\n".join(summary_lines)
