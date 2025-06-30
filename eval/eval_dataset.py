#!/usr/bin/env python3
"""
Pure evaluation pipeline for Qwen2.5-VL dense captioning/grounding task.

This script ONLY handles evaluation - no inference, just parsing and metrics.
It expects raw responses containing a 'ground_truth' field and a prediction field 'pred_result'.

Enhanced features:
- Soft semantic matching
- Hierarchical label matching
- Novel object detection metrics
- Multi-threshold semantic analysis
- Fine-grained error analysis

Usage:
    python eval/eval_dataset.py \
        --responses_file raw_responses.json \
        --output_file eval_results.json \
        --iou_threshold 0.5 \
        --semantic_threshold 0.7 \
        --enable_soft_matching \
        --enable_hierarchical \
        --enable_novel_detection
"""

import argparse
import logging
import os
import sys

# Directly import the main evaluator to avoid an intermediate subprocess
from eval.coco_metrics import COCOStyleMetrics

# New unified logging helpers
from src.logger_utils import configure_global_logging, get_logger


def run_evaluation(
    responses_file: str,
    output_file: str,
    iou_threshold: float = 0.5,
    semantic_threshold: float = 0.7,
    enable_soft_matching: bool = True,
    enable_hierarchical: bool = True,
    enable_novel_detection: bool = True,
    log_level: str = "info",
    verbose: bool = False,
    minimal_output: bool = True,
) -> bool:
    """
    Parse responses JSON and compute COCO-style metrics for dense captioning/grounding.

    Args:
        responses_file: Path to responses JSON with 'ground_truth' and 'result' fields
        output_file: Path to save evaluation results JSON
        iou_threshold: IoU threshold for matching (default: 0.5)
        semantic_threshold: Semantic similarity threshold (default: 0.7)
        enable_soft_matching: Use soft semantic scores instead of binary threshold
        enable_hierarchical: Enable hierarchical label matching
        enable_novel_detection: Track novel object detection performance
        log_level: Logging level (debug or info)
        verbose: Enable verbose logging
        minimal_output: Save only representative (compact) metrics to the output JSON

    Returns:
        True if evaluation completed successfully
    """
    # ------------------------------------------------------------------
    # Configure global logging once for the evaluation run
    # ------------------------------------------------------------------
    configure_global_logging(
        log_dir="logs",
        log_file="run.log",
        log_level=log_level.upper(),
        verbose=verbose,
    )

    # Acquire logger after global configuration
    logger = get_logger("eval")

    logger.info("=" * 60)
    logger.info("QWEN2.5-VL ENHANCED DENSE CAPTIONING/GROUNDING EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Input: {responses_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"IoU Threshold: {iou_threshold}")
    logger.info(f"Semantic Threshold: {semantic_threshold}")
    logger.info("-" * 40)
    logger.info("Enhanced Features:")
    logger.info(f"  Soft Semantic Matching: {enable_soft_matching}")
    logger.info(f"  Hierarchical Matching: {enable_hierarchical}")
    logger.info(f"  Novel Detection: {enable_novel_detection}")
    logger.info("-" * 60)

    # Validate input file exists
    if not os.path.exists(responses_file):
        raise FileNotFoundError(f"Responses file not found: {responses_file}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize COCO-style metrics evaluator with enhanced features
    # Use COCO standard IoU thresholds: 0.5:0.05:0.95
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    logger.info(f"Initializing enhanced COCO-style metrics evaluator...")
    evaluator = COCOStyleMetrics(
        iou_thresholds=iou_thresholds,
        semantic_threshold=semantic_threshold,
        enable_soft_matching=enable_soft_matching,
        enable_hierarchical=enable_hierarchical,
        enable_novel_detection=enable_novel_detection,
    )

    # Run evaluation
    logger.info("Starting evaluation...")

    results = evaluator.evaluate_dataset(
        responses_file, output_file, minimal_output=minimal_output
    )

    logger.info("✅ Evaluation completed successfully!")
    logger.info(f"Results saved to: {output_file}")

    return results is not None


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Enhanced COCO-style evaluation for Qwen2.5-VL dense captioning/grounding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python eval/eval_dataset.py \\
        --responses_file eval_results_chinese/chinese-train_responses.json \\
        --output_file eval_results_chinese/chinese-train_metrics.json

    # With enhanced features for open-vocabulary
    python eval/eval_dataset.py \\
        --responses_file eval_results_chinese/chinese-val_responses.json \\
        --output_file eval_results_chinese/chinese-val_metrics_enhanced.json \\
        --enable_soft_matching \\
        --enable_hierarchical \\
        --enable_novel_detection

    # With custom thresholds and debug logging
    python eval/eval_dataset.py \\
        --responses_file eval_results_chinese/chinese-val_responses.json \\
        --output_file eval_results_chinese/chinese-val_metrics.json \\
        --iou_threshold 0.75 \\
        --semantic_threshold 0.8 \\
        --log_level debug \\
        --verbose
        """,
    )

    parser.add_argument(
        "--responses_file",
        type=str,
        required=True,
        help="Path to JSON file with raw responses containing 'ground_truth' and 'pred_result' fields",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save COCO-style evaluation results JSON",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for object matching (default: 0.5)",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.7,
        help="Semantic similarity threshold for open-vocabulary matching (default: 0.7)",
    )
    parser.add_argument(
        "--enable_soft_matching",
        action="store_true",
        help="Enable soft semantic matching with continuous scores instead of binary threshold",
    )
    parser.add_argument(
        "--enable_hierarchical",
        action="store_true",
        help="Enable hierarchical label matching for structured labels like '螺丝或连接点/BBU安装螺丝'",
    )
    parser.add_argument(
        "--enable_novel_detection",
        action="store_true",
        help="Enable novel object detection metrics to track performance on unseen descriptions",
    )
    parser.add_argument(
        "--disable_soft_matching",
        action="store_true",
        help="Disable soft semantic matching (use binary threshold only)",
    )
    parser.add_argument(
        "--disable_hierarchical",
        action="store_true",
        help="Disable hierarchical label matching",
    )
    parser.add_argument(
        "--disable_novel_detection",
        action="store_true",
        help="Disable novel object detection metrics",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["debug", "info"],
        default="info",
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging with timestamps"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Save only representative (compact) metrics to the output JSON",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 <= args.iou_threshold <= 1.0):
        raise ValueError(
            f"IoU threshold must be between 0.0 and 1.0, got {args.iou_threshold}"
        )

    if not (0.0 <= args.semantic_threshold <= 1.0):
        raise ValueError(
            f"Semantic threshold must be between 0.0 and 1.0, got {args.semantic_threshold}"
        )

    # Handle enable/disable flags (disable takes precedence)
    enable_soft = (
        not args.disable_soft_matching
        if args.disable_soft_matching
        else args.enable_soft_matching
    )
    enable_hier = (
        not args.disable_hierarchical
        if args.disable_hierarchical
        else args.enable_hierarchical
    )
    enable_novel = (
        not args.disable_novel_detection
        if args.disable_novel_detection
        else args.enable_novel_detection
    )

    # Default to True if neither enable nor disable is specified
    if not args.enable_soft_matching and not args.disable_soft_matching:
        enable_soft = True
    if not args.enable_hierarchical and not args.disable_hierarchical:
        enable_hier = True
    if not args.enable_novel_detection and not args.disable_novel_detection:
        enable_novel = True

    # Run evaluation
    success = run_evaluation(
        responses_file=args.responses_file,
        output_file=args.output_file,
        iou_threshold=args.iou_threshold,
        semantic_threshold=args.semantic_threshold,
        enable_soft_matching=enable_soft,
        enable_hierarchical=enable_hier,
        enable_novel_detection=enable_novel,
        log_level=args.log_level,
        verbose=args.verbose,
        minimal_output=args.minimal,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
