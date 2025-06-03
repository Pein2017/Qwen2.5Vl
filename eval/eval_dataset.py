#!/usr/bin/env python3
"""
Pure evaluation pipeline for Qwen2.5-VL model.

This script ONLY handles evaluation - no inference, just parsing and metrics.
It takes raw responses from infer_dataset.py and calculates evaluation metrics.

Role: Pure evaluation (parsing + metrics calculation)
Input: Raw responses JSON file from infer_dataset.py
Output: Evaluation metrics and results

Usage:
    # Step 1: Generate raw responses (separate step)
    python eval/infer_dataset.py \
        --model_path output/checkpoint-XXX \
        --validation_jsonl 521_qwen_val.jsonl \
        --output_file raw_responses.json

    # Step 2: Evaluate responses (this script)
    python eval/eval_dataset.py \
        --responses_file raw_responses.json \
        --validation_jsonl 521_qwen_val.jsonl \
        --output_dir eval_results \
        --iou_threshold 0.5
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_evaluation(
    responses_file: str,
    metrics_file: str,
    iou_threshold: float,
) -> bool:
    """
    Run evaluation using metrics.py

    Role: Pure evaluation - parse responses and calculate metrics

    Returns:
        True if successful, False otherwise
    """
    print("📊 Parsing raw responses and calculating metrics...")

    cmd = [
        sys.executable,
        "eval/metrics.py",
        "--responses_file",
        responses_file,
        "--output_file",
        metrics_file,
        "--iou_threshold",
        str(iou_threshold),
    ]

    try:
        _ = subprocess.run(args=cmd, check=True, capture_output=True, text=True)
        print("✅ Metrics calculation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Metrics calculation failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main evaluation function - pure evaluation only."""
    parser = argparse.ArgumentParser(
        description="Pure evaluation pipeline for Qwen2.5-VL model (no inference)"
    )

    # Input configuration
    parser.add_argument(
        "--responses_file",
        type=str,
        required=True,
        help="Path to raw responses JSON file from infer_dataset.py",
    )
    parser.add_argument(
        "--validation_jsonl",
        type=str,
        required=True,
        help="Path to validation JSONL file (for reference)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation outputs (default: eval_results)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Optional run name for organizing outputs",
    )

    # Evaluation configuration
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a detection correct (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.responses_file):
        raise FileNotFoundError(f"Responses file does not exist: {args.responses_file}")

    if not os.path.exists(args.validation_jsonl):
        raise FileNotFoundError(
            f"Validation file does not exist: {args.validation_jsonl}"
        )

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate run name if not provided
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output files
    metrics_file = os.path.join(args.output_dir, f"evaluation_results_{run_name}.json")

    print("🎯 PURE EVALUATION MODE:")
    print(f"📁 Input responses: {args.responses_file}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🏷️  Run name: {run_name}")
    print(f"📄 Metrics file: {metrics_file}")
    print(f"🎯 IoU threshold: {args.iou_threshold}")
    print("🔧 Role: ONLY parse responses and calculate metrics (no inference)")
    print("=" * 60)

    # Run evaluation (pure evaluation only)
    success = run_evaluation(
        responses_file=args.responses_file,
        metrics_file=metrics_file,
        iou_threshold=args.iou_threshold,
    )

    if not success:
        print("❌ Evaluation failed")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ PURE EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Input responses: {args.responses_file}")
    print(f"📊 Evaluation results: {metrics_file}")
    print(f"🎯 IoU threshold: {args.iou_threshold}")
    print("🔧 Role: Pure evaluation (no inference performed)")

    # Read and display final metrics
    try:
        import json

        with open(metrics_file, "r") as f:
            results = json.load(f)

        overall_metrics = results.get("overall_metrics", {})
        print("📈 Final Results:")
        print(f"   - Precision: {overall_metrics.get('precision', 0):.4f}")
        print(f"   - Recall: {overall_metrics.get('recall', 0):.4f}")
        print(f"   - F1 Score: {overall_metrics.get('f1', 0):.4f}")

        # Show dataset stats
        dataset_stats = results.get("dataset_stats", {})
        print("📊 Dataset Statistics:")
        print(f"   - Total GT objects: {dataset_stats.get('total_gt_objects', 0)}")
        print(
            f"   - Total predicted objects: {dataset_stats.get('total_pred_objects', 0)}"
        )
        print(f"   - Total matches: {dataset_stats.get('total_matches', 0)}")

    except Exception as e:
        print(f"⚠️  Could not read final metrics: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
