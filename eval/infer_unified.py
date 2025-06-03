#!/usr/bin/env python3
"""
Unified inference script using the same preprocessing pipeline as training.

This script uses the unified preprocessing and inference components from src/
to ensure consistency between training and evaluation, and to benefit from
all training optimizations.

Key benefits:
- Same preprocessing pipeline as training (no inconsistencies)
- Uses training's flash attention optimizations  
- Supports both single-image and multi-image inputs
- Fast inference with training-optimized components
- Eliminates redundancy between training and evaluation

Usage:
    python eval/infer_unified.py \
        --model_path output/checkpoint-XXX \
        --validation_jsonl 521_qwen_val_multi_image.jsonl \
        --output_file results.json \
        --max_new_tokens 1024
"""

import argparse
import os
import sys

# Add src to path to import unified components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import create_unified_inference_engine


def main():
    """Main inference function using unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified inference script using training's optimized pipeline"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--validation_jsonl",
        type=str,
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save results JSON file"
    )

    # Optional arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to load model on"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--data_root", type=str, default="./", help="Root directory for image paths"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    if not os.path.exists(args.validation_jsonl):
        raise FileNotFoundError(
            f"Validation file does not exist: {args.validation_jsonl}"
        )

    print("🎯 UNIFIED INFERENCE MODE:")
    print("   - Uses same preprocessing pipeline as training")
    print("   - Benefits from training's flash attention optimizations")
    print("   - Supports both single-image and multi-image inputs")
    print("   - Eliminates redundancy between training and evaluation")
    print(f"   - Max new tokens: {args.max_new_tokens}")
    print(f"   - Device: {args.device}")

    # Create unified inference engine
    engine = create_unified_inference_engine(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        data_root=args.data_root,
    )

    # Process dataset using unified pipeline
    engine.process_dataset(
        validation_jsonl=args.validation_jsonl,
        output_file=args.output_file,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
