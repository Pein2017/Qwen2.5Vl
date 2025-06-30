#!/usr/bin/env python3
"""
Result validation and summary script for inference outputs.
Replaces hardcoded Python snippets in bash scripts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_and_summarize_results(
    output_file: str, show_preview: bool = True
) -> Tuple[int, int, int]:
    """
    Validate inference results and return statistics.

    Args:
        output_file: Path to the JSON results file
        show_preview: Whether to show sample results preview

    Returns:
        Tuple of (total_samples, successful_samples, failed_samples)
    """
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error reading results file: {e}", file=sys.stderr)
        return 0, 0, 0

    if not isinstance(results, list):
        print(
            f"❌ Invalid results format: expected list, got {type(results)}",
            file=sys.stderr,
        )
        return 0, 0, 0

    total = len(results)
    successful = len([r for r in results if "error" not in r and r.get("pred_result")])
    failed = total - successful

    print(f"📊 Results Summary:")
    print(f"   Total samples: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")

    if show_preview and successful > 0:
        print(f"\n📋 Sample Results:")
        preview_count = 0
        for i, result in enumerate(results):
            if "error" not in result and result.get("pred_result"):
                preview_count += 1
                print(f"\n  Sample {preview_count}:")
                print(f"    Image: {result.get('image', 'unknown')}")

                pred = result.get("pred_result", "")
                if pred:
                    # Truncate long predictions
                    if len(pred) > 100:
                        print(f"    Prediction: {pred[:100]}...")
                    else:
                        print(f"    Prediction: {pred}")
                else:
                    print(f"    No prediction generated")

                if preview_count >= 2:  # Show max 2 samples
                    break

    return total, successful, failed


def count_samples_in_file(file_path: str) -> int:
    """Count samples in a JSONL or JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                return sum(1 for line in f if line.strip())
            else:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate and summarize inference results"
    )
    parser.add_argument("output_file", help="Path to the JSON results file")
    parser.add_argument(
        "--no-preview", action="store_true", help="Skip showing sample results"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count results, no detailed output",
    )

    args = parser.parse_args()

    if not Path(args.output_file).exists():
        print(f"❌ Results file not found: {args.output_file}", file=sys.stderr)
        sys.exit(1)

    total, successful, failed = validate_and_summarize_results(
        args.output_file, show_preview=not args.no_preview and not args.count_only
    )

    if args.count_only:
        print(f"{total}:{successful}:{failed}")

    # Exit with non-zero code if all samples failed
    if total > 0 and successful == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
