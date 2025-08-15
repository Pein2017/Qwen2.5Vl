#!/usr/bin/env python3
"""
Utility script to find the latest checkpoint in the output directory.

This helps users identify the final inference-ready checkpoint after training.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_checkpoints(output_dir: Path) -> List[Tuple[int, Path]]:
    """Find all checkpoint directories and their step numbers.

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step number
    """
    checkpoints = []

    if not output_dir.exists():
        return checkpoints

    # Pattern to match checkpoint-xxxx directories
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    for item in output_dir.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step_number = int(match.group(1))
                checkpoints.append((step_number, item))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def validate_checkpoint(checkpoint_path: Path) -> dict:
    """Validate that a checkpoint contains essential files.

    Returns:
        Dictionary with validation results
    """
    essential_files = [
        "config.json",
        "coordinate_config.json",
        "coord_token_ids.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]

    results = {
        "valid": True,
        "missing_files": [],
        "has_model_weights": False,
        "has_metrics": False,
        "file_count": 0,
    }

    # Check essential files
    for filename in essential_files:
        if not (checkpoint_path / filename).exists():
            results["missing_files"].append(filename)
            results["valid"] = False

    # Check for model weights
    safetensors_files = list(checkpoint_path.glob("model*.safetensors*"))
    if safetensors_files:
        results["has_model_weights"] = True
    else:
        results["valid"] = False

    # Check for metrics
    if (checkpoint_path / "metrics-final.json").exists():
        results["has_metrics"] = True

    # Count total files
    results["file_count"] = len(list(checkpoint_path.iterdir()))

    return results


def main():
    """Main function to find and display checkpoint information."""

    if len(sys.argv) != 2:
        print("Usage: python find_latest_checkpoint.py <output_directory>")
        print("\nExample:")
        print("  python find_latest_checkpoint.py src_coord_pretrain/output")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        sys.exit(1)

    print(f"🔍 Searching for checkpoints in: {output_dir}")
    print("=" * 60)

    # Find all checkpoints
    checkpoints = find_checkpoints(output_dir)

    if not checkpoints:
        print("❌ No checkpoints found!")
        print("\nMake sure you're pointing to the correct output directory.")
        print("Expected structure: output_dir/checkpoint-xxxx/")
        sys.exit(1)

    print(f"📁 Found {len(checkpoints)} checkpoint(s):")
    print()

    # Display all checkpoints
    for i, (step, checkpoint_path) in enumerate(checkpoints):
        is_latest = i == len(checkpoints) - 1
        status_icon = "🎯" if is_latest else "📋"

        print(f"{status_icon} checkpoint-{step}")
        print(f"   Path: {checkpoint_path}")

        # Validate checkpoint
        validation = validate_checkpoint(checkpoint_path)

        if validation["valid"]:
            print(f"   Status: ✅ Valid ({validation['file_count']} files)")
            if validation["has_metrics"]:
                print(f"   Type: 🎉 Final inference-ready checkpoint")
            else:
                print(f"   Type: 🔄 Regular training checkpoint")
        else:
            print(f"   Status: ❌ Invalid")
            if validation["missing_files"]:
                print(f"   Missing: {', '.join(validation['missing_files'])}")
            if not validation["has_model_weights"]:
                print(f"   Missing: Model weights (*.safetensors)")

        print()

    # Highlight the latest checkpoint
    latest_step, latest_path = checkpoints[-1]
    latest_validation = validate_checkpoint(latest_path)

    print("🎯 LATEST CHECKPOINT:")
    print(f"   Step: {latest_step}")
    print(f"   Path: {latest_path}")

    if latest_validation["valid"]:
        print(f"   Status: ✅ Ready for use")

        if latest_validation["has_metrics"]:
            print(f"   Type: 🎉 Final inference-ready checkpoint")
            print()
            print("📋 Usage Instructions:")
            print(
                f"   1. Validate: python scripts/validate_integration.py {latest_path}"
            )
            print(
                f'   2. Use in src_new config: model_path: "{latest_path.absolute()}"'
            )
            print(f"   3. Load in Python:")
            print(f"      from transformers import Qwen2_5_VLForConditionalGeneration")
            print(
                f"      model = Qwen2_5_VLForConditionalGeneration.from_pretrained('{latest_path}')"
            )
        else:
            print(f"   Type: 🔄 Regular training checkpoint")
            print(f"   Note: This may not have all inference-ready features")
    else:
        print(f"   Status: ❌ Invalid - cannot be used")
        if latest_validation["missing_files"]:
            print(f"   Missing files: {', '.join(latest_validation['missing_files'])}")

    print()
    print("💡 Tips:")
    print("   - Always use the full checkpoint path (including checkpoint-xxxx/)")
    print("   - Final checkpoints have metrics-final.json and README_INFERENCE.json")
    print("   - Run validate_integration.py before using in src_new")


if __name__ == "__main__":
    main()
