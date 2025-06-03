#!/usr/bin/env python3
"""
JSONL Validation Script for Qwen2.5-VL Dataset

This script validates the format and structure of JSONL files generated
by the data conversion pipeline.
"""

import argparse
import json
import sys
from pathlib import Path


def validate_jsonl_format(filename: str) -> bool:
    """Validate that file contains valid JSON lines."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    json.loads(line)
        print(f"   ✅ {filename}: Valid JSONL format")
        return True
    except Exception as e:
        print(f"   ❌ {filename}: Invalid JSON on line {i}: {e}")
        return False


def validate_conversation_structure(
    filename: str, expect_examples: bool = False
) -> bool:
    """Validate conversation structure in JSONL file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue

                data = json.loads(line)

                # Check required fields
                if "conversations" not in data:
                    print(f"   ❌ {filename} line {i}: Missing conversations field")
                    return False

                conversations = data["conversations"]
                if not isinstance(conversations, list):
                    print(f"   ❌ {filename} line {i}: conversations must be a list")
                    return False

                # Check for required roles
                roles = [conv.get("role") for conv in conversations]
                if "system" not in roles or "assistant" not in roles:
                    print(
                        f"   ❌ {filename} line {i}: Missing required roles (system, assistant)"
                    )
                    return False

                # For training data with examples, expect multiple user/assistant pairs
                if expect_examples:
                    user_count = roles.count("user")
                    assistant_count = roles.count("assistant")
                    # Note: Some samples might not have examples, which is okay
                    if user_count < 1 or assistant_count < 1:
                        print(
                            f"   ❌ {filename} line {i}: Missing user/assistant interactions"
                        )
                        return False

                # Check images field
                if "images" in data:
                    images = data["images"]
                    if not isinstance(images, list) or len(images) == 0:
                        print(
                            f"   ❌ {filename} line {i}: images must be a non-empty list"
                        )
                        return False

        print(f"   ✅ {filename}: Valid conversation structure")
        return True

    except Exception as e:
        print(f"   ❌ {filename}: Validation error: {e}")
        return False


def validate_files(
    train_file: str, val_file: str, include_examples: bool = False
) -> bool:
    """Validate both training and validation files."""
    print("   Validating JSON format...")

    # Validate JSON format
    train_format_valid = validate_jsonl_format(train_file)
    val_format_valid = validate_jsonl_format(val_file)

    if not (train_format_valid and val_format_valid):
        return False

    print("   Validating conversation structure...")

    # Validate conversation structure
    train_structure_valid = validate_conversation_structure(
        train_file, include_examples
    )
    val_structure_valid = validate_conversation_structure(val_file, False)

    return train_structure_valid and val_structure_valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL files for Qwen2.5-VL dataset"
    )
    parser.add_argument("--train_file", required=True, help="Training JSONL file")
    parser.add_argument("--val_file", required=True, help="Validation JSONL file")
    parser.add_argument(
        "--include_examples",
        action="store_true",
        help="Expect examples in training data",
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.train_file).exists():
        print(f"❌ Error: Training file {args.train_file} not found")
        sys.exit(1)

    if not Path(args.val_file).exists():
        print(f"❌ Error: Validation file {args.val_file} not found")
        sys.exit(1)

    # Validate files
    if validate_files(args.train_file, args.val_file, args.include_examples):
        print("✅ All validation checks passed")
        sys.exit(0)
    else:
        print("❌ Validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
