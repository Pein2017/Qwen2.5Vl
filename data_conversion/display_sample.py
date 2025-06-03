#!/usr/bin/env python3
"""
Sample Display Script for Qwen2.5-VL Dataset

This script displays sample data format and statistics from JSONL files.
"""

import argparse
import json
from pathlib import Path


def display_file_statistics(train_file: str, val_file: str, examples_file: str = None):
    """Display file statistics."""
    print("📊 Dataset Statistics:")
    print("=====================")

    if Path(train_file).exists():
        with open(train_file, "r", encoding="utf-8") as f:
            train_count = sum(1 for line in f if line.strip())
        print(f"📊 Training samples: {train_count}")

    if Path(val_file).exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_count = sum(1 for line in f if line.strip())
        print(f"📊 Validation samples: {val_count}")

    if examples_file and Path(examples_file).exists():
        try:
            with open(examples_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    example_count = len(data)
                elif isinstance(data, dict):
                    example_count = len(
                        [v for v in data.values() if isinstance(v, dict)]
                    )
                else:
                    example_count = 0
            print(f"📊 Few-shot examples: {example_count}")
        except Exception as e:
            print(f"⚠️  Could not read examples file: {e}")


def display_sample_format(train_file: str):
    """Display sample training data format."""
    print("\n📋 Sample Training Data Format:")
    print("================================")

    try:
        with open(train_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                print("❌ Training file is empty")
                return

            sample = json.loads(first_line)

            conversations = sample.get("conversations", [])
            images = sample.get("images", [])

            print(f"Conversations: {len(conversations)}")
            print(f"Images: {len(images)}")

            # Show conversation structure
            for i, conv in enumerate(conversations[:3]):  # Show first 3 conversations
                role = conv.get("role", "unknown")
                content = conv.get("content", "")
                content_preview = (
                    content[:100] + "..." if len(content) > 100 else content
                )
                print(f"  {i + 1}. {role}: {content_preview}")

            if len(conversations) > 3:
                print(f"  ... and {len(conversations) - 3} more conversation turns")

    except Exception as e:
        print(f"❌ Error reading sample: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Display sample data format and statistics"
    )
    parser.add_argument("--train_file", required=True, help="Training JSONL file")
    parser.add_argument("--val_file", required=True, help="Validation JSONL file")
    parser.add_argument("--examples_file", help="Examples JSON file")
    parser.add_argument("--show_sample", action="store_true", help="Show sample format")
    parser.add_argument("--show_stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    if args.show_stats:
        display_file_statistics(args.train_file, args.val_file, args.examples_file)

    if args.show_sample:
        display_sample_format(args.train_file)


if __name__ == "__main__":
    main()
