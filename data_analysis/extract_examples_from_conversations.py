#!/usr/bin/env python3
"""
Extract Examples from Conversation Format

This script extracts representative samples from conversation-format training data
and formats them for use as few-shot examples in the qwen_converter.py training pipeline.

Uses core modules for consistent processing and field standardization.
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

# Add parent directory to path for core_modules import
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from data_conversion.core_modules import CompactResponseFormatter, ResponseFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def transform_objects_for_response_type(
    objects: List[Dict[str, Any]], response_types: Set[str]
) -> List[Dict[str, Any]]:
    """Transform objects based on selected response types using compact format."""
    transformed_objects = []

    for obj in objects:
        if "description" not in obj:
            continue

        # Parse the original description using core modules
        components = ResponseFormatter.parse_description_string(obj["description"])

        # Convert to compact format using CompactResponseFormatter
        new_description = CompactResponseFormatter.format_to_compact_string(components)

        # Create new object with transformed description
        new_obj = obj.copy()
        new_obj["description"] = new_description
        transformed_objects.append(new_obj)

    return transformed_objects


def parse_assistant_response(response: str) -> List[Dict[str, Any]]:
    """Parse assistant response to extract objects."""
    try:
        objects = json.loads(response)
        return objects if isinstance(objects, list) else []
    except json.JSONDecodeError:
        return []


def analyze_sample_complexity(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complexity of a sample (supports both conversation and raw formats)."""

    # Handle conversation format
    if "conversations" in sample:
        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])

        # Find assistant response
        assistant_response = None
        for conv in conversations:
            if conv.get("role") == "assistant":
                assistant_response = conv.get("content", "")
                break

        if not assistant_response:
            return {
                "image": image_path,
                "num_objects": 0,
                "complexity_score": 0,
                "object_types": [],
                "questions": [],
                "objects": [],
            }

        # Parse objects from assistant response
        objects = parse_assistant_response(assistant_response)

    # Handle raw format
    elif "objects" in sample:
        image_path = sample.get("images", [""])[0]
        objects_data = sample.get("objects", {})
        ref_items = objects_data.get("ref", [])
        bbox_items = objects_data.get("bbox", [])

        # Convert raw format to objects format
        objects = []
        for i, ref in enumerate(ref_items):
            if i < len(bbox_items):
                bbox = bbox_items[i]

                # Parse ref string or dict and convert to compact format
                if isinstance(ref, str):
                    if ";" in ref:
                        # Parse verbose format and convert to compact
                        components = ResponseFormatter.parse_description_string(ref)
                        description = CompactResponseFormatter.format_to_compact_string(
                            components
                        )
                    else:
                        # It's a direct Chinese description
                        description = ref
                elif isinstance(ref, dict):
                    # Convert dict to compact format using standardized field names only
                    object_type = ref.get("object_type", "")
                    property_value = ref.get("property", "")
                    extra_info = ref.get("extra_info", "")

                    # Create components dict and convert to compact format
                    components = {
                        "object_type": object_type,
                        "property": property_value,
                        "extra_info": extra_info,
                    }
                    description = CompactResponseFormatter.format_to_compact_string(
                        components
                    )
                else:
                    description = str(ref)

                objects.append({"bbox": bbox, "description": description})

    else:
        return {
            "image": sample.get("images", [""])[0] if "images" in sample else "",
            "num_objects": 0,
            "complexity_score": 0,
            "object_types": [],
            "questions": [],
            "objects": [],
        }

    # Extract characteristics using CompactResponseFormatter
    object_types = []
    questions = []

    for obj in objects:
        description = obj.get("description", "")
        if ";" in description:
            # For compact format, parse back to components for analysis
            components = CompactResponseFormatter.parse_compact_string(description)

            object_types.append(components.get("object_type", ""))
            property_value = components.get("property", "")
            if property_value and property_value != "none":
                questions.append(property_value)
        else:
            # For Chinese descriptions, the description is the object type
            object_types.append(description)

    # Calculate complexity score
    num_objects = len(objects)
    unique_types = len(set(object_types))
    unique_questions = len(set(questions))

    complexity_score = num_objects * 2.5 + unique_types * 1.5 + unique_questions * 1.0

    return {
        "image": image_path,
        "num_objects": num_objects,
        "complexity_score": complexity_score,
        "object_types": object_types,
        "questions": questions,
        "objects": objects,
        "unique_types": unique_types,
        "unique_questions": unique_questions,
    }


def categorize_samples(
    samples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize samples by complexity and characteristics."""
    analyzed_samples = []

    for sample in samples:
        analysis = analyze_sample_complexity(sample)
        analyzed_samples.append({**sample, "_analysis": analysis})

    # Sort by complexity
    analyzed_samples.sort(key=lambda x: x["_analysis"]["complexity_score"])

    categories = {"sparse": [], "medium": [], "dense": [], "diverse": [], "rare": []}

    # Categorize by object count
    for sample in analyzed_samples:
        analysis = sample["_analysis"]
        num_objects = analysis["num_objects"]

        if num_objects <= 3:
            categories["sparse"].append(sample)
        elif 4 <= num_objects <= 10:
            categories["medium"].append(sample)
        elif num_objects > 10:
            categories["dense"].append(sample)

    # Find diverse samples (high unique type count)
    diverse_samples = sorted(
        analyzed_samples, key=lambda x: x["_analysis"]["unique_types"], reverse=True
    )
    categories["diverse"] = diverse_samples[:10]  # Top 10 most diverse

    # Find rare samples (uncommon object types)
    all_object_types = []
    for sample in analyzed_samples:
        all_object_types.extend(sample["_analysis"]["object_types"])

    type_counts = Counter(all_object_types)
    rare_threshold = max(1, len(analyzed_samples) * 0.05)  # 5% threshold

    for sample in analyzed_samples:
        analysis = sample["_analysis"]
        has_rare = any(
            type_counts[obj_type] <= rare_threshold
            for obj_type in set(analysis["object_types"])
        )
        if has_rare:
            categories["rare"].append(sample)

    return categories


def select_best_examples(
    categories: Dict[str, List[Dict[str, Any]]],
    num_examples: int = 5,
    response_types: Set[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Select the best examples from each category."""
    selected = {}

    # Priority order
    priority_categories = ["sparse", "medium", "dense", "diverse", "rare"]

    for category in priority_categories:
        if (
            category in categories
            and categories[category]
            and len(selected) < num_examples
        ):
            # Sort by complexity score and select the best one
            category_samples = sorted(
                categories[category], key=lambda x: x["_analysis"]["complexity_score"]
            )

            # For sparse, take the simplest; for others, take middle complexity
            if category == "sparse":
                best_sample = category_samples[0]
            else:
                mid_idx = len(category_samples) // 2
                best_sample = (
                    category_samples[mid_idx]
                    if category_samples
                    else category_samples[0]
                )

            # Convert to example format
            analysis = best_sample["_analysis"]

            # Use objects directly (already in compact format from analysis)
            objects = analysis["objects"]

            example = {"image": analysis["image"], "objects": objects}

            selected[category] = example
            logger.info(
                f"Selected {category} example: {analysis['image']} ({analysis['num_objects']} objects)"
            )

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract examples from conversation format training data"
    )
    parser.add_argument(
        "data_path", help="Path to the conversation-format JSONL dataset file"
    )
    parser.add_argument(
        "--output",
        default="data_analysis/training_examples.json",
        help="Output JSON file for training examples (default: data_analysis/training_examples.json)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to extract (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--response_types",
        nargs="+",
        choices=["object_type", "property", "extra_info"],
        default=["object_type", "property", "extra_info"],
        help="Response types to include in output (default: all types)",
    )

    args = parser.parse_args()

    # Use response types directly
    response_types = set(args.response_types)
    logger.info(f"Using response types: {sorted(response_types)}")

    # Set random seed
    random.seed(args.seed)

    # Load data
    logger.info(f"Loading conversation data from {args.data_path}")
    samples = []

    with open(args.data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    logger.info(f"Loaded {len(samples)} samples")

    # Categorize samples
    logger.info("Categorizing samples by complexity...")
    categories = categorize_samples(samples)

    for category, category_samples in categories.items():
        logger.info(f"{category.upper()}: {len(category_samples)} samples")

    # Select best examples
    logger.info(f"Selecting {args.num_examples} best examples...")
    selected_examples = select_best_examples(
        categories, args.num_examples, response_types
    )

    # Save examples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(selected_examples)} examples to {output_path}")

    # Print summary
    print("\n=== EXTRACTED EXAMPLES SUMMARY ===")
    print(f"Response types: {sorted(response_types)}")

    print()

    for category, example in selected_examples.items():
        print(f"{category.upper()}: {example['image']}")
        print(f"  Objects: {len(example['objects'])}")
        if example["objects"]:
            sample_descriptions = [
                obj.get("description", "") for obj in example["objects"][:3]
            ]
            print(f"  Sample objects: {sample_descriptions}")
            if len(example["objects"]) > 3:
                print(f"  ... and {len(example['objects']) - 3} more")
        print()


if __name__ == "__main__":
    main()
