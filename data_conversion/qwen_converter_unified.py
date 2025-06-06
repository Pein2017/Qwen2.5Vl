#!/usr/bin/env python3
"""
Clean Semantic Data Converter for Qwen2.5-VL

Converts intermediate JSONL (with verbose format) to clean semantic data format.
Uses compact descriptions without schema wrappers and "none" values.

Input format (intermediate JSONL):
{"images": ["path.jpg"], "objects": {"ref": ["object_type:X;property:Y;extra_info:Z"], "bbox": [[x1,y1,x2,y2]]}}

Output format (clean semantic):
{"images": ["path.jpg"], "objects": [{"box": [x1,y1,x2,y2], "desc": "type, property, extra_info"}]}

Extended format (multi-round with examples):
{
  "examples": [
    {"images": ["example1.jpg"], "objects": [{"box": [x1,y1,x2,y2], "desc": "description"}]}
  ],
  "target": {"images": ["target.jpg"], "objects": [{"box": [x1,y1,x2,y2], "desc": "description"}]}
}
"""

import argparse
import json
import logging
import random

# Import core modules for processing
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(str(Path(__file__).parent))
from core_modules import CompactResponseFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CleanSemanticConverter:
    """Converts intermediate JSONL to clean semantic data format."""

    def __init__(self, response_types: Set[str] = None):
        self.response_types = response_types or {
            "object_type",
            "property",
            "extra_info",
        }
        self.examples = {}

    def _load_examples(self, examples_file: str) -> Dict:
        """Load examples from file."""
        if not examples_file or not Path(examples_file).exists():
            logger.warning(f"Examples file not found: {examples_file}")
            return {}

        try:
            with open(examples_file, "r", encoding="utf-8") as f:
                examples_data = json.load(f)

            # Handle both old format and new category-based format
            if isinstance(examples_data, dict) and any(
                key in examples_data for key in ["sparse", "medium", "dense"]
            ):
                # New category-based format
                logger.info("Loading examples from category-based format")
                return examples_data
            else:
                # Old format - convert to new format
                logger.info(
                    "Converting examples from old format to category-based format"
                )
                return {
                    "sparse": examples_data,
                    "medium": examples_data,
                    "dense": examples_data,
                }

        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            return {}

    def _convert_verbose_to_compact(self, verbose_desc: str) -> str:
        """Convert verbose description to compact format."""
        return CompactResponseFormatter.convert_from_verbose_format(
            verbose_desc, self.response_types
        )

    def _convert_sample(self, sample: Dict) -> Dict:
        """Convert a single sample from intermediate to clean semantic format."""
        try:
            # Extract basic info
            images = sample.get("images", [])
            objects_data = sample.get("objects", {})

            # Handle the intermediate format structure
            if isinstance(objects_data, dict):
                ref_list = objects_data.get("ref", [])
                bbox_list = objects_data.get("bbox", [])
            else:
                logger.warning("Unexpected objects format, skipping sample")
                return None

            if len(ref_list) != len(bbox_list):
                logger.warning(
                    f"Mismatch between ref ({len(ref_list)}) and bbox ({len(bbox_list)}) counts"
                )
                return None

            # Convert to clean semantic format
            clean_objects = []
            for ref_desc, bbox in zip(ref_list, bbox_list):
                # Convert verbose description to compact format
                compact_desc = self._convert_verbose_to_compact(ref_desc)

                clean_objects.append({"box": bbox, "desc": compact_desc})

            return {"images": images, "objects": clean_objects}

        except Exception as e:
            logger.warning(f"Failed to convert sample: {e}")
            return None

    def _get_example_by_category(self, num_objects: int) -> Dict:
        """Get an appropriate example based on object density."""
        if not self.examples:
            return None

        # Categorize by object count
        if num_objects <= 2:
            category = "sparse"
        elif num_objects <= 5:
            category = "medium"
        else:
            category = "dense"

        # Get example from appropriate category
        category_examples = self.examples.get(category, {})
        if not category_examples:
            # Fallback to any available category
            for cat in ["sparse", "medium", "dense"]:
                if self.examples.get(cat):
                    category_examples = self.examples[cat]
                    break

        if not category_examples:
            return None

        # Convert example to clean semantic format
        example_images = [category_examples.get("image", "")]
        example_objects = []

        for obj in category_examples.get("objects", []):
            if "bbox" in obj and "description" in obj:
                example_objects.append({"box": obj["bbox"], "desc": obj["description"]})

        return {"images": example_images, "objects": example_objects}

    def _create_multi_round_sample(
        self, target_sample: Dict, max_examples: int = 1
    ) -> Dict:
        """Create multi-round sample with examples and target."""
        # If max_examples is 0, return simple format
        if max_examples <= 0:
            return target_sample

        num_objects = len(target_sample.get("objects", []))

        # Get examples
        examples = []
        for _ in range(max_examples):
            example = self._get_example_by_category(num_objects)
            if example:
                examples.append(example)

        # Create multi-round format
        if examples:
            return {"examples": examples, "target": target_sample}
        else:
            # Fallback to simple format if no examples available
            return target_sample

    def convert_and_split(
        self,
        input_jsonl: str,
        output_train: str,
        output_val: str,
        val_ratio: float = 0.1,
        seed: int = 42,
        multi_round: bool = False,
        include_examples: bool = False,
        examples_file: str = None,
        max_examples: int = 1,
    ):
        """Convert intermediate JSONL to clean semantic format and split into train/val."""

        # Load examples if needed
        if include_examples and examples_file:
            self.examples = self._load_examples(examples_file)
            if self.examples:
                logger.info(
                    f"✅ Successfully loaded examples for clean semantic conversion"
                )

        # Load and convert samples
        samples = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    converted = self._convert_sample(sample)
                    if converted:
                        # Apply multi-round format if requested
                        if multi_round and include_examples and max_examples > 0:
                            converted = self._create_multi_round_sample(
                                converted, max_examples
                            )
                        samples.append(converted)
                except Exception as e:
                    logger.warning(f"Failed to process line {line_num}: {e}")

        logger.info(f"Successfully converted {len(samples)} samples")

        if len(samples) == 0:
            raise ValueError("No valid samples found after conversion")

        # Split into train/val
        random.seed(seed)
        random.shuffle(samples)

        val_size = int(len(samples) * val_ratio)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]

        # Write output files
        self._write_jsonl(train_samples, output_train)
        self._write_jsonl(val_samples, output_val)

        logger.info(
            f"Split complete: {len(train_samples)} train, {len(val_samples)} val samples"
        )

        # Log format information
        if multi_round and include_examples and max_examples > 0:
            logger.info("✅ Generated multi-round format with examples and target")
        else:
            logger.info("✅ Generated simple format (no examples)")

    def _write_jsonl(self, samples: List[Dict], output_file: str):
        """Write samples to JSONL file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert intermediate JSONL to clean semantic data"
    )
    parser.add_argument(
        "--input_jsonl", required=True, help="Input intermediate JSONL file"
    )
    parser.add_argument(
        "--output_train", required=True, help="Output training JSONL file"
    )
    parser.add_argument(
        "--output_val", required=True, help="Output validation JSONL file"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--multi_round", action="store_true", help="Enable multi-round conversations"
    )
    parser.add_argument(
        "--include_examples", action="store_true", help="Include few-shot examples"
    )
    parser.add_argument("--examples_file", help="Path to examples JSON file")
    parser.add_argument(
        "--max_examples", type=int, default=1, help="Maximum examples per sample"
    )
    parser.add_argument(
        "--response_types",
        default="object_type property extra_info",
        help="Space-separated response types to include",
    )

    args = parser.parse_args()

    # Parse response types
    response_types = set(args.response_types.split())

    # Create converter
    converter = CleanSemanticConverter(response_types=response_types)

    # Convert and split
    converter.convert_and_split(
        input_jsonl=args.input_jsonl,
        output_train=args.output_train,
        output_val=args.output_val,
        val_ratio=args.val_ratio,
        seed=args.seed,
        multi_round=args.multi_round,
        include_examples=args.include_examples,
        examples_file=args.examples_file,
        max_examples=args.max_examples,
    )

    logger.info("✅ Conversion completed successfully!")


if __name__ == "__main__":
    main()
