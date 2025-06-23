#!/usr/bin/env python3
"""
Clean Semantic Data Converter for Qwen2.5-VL

Converts intermediate JSONL (with verbose format) to clean semantic data format.
Uses compact descriptions without schema wrappers and "none" values.

Input format (intermediate JSONL):
- English: {"images": ["path.jpg"], "objects": {"ref": ["object_type:X;property:Y;extra_info:Z"], "bbox": [[x1,y1,x2,y2]]}}
- Chinese: {"images": ["path.jpg"], "objects": {"ref": ["类型/属性/额外信息"], "bbox": [[x1,y1,x2,y2]]}}

Output format (clean semantic):
{"images": ["path.jpg"], "objects": [{"box": [x1,y1,x2,y2], "desc": "type/property/extra_info"}]}

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
from core_modules import ResponseFormatter

# Configure logging to file
LOG_FILE = Path(__file__).parent / "convert.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=str(LOG_FILE),
    filemode="a",
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
        # Load label hierarchy mapping for filtering properties
        mapping_path = Path(__file__).parent / "label_hierarchy.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping_raw = json.load(f)
            # Normalize to dict mapping object_type -> list of allowed properties
            if isinstance(mapping_raw, list):
                self.label_hierarchy = {
                    entry["object_type"]: entry.get("property", [])
                    for entry in mapping_raw
                }
            elif isinstance(mapping_raw, dict):
                # Old dict-of-lists format
                if all(isinstance(v, list) for v in mapping_raw.values()):
                    self.label_hierarchy = mapping_raw
                else:
                    # dict-of-dicts with 'property' key
                    self.label_hierarchy = {
                        k: v.get("property", []) for k, v in mapping_raw.items()
                    }
            else:
                self.label_hierarchy = {}
        else:
            self.label_hierarchy = {}

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

    def _filter_slash_description(self, description: str) -> str:
        """Filters a slash-separated description based on response_types and label hierarchy mapping."""
        # Unify separator: convert commas to slashes
        description = description.replace(", ", "/").replace(",", "/")
        parts = [p.strip() for p in description.split("/") if p.strip()]
        final_parts = []
        obj = parts[0] if len(parts) > 0 else ""
        # object_type
        if "object_type" in self.response_types and obj:
            final_parts.append(obj)
        # property: include only if allowed in hierarchy
        if "property" in self.response_types and len(parts) > 1:
            prop = parts[1]
            allowed_props = self.label_hierarchy.get(obj, [])
            if prop in allowed_props:
                final_parts.append(prop)
        # extra_info: combine all segments beyond property
        if "extra_info" in self.response_types and len(parts) > 2:
            extra_info_segments = parts[2:]
            final_parts.append("/".join(extra_info_segments))
        return "/".join(final_parts)

    def _convert_verbose_to_compact(self, verbose_desc: str) -> str:
        """Convert verbose (semicolon-separated) description to compact (slash-separated) format."""
        if not verbose_desc:
            return "unknown"
        components = ResponseFormatter.parse_description_string(verbose_desc)
        final_parts = []
        if (
            "object_type" in self.response_types
            and components.get("object_type")
            and components["object_type"] != "none"
        ):
            final_parts.append(components["object_type"])
        if (
            "property" in self.response_types
            and components.get("property")
            and components["property"] != "none"
        ):
            final_parts.append(components["property"])
        if (
            "extra_info" in self.response_types
            and components.get("extra_info")
            and components["extra_info"] != "none"
        ):
            final_parts.append(components["extra_info"])
        return "/".join(final_parts)

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
                # If 'ref_desc' contains ';', it's a verbose English description
                if ";" in ref_desc:
                    compact_desc = self._convert_verbose_to_compact(ref_desc)
                else:
                    # Otherwise, it's a direct Chinese description
                    compact_desc = self._filter_slash_description(ref_desc)

                # Fail-fast if description is empty
                if not compact_desc:
                    raise ValueError(
                        f"Empty description for ref_desc '{ref_desc}' in sample: {sample}"
                    )
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
                desc = obj["description"]
                # Handle both verbose and compact/Chinese descriptions in examples
                if ";" in desc:
                    compact_desc = self._convert_verbose_to_compact(desc)
                else:
                    compact_desc = self._filter_slash_description(desc)
                # Fail-fast if example description is empty
                if not compact_desc:
                    raise ValueError(
                        f"Empty example description for desc '{desc}' in examples file"
                    )
                example_objects.append({"box": obj["bbox"], "desc": compact_desc})

        return {"images": example_images, "objects": example_objects}

    def _create_multi_round_sample(
        self, target_sample: Dict, max_examples: int = 1
    ) -> Dict:
        """Create multi-round sample with examples from diverse categories."""
        if max_examples <= 0 or not self.examples:
            return target_sample

        # Flatten all loaded examples into a unique pool
        all_items = []
        for val in self.examples.values():
            if isinstance(val, list):
                all_items.extend(val)
            else:
                all_items.append(val)
        # Deduplicate by JSON content
        unique_items = []
        seen = set()
        for item in all_items:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        # Sample up to max_examples without replacement
        num_to_sample = min(max_examples, len(unique_items))
        sampled = random.sample(unique_items, num_to_sample)

        examples = []
        for ex_data in sampled:
            # Build example entry; support both 'images' and 'image' keys
            if "images" in ex_data:
                ex_images = ex_data["images"]
            elif "image" in ex_data:
                ex_images = [ex_data["image"]]
            else:
                ex_images = []
            example_objects = []
            for obj in ex_data.get("objects", []):
                desc = obj.get("description", obj.get("desc", ""))
                if ";" in desc:
                    compact_desc = self._convert_verbose_to_compact(desc)
                else:
                    compact_desc = self._filter_slash_description(desc)
                if not compact_desc:
                    raise ValueError(f"Empty example description for desc '{desc}'")
                example_objects.append(
                    {
                        "box": obj.get("bbox", obj.get("box", [])),
                        "desc": compact_desc,
                    }
                )
            examples.append({"images": ex_images, "objects": example_objects})

        return {"examples": examples, "target": target_sample}

    def convert_and_split(
        self,
        input_jsonl: str,
        output_train: str,
        output_val: str,
        val_ratio: float = 0.1,
        seed: int = 42,
        teacher_type: str = "random",  # "random" or "predefined"
        examples_file: str = None,
        max_examples: int = 1,
    ):
        """Convert intermediate JSONL to clean semantic format and split into train/val."""

        # ------------------------------------------------------------------
        # Determine if we should attach pre-selected teacher examples and
        # **fail fast** if the required examples file is missing.
        # ------------------------------------------------------------------
        attach_examples = teacher_type == "predefined"

        if attach_examples:
            if not examples_file:
                raise ValueError(
                    "teacher_type 'predefined' requires --examples_file to be set"
                )
            from pathlib import Path

            if not Path(examples_file).exists():
                raise FileNotFoundError(
                    f"Examples file not found: {examples_file}. Cannot proceed with predefined teachers."
                )

        # Load examples if needed
        if attach_examples and examples_file:
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
                        # Attach pre-selected examples if requested
                        if attach_examples and max_examples > 0:
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

        # Strip teacher examples from validation set for zero-shot
        stripped_val_samples = []
        for s in val_samples:
            if isinstance(s, dict) and "target" in s:
                stripped_val_samples.append(s["target"])
            else:
                stripped_val_samples.append(s)
        val_samples = stripped_val_samples

        # Write output files
        self._write_jsonl(train_samples, output_train)
        self._write_jsonl(val_samples, output_val)

        logger.info(
            f"Split complete: {len(train_samples)} train, {len(val_samples)} val samples"
        )

        # Log format information
        if attach_examples and max_examples > 0:
            logger.info("✅ Generated data with pre-selected teacher examples")
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
        "--teacher_type",
        choices=["random", "predefined"],
        default="random",
        help="Choose teacher sampling strategy: 'random' (dataset picks) or 'predefined' (converter attaches)",
    )
    parser.add_argument("--examples_file", help="Path to examples JSON file")
    parser.add_argument(
        "--max_examples", type=int, default=1, help="Maximum examples per sample"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--response_types",
        default="object_type property extra_info",
        help="Space-separated response types to include",
    )

    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Parse response types
    response_types = set(args.response_types.split())

    # Create converter
    converter = CleanSemanticConverter(response_types=response_types)

    # Convert and split
    converter.convert_and_split(
        args.input_jsonl,
        args.output_train,
        args.output_val,
        val_ratio=args.val_ratio,
        seed=args.seed,
        teacher_type=args.teacher_type,
        examples_file=args.examples_file,
        max_examples=args.max_examples,
    )

    logger.info("✅ Conversion completed successfully!")


if __name__ == "__main__":
    main()
