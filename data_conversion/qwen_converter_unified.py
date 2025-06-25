#!/usr/bin/env python3
"""
Clean Semantic Data Converter for Qwen2.5-VL

Converts intermediate JSONL to clean semantic data format for training.
Uses compact slash-separated descriptions without schema wrappers.

Input format (intermediate JSONL):
- English: {"images": ["path.jpg"], "objects": {"ref": ["object_type:X;property:Y;extra_info:Z"], "bbox": [[x1,y1,x2,y2]]}}
- Chinese: {"images": ["path.jpg"], "objects": {"ref": ["类型/属性/额外信息"], "bbox": [[x1,y1,x2,y2]]}}

Output format (clean semantic):
{"images": ["path.jpg"], "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": "type/property/extra_info"}]}

Note: Teacher pool samples are handled separately for multi-chat conversations during training.
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
                clean_objects.append({"bbox_2d": bbox, "desc": compact_desc})

            return {"images": images, "objects": clean_objects}

        except Exception as e:
            logger.warning(f"Failed to convert sample: {e}")
            return None

    def convert_and_split(
        self,
        input_jsonl: str,
        output_train: str,
        output_val: str,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Convert intermediate JSONL to clean semantic format and split into train/val."""
        samples = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw = json.loads(line.strip())
                    converted = self._convert_sample(raw)
                    if converted:
                        samples.append(converted)
                except Exception as e:
                    logger.warning(f"Failed to process line {line_num}: {e}")
        if not samples:
            raise ValueError("No valid samples found after conversion")
        random.seed(seed)
        random.shuffle(samples)
        val_size = int(len(samples) * val_ratio)
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]
        self._write_jsonl(train_samples, output_train)
        self._write_jsonl(val_samples, output_val)
        logger.info(
            f"Split complete: {len(train_samples)} train, {len(val_samples)} val samples"
        )

    def _write_jsonl(self, samples: List[Dict], output_file: str):
        """Write samples to JSONL file."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert intermediate JSONL to clean semantic train/val splits"
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
    )

    logger.info("✅ Conversion completed successfully!")


if __name__ == "__main__":
    main()
