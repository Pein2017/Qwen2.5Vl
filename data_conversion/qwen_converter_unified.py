#!/usr/bin/env python3
"""
Qwen2.5-VL Dataset Converter

This script provides conversion for Qwen2.5-VL with two modes:
- Standard: Enhanced prompts for optimal performance
- Multi-Image: Few-shot learning with multiple images

Uses core modules for consistent processing and field standardization.
"""

import argparse
import json
import logging
import random
from typing import Any, Dict, List, Set

from core_modules import (
    CompactResponseFormatter,
    FieldStandardizer,
    ObjectProcessor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ――― System Prompts (Improved Format) ―――

# ――― Categorized Candidate Phrases Template ―――
CANDIDATE_PHRASES_TEMPLATE = """\
AVAILABLE PHRASES FOR REFERENCE (choose only those matching visible objects):

1. BBU Types:
   - huawei bbu
   - zte bbu
   - ericsson bbu

2. Shield/Baffle Equipment:
   - bbu shield installed
   - bbu shield not installed
   - shield orientation correct
   - shield unobstructed
   - shield obstructed
   - shield brand mismatch
   - shield installed in wrong position
   - shield screws not fully installed

3. Cabinet Status:
   - cabinet fully occupied
   - cabinet not fully occupied
   - cabinet grounding correct
   - cabinet grounding incorrect

4. Screw/Installation:
   - install screw correct
   - install screw incorrect
   - floor screw installed
   - not tightened
   - installation position incorrect

5. Cable/Connection:
   - fiber cable
   - non-fiber cable
   - fibre bend radius proper
   - fibre bend radius improper
   - snake tube protection
   - armour protection
   - no snake tube or armour protection
   - fibre is protected by both armour and snake tube
   - cpri connection correct
   - cpri connection incorrect
   - odf connection correct
   - binding aligned horizontally and vertically
   - binding not aligned horizontally and vertically
   - only part of the fibre is visible
   - copper exposed

6. Label/Marking:
   - label matches
   - label does not match
   - match
   - not match

7. Other Abnormal:
   - rust
   - bbu not inserted
   - foreign object above bbu
   - unable to assess bend radius
   - which is usually unnecessary
   - other case

Select phrases that apply to objects visible in the image. You may use multiple phrases per object (e.g., 'huawei bbu, shield orientation correct, shield unobstructed').
"""

# ――― Base System Prompts (Improved Structure) ―――
BASE_COMPACT_SYSTEM_PROMPT = """\
You are Q-Vision-QC, an expert assistant specialized in telecom-equipment inspection.
Your task: produce exactly one JSON array of detected objects for each input image.

OUTPUT FORMAT:
- A JSON array where each element has:
    bbox: [x1, y1, x2, y2],
    desc: 'comma-separated object_type and details'
- Sort by top-to-bottom (increasing y), then left-to-right (increasing x).
- Use unquoted keys: bbox and desc.
- Wrap string in single quotes. No whitespace or comments outside the JSON.
- Always respond in English only.
- Output only the JSON array (no extra text or explanations).
"""

BASE_MULTI_IMAGE_SYSTEM_PROMPT = """\
You are Q-Vision-QC, an expert assistant specialized in telecom-equipment inspection.
Your task: produce exactly one JSON array of detected objects for each input image.

OUTPUT FORMAT:
- A JSON array where each element has:
    bbox: [x1, y1, x2, y2],
    desc: 'comma-separated object_type and details'
- Sort by top-to-bottom (increasing y), then left-to-right (increasing x).
- Use unquoted keys: bbox and desc.
- Wrap string in single quotes. No whitespace or comments outside the JSON.
- Always respond in English only.
- Output only the JSON array (no extra text or explanations).

MULTI-ROUND INSTRUCTIONS:
1) You will see K example rounds. Each round has:
   - A user turn with `<image>`
   - An assistant turn with the correct JSON array.

2) Then you will see one final user turn with `<image>`. Your job is to reply with the JSON array for that image.
"""

# ――― Dense Captioning Equipment List (Fallback for non-candidate mode) ―――
DENSE_CAPTIONING_EQUIPMENT_LIST = """\
Detect only these types (omit if absent):
– huawei bbu, zte bbu, ericsson bbu
– cabinet fully occupied, cabinet not fully occupied
– install screw correct, install screw incorrect, floor screw installed
– cpri connection correct, cpri connection incorrect, odf connection correct
– fiber cable, non-fiber cable
– bbu shield installed, bbu shield not installed
– label matches, label does not match
– cabinet grounding correct, cabinet grounding incorrect
Use these quality details if visible (append after type):
– fibre bend radius proper; snake tube protection; armour protection
– shield orientation correct; shield unobstructed
– binding aligned horizontally and vertically
– rust; not tightened; loose; damaged
– match; not match; blurred; missing
"""

# Build complete system prompts
COMPACT_SYSTEM_PROMPT = (
    f"{BASE_COMPACT_SYSTEM_PROMPT}\n\n{DENSE_CAPTIONING_EQUIPMENT_LIST}"
)

MULTI_IMAGE_SYSTEM_PROMPT = (
    f"{BASE_MULTI_IMAGE_SYSTEM_PROMPT}\n\n{DENSE_CAPTIONING_EQUIPMENT_LIST}"
)

# ――― User Prompts (Simplified) ―――

# Standard single-round prompt
STANDARD_USER_PROMPT = "<image>"

# Few-shot example prompt (cleaner)
MULTI_IMAGE_EXAMPLE_PROMPT = "<image>"

# Multi-round query (simplified)
MULTI_IMAGE_QUERY_PROMPT = "<image>"


def format_compact_json_string(objects: List[Dict[str, Any]]) -> str:
    """
    Format objects as ultra-compact JSON string for natural language descriptions.
    Uses the new natural language format with comma-separated descriptions.
    """
    json_parts = []
    for obj in objects:
        bbox = obj["bbox"]
        description = obj["description"]

        # Description is already simplified by _process_line_to_sample, no need to convert again
        simplified_description = description

        # Ultra-compact: unquoted keys, single quotes for strings
        obj_str = f"{{bbox:[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}],desc:'{simplified_description}'}}"
        json_parts.append(obj_str)

    return f"[{','.join(json_parts)}]"


def write_compact_jsonl_line(sample: Dict[str, Any]) -> str:
    """
    Write ultra-compact JSONL line with minimal escaping for optimal LLM training.
    """
    conversations = sample.get("conversations", [])
    images = sample.get("images", [])

    # Build conversation parts with minimal escaping
    conv_parts = []
    for conv in conversations:
        role = conv["role"]
        content = conv["content"]

        if role == "assistant" and isinstance(content, list):
            # Use compact format for assistant responses
            compact_content = format_compact_json_string(content)
            # Treat compact format as a string - escape for JSON string context
            escaped_content = (
                compact_content.replace("\\", "\\\\")  # Escape backslashes first
                .replace('"', '\\"')  # Escape double quotes
                .replace("\n", " ")  # Replace newlines with spaces
                .replace("\r", " ")  # Replace carriage returns
                .replace("\t", " ")  # Replace tabs with spaces
            )
            conv_str = f'{{"role":"assistant","content":"{escaped_content}"}}'
        else:
            # Minimal escaping for other content
            if isinstance(content, str):
                # Only escape absolutely necessary characters
                clean_content = (
                    content.replace("\\", "\\\\")  # Escape backslashes first
                    .replace('"', '\\"')  # Escape double quotes
                    .replace("\n", " ")  # Replace newlines with spaces
                    .replace("\r", " ")  # Replace carriage returns
                    .replace("\t", " ")
                )  # Replace tabs with spaces
                conv_str = f'{{"role":"{role}","content":"{clean_content}"}}'
            else:
                # Fallback for non-string content
                content_json = json.dumps(
                    content, ensure_ascii=False, separators=(",", ":")
                )
                conv_str = f'{{"role":"{role}","content":{content_json}}}'

        conv_parts.append(conv_str)

    # Compact images array
    image_parts = [f'"{img}"' for img in images]
    images_str = f"[{','.join(image_parts)}]"

    # Final compact JSON line
    conversations_str = f"[{','.join(conv_parts)}]"
    json_line = f'{{"conversations":{conversations_str},"images":{images_str}}}'

    return json_line


class QwenConverter:
    def __init__(
        self,
        multi_image: bool = False,
        include_examples: bool = False,
        max_examples: int = 3,
        examples_file: str = None,
        response_types: Set[str] = None,
        use_candidates: bool = False,
        candidates_file: str = None,
    ):
        """
        Initialize Qwen converter with compact format only.

        Args:
            multi_image: Whether to use multi-image mode for few-shot learning
            include_examples: Whether to include examples in training prompts
            max_examples: Maximum number of examples to include
            examples_file: Path to JSON file containing examples
            response_types: Set of response types to include
            use_candidates: Whether to use reference-based grounding with candidate phrases
            candidates_file: Path to JSON file containing candidate phrases
        """
        self.multi_image = multi_image
        self.include_examples = include_examples
        self.max_examples = max_examples
        self.response_types = response_types or {
            "object_type",
            "property",
            "extra_info",
        }
        self.use_candidates = use_candidates
        self.candidate_phrases = (
            self._load_candidate_phrases(candidates_file) if candidates_file else []
        )

        # Build system prompts based on mode and candidate usage
        if use_candidates and self.candidate_phrases:
            # Use the new categorized candidate phrases template
            if multi_image:
                self.system_prompt = (
                    f"{BASE_MULTI_IMAGE_SYSTEM_PROMPT}\n\n{CANDIDATE_PHRASES_TEMPLATE}"
                )
            else:
                self.system_prompt = (
                    f"{BASE_COMPACT_SYSTEM_PROMPT}\n\n{CANDIDATE_PHRASES_TEMPLATE}"
                )
        else:
            # Use dense captioning prompts (original behavior)
            if multi_image:
                self.system_prompt = MULTI_IMAGE_SYSTEM_PROMPT
            else:
                self.system_prompt = COMPACT_SYSTEM_PROMPT

        # Set user prompts
        if multi_image:
            self.user_prompt = MULTI_IMAGE_QUERY_PROMPT
        else:
            self.user_prompt = STANDARD_USER_PROMPT

        # Load examples from file
        self.examples = self._load_examples(examples_file) if examples_file else []

        mode_name = "Multi-Image" if multi_image else "Standard"
        grounding_mode = "Reference-Based" if use_candidates else "Dense Captioning"
        logger.info("Initialized QwenConverter with:")
        logger.info(f"  Mode: {mode_name}")
        logger.info(f"  Grounding: {grounding_mode}")
        logger.info("  Format: Compact (optimized)")
        logger.info(f"  Include examples: {self.include_examples}")
        logger.info(f"  Max examples: {self.max_examples}")
        logger.info(f"  Loaded examples: {len(self.examples)}")
        if use_candidates:
            logger.info(f"  Candidate phrases: {len(self.candidate_phrases)}")

    def _load_examples(self, examples_file: str) -> List[Dict[str, Any]]:
        """Load example data from JSON file."""
        try:
            with open(examples_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, list):
                # Already a list of examples
                examples = data
            elif isinstance(data, dict):
                # Dictionary with categories - extract all examples
                examples = []
                for category, example in data.items():
                    if (
                        isinstance(example, dict)
                        and "image" in example
                        and "objects" in example
                    ):
                        examples.append(example)
                    elif isinstance(example, list):
                        examples.extend(example)
            else:
                logger.error(f"Unexpected examples file format: {type(data)}")
                return []

            logger.info(f"Loaded {len(examples)} examples from {examples_file}")
            return examples
        except FileNotFoundError:
            logger.warning(f"Examples file not found: {examples_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing examples file {examples_file}: {e}")
            return []

    def _load_candidate_phrases(self, candidates_file: str) -> List[str]:
        """Load candidate phrases from JSON file."""
        try:
            with open(candidates_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract phrase list
            if isinstance(data, dict) and "phrase_list" in data:
                phrases = data["phrase_list"]
            elif isinstance(data, dict) and "phrases" in data:
                # Get keys from phrases dict (sorted by frequency)
                phrases = list(data["phrases"].keys())
            elif isinstance(data, list):
                # Direct list of phrases
                phrases = data
            else:
                logger.error(f"Unexpected candidates file format: {type(data)}")
                return []

            logger.info(
                f"Loaded {len(phrases)} candidate phrases from {candidates_file}"
            )
            return phrases
        except FileNotFoundError:
            logger.warning(f"Candidates file not found: {candidates_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing candidates file {candidates_file}: {e}")
            return []

    def _extract_ref_object_string(self, ref_dict: Dict[str, Any]) -> str:
        """Convert complex JSON ref-object to simplified string format with flexible response types."""
        # Standardize field names first
        standardized_dict = FieldStandardizer.standardize_field_names(ref_dict)

        # Use compact formatter to create simplified comma-separated format
        return CompactResponseFormatter.format_to_compact_string(standardized_dict)

    def _filter_string_description(self, description: str) -> str:
        """Filter existing description string and convert to simplified format."""
        # Convert from verbose format to simplified comma-separated format
        return CompactResponseFormatter.convert_from_verbose_format(description)

    def _process_line_to_sample(
        self, data: Dict[str, Any], is_training: bool = True
    ) -> Dict[str, Any]:
        """Convert input data to Qwen-VL format."""
        images_list = data.get("images")
        if not isinstance(images_list, list) or len(images_list) == 0:
            raise KeyError("Field 'images' missing or empty in input JSONL line")

        # Main image for the query
        main_image = images_list[0]

        objects_map = data.get("objects")
        if not isinstance(objects_map, dict):
            raise KeyError("Field 'objects' missing or not a dict in input JSONL line")

        ref_items_orig = objects_map.get("ref", [])
        bbox_items_orig = objects_map.get("bbox", [])

        if not isinstance(ref_items_orig, list) or not isinstance(
            bbox_items_orig, list
        ):
            raise KeyError(
                "Fields 'objects.ref' or 'objects.bbox' are missing or not lists"
            )

        if len(ref_items_orig) != len(bbox_items_orig):
            raise ValueError(
                "Mismatch between the number of reference items and bounding box items."
            )

        # Process and sort objects using core modules
        processed_refs = []
        for i in range(len(ref_items_orig)):
            ref_string = ref_items_orig[i]

            # Handle both string and dict formats - convert old field names to new ones
            if isinstance(ref_string, dict):
                # Ensure all required fields are present with standardized names
                if "object_type" not in ref_string:
                    ref_string["object_type"] = ""
                if "property" not in ref_string:
                    ref_string["property"] = ""
                if "extra_info" not in ref_string:
                    ref_string["extra_info"] = ""

                ref_string = self._extract_ref_object_string(ref_string)
            elif isinstance(ref_string, str):
                # Filter existing string description based on response types
                ref_string = self._filter_string_description(ref_string)
            else:
                raise ValueError(
                    f"Invalid ref item type: {type(ref_string)}. Expected string or dict."
                )

            processed_refs.append(ref_string)

        # Sort objects using ObjectProcessor
        processed_refs, bbox_items_orig = ObjectProcessor.sort_objects_by_position(
            processed_refs, bbox_items_orig
        )

        # Generate assistant response objects
        response_objects = []
        for i, ref_string in enumerate(processed_refs):
            bbox = bbox_items_orig[i]  # [x1, y1, x2, y2]
            response_objects.append({"bbox": bbox, "description": ref_string})

        # Store as objects for now, we'll handle JSON conversion during writing
        assistant_content = response_objects

        # Build multi-round conversation structure based on guidance
        conversations = [{"role": "system", "content": self.system_prompt}]

        all_images = []

        # Add examples as separate user/assistant pairs for multi-image training
        if self.multi_image and is_training and self.include_examples and self.examples:
            # Select examples
            selected_examples = random.sample(
                self.examples, min(self.max_examples, len(self.examples))
            )

            for example in selected_examples:
                # Get example image and response
                example_image = example.get("image", "")
                example_objects = example.get("objects", [])

                # Format example response objects
                example_response_objects = []
                for obj in example_objects:
                    bbox = obj["bbox"]
                    description = obj["description"]
                    example_response_objects.append(
                        {"bbox": bbox, "description": description}
                    )

                # Store as objects for now
                example_response = example_response_objects

                # Add example as user/assistant pair
                conversations.append(
                    {"role": "user", "content": MULTI_IMAGE_EXAMPLE_PROMPT}
                )
                conversations.append({"role": "assistant", "content": example_response})

                # Add example image to the list
                all_images.append(example_image)

        # Add the main query
        if self.multi_image and is_training and self.include_examples and self.examples:
            # For multi-image with examples, use the multi-image query prompt
            user_content = MULTI_IMAGE_QUERY_PROMPT
        else:
            # For single image or validation, use the appropriate prompt
            if is_training:
                user_content = self.user_prompt
            else:
                # For validation, always use clean prompt
                user_content = STANDARD_USER_PROMPT

        conversations.append({"role": "user", "content": user_content})

        # Add main image
        all_images.append(main_image)

        # Add final assistant response
        conversations.append({"role": "assistant", "content": assistant_content})

        # UNIFIED FORMAT: Always return images as array (even for single image)
        return {
            "conversations": conversations,
            "images": all_images,  # Always array format for unified processing
        }

    def convert_and_split(
        self, input_jsonl: str, output_train: str, output_val: str, val_ratio: float
    ):
        """Convert JSONL and split into train/val sets."""
        all_raw_data: List[Dict[str, Any]] = []

        # Get example image paths to exclude from training
        example_image_paths = set()
        if self.include_examples and self.examples:
            for example in self.examples:
                example_image_paths.add(example.get("image", ""))
            logger.info(
                f"Excluding {len(example_image_paths)} example images from training set"
            )

        # Load all raw data
        with open(input_jsonl, "r", encoding="utf-8") as in_f:
            for line_num, line in enumerate(in_f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    all_raw_data.append(data)
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

        if not all_raw_data:
            logger.warning(
                f"No samples processed from {input_jsonl}. Output files will be empty."
            )
            open(output_train, "w").close()
            open(output_val, "w").close()
            return

        # Filter out example images from training samples
        filtered_raw_data = []
        for data in all_raw_data:
            images_list = data.get("images", [])
            if images_list and images_list[0] not in example_image_paths:
                filtered_raw_data.append(data)
            elif images_list and images_list[0] in example_image_paths:
                logger.debug(f"Excluding example image from training: {images_list[0]}")

        logger.info(f"Total samples: {len(all_raw_data)}")
        logger.info(f"Filtered samples (excluding examples): {len(filtered_raw_data)}")

        # Shuffle and split
        random.shuffle(filtered_raw_data)
        split_index = int(len(filtered_raw_data) * (1 - val_ratio))
        train_data = filtered_raw_data[:split_index]
        val_data = filtered_raw_data[split_index:]

        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")

        # Process training data
        with open(output_train, "w", encoding="utf-8") as train_f:
            for data in train_data:
                try:
                    sample = self._process_line_to_sample(data, is_training=True)
                    train_f.write(write_compact_jsonl_line(sample) + "\n")
                except Exception as e:
                    logger.error(f"Error processing training sample: {e}")
                    continue

        # Process validation data (always without examples)
        with open(output_val, "w", encoding="utf-8") as val_f:
            for data in val_data:
                try:
                    sample = self._process_line_to_sample(data, is_training=False)
                    val_f.write(write_compact_jsonl_line(sample) + "\n")
                except Exception as e:
                    logger.error(f"Error processing validation sample: {e}")
                    continue

        mode_name = "Multi-Image" if self.multi_image else "Standard"
        logger.info(f"✅ {mode_name} conversion completed successfully!")
        logger.info(f"📁 Training file: {output_train}")
        logger.info(f"📁 Validation file: {output_val}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Dataset Converter")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file")
    parser.add_argument(
        "--output_train", required=True, help="Output training JSONL file"
    )
    parser.add_argument(
        "--output_val", required=True, help="Output validation JSONL file"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)"
    )

    # Mode selection
    parser.add_argument(
        "--multi_image",
        action="store_true",
        help="Enable multi-image few-shot learning mode",
    )

    # Example options
    parser.add_argument(
        "--include_examples",
        action="store_true",
        help="Include examples in training prompts",
    )
    parser.add_argument("--examples_file", help="Path to examples JSON file")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=3,
        help="Maximum number of examples to include",
    )

    # Reference-based grounding options
    parser.add_argument(
        "--use_candidates",
        action="store_true",
        help="Enable reference-based grounding with candidate phrases",
    )
    parser.add_argument("--candidates_file", help="Path to candidate phrases JSON file")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--response_types",
        nargs="+",
        choices=["object_type", "property", "extra_info"],
        default=["object_type", "property", "extra_info"],
        help="Response types to include in output (default: all types)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Convert response types to set
    response_types = set(args.response_types)
    logger.info(f"Using response types: {sorted(response_types)}")

    # Initialize converter
    converter = QwenConverter(
        multi_image=args.multi_image,
        include_examples=args.include_examples,
        max_examples=args.max_examples,
        examples_file=args.examples_file,
        response_types=response_types,
        use_candidates=args.use_candidates,
        candidates_file=args.candidates_file,
    )

    # Convert and split
    converter.convert_and_split(
        input_jsonl=args.input_jsonl,
        output_train=args.output_train,
        output_val=args.output_val,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
