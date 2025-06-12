#!/usr/bin/env python3
"""
Raw JSON to intermediate JSONL converter using core modules.
Converts raw JSON annotation files to intermediate JSONL format(alinged with ms-swift!) with token mapping.


example outut:
{"images": ["ds_rescaled/QC-20230106-0000211_16520.jpeg"], "objects": {"ref": ["object_type:cabinet not fully occupied;property:match;extra_info:none", "object_type:huawei bbu;property:none;extra_info:none", "object_type:install screw correct;property:none;extra_info:none", "object_type:label matches;property:none;extra_info:none"], "bbox": [[0, 0, 455, 1094], [0, 340, 476, 736], [304, 353, 390, 438], [353, 664, 516, 922]]}, "height": 1200, "width": 900}

"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from core_modules import (
    FieldStandardizer,
    ObjectProcessor,
    ResponseFormatter,
    TokenMapper,
)
from PIL import Image

from vision_process import smart_resize

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON folder to a single JSONL file, with image resizing and token mapping."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Absolute path to the folder containing input JSON files (e.g., /data4/swift/ds).",
    )
    parser.add_argument(
        "--output_image_folder",
        type=str,
        required=True,
        help="Absolute path to the folder where rescaled images will be saved (e.g., /data4/swift/ds_rescaled).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Absolute path to the output JSONL file (e.g., /data4/swift/data_conversion/temp_pure.jsonl).",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=None,
        help="Absolute path to the token map JSON file, required for English.",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=False,
        help="Enable image resizing and bounding box scaling",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "chinese"],
        help="Specify the language for label extraction.",
    )
    args = parser.parse_args()
    input_folder_path = Path(args.input_folder).resolve()
    output_image_folder_path = Path(args.output_image_folder).resolve()
    output_jsonl_path = Path(args.output_jsonl).resolve()

    # Define response types directly in script (no command line argument needed)
    response_types = {"object_type", "property", "extra_info"}
    logger.info(f"Using response types: {sorted(response_types)}")
    logger.info(f"Language mode: {args.language}")

    # Initialize core modules
    token_mapper = None
    if args.language == "english":
        if not args.map_file:
            raise ValueError("--map_file is required for English language mode.")
        token_map_path = Path(args.map_file).resolve()
        token_mapper = TokenMapper(token_map_path)

    if not input_folder_path.is_dir():
        raise FileNotFoundError(
            f"Input folder not found or is not a directory: {input_folder_path}"
        )
    output_image_folder_path.mkdir(parents=True, exist_ok=True)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    file_list = sorted(input_folder_path.rglob("*.json"))
    if not file_list:
        raise FileNotFoundError(f"No JSON files found in {input_folder_path}")

    processed_samples = []

    for input_json_file_abs_path in file_list:
        try:
            data = json.load(input_json_file_abs_path.open("r", encoding="utf-8"))

            # Extract original annotation dimensions
            height = data.get("info", {}).get("height")
            width = data.get("info", {}).get("width")

            # Derive image path directly from JSON filename
            jpeg_path = input_json_file_abs_path.with_suffix(".jpeg")
            jpg_path = input_json_file_abs_path.with_suffix(".jpg")

            if jpeg_path.is_file():
                original_image_abs_path = jpeg_path
            elif jpg_path.is_file():
                original_image_abs_path = jpg_path
            else:
                logger.error(f"Image file not found for {input_json_file_abs_path}")
                raise FileNotFoundError(
                    f"Image file not found for {input_json_file_abs_path}"
                )

            # Get real image size from the file
            with Image.open(original_image_abs_path) as img:
                real_width, real_height = img.size

            # Log the findings
            logger.info(f"Processing {input_json_file_abs_path.name}:")
            if width != real_width or height != real_height:
                logger.warning(
                    f"  Dimension mismatch! JSON: {width}x{height}, Actual: {real_width}x{real_height}"
                )
            else:
                logger.info(f"  Dimensions: {width}x{height}")

            objects_ref = []
            objects_bbox = []

            def extract_and_process_fields(source_dict):
                """Extract and process fields based on language."""
                if args.language == "chinese":
                    content_zh = source_dict.get("contentZh", {})
                    if not content_zh:
                        return ""
                    # Join all values from the contentZh dictionary, handling lists
                    processed_values = []
                    for v in content_zh.values():
                        if isinstance(v, list):
                            # Join list elements into a single string
                            processed_values.append(", ".join(map(str, v)))
                        elif v:
                            # Append non-empty string values
                            processed_values.append(str(v))
                    return ", ".join(processed_values)
                else:  # English
                    # Use FieldStandardizer to extract content
                    content_dict = FieldStandardizer.extract_content_dict(
                        source_dict.get("content", {}), token_mapper
                    )
                    # Convert to string format using ResponseFormatter
                    return ResponseFormatter.format_to_string(
                        content_dict, response_types
                    )

            # Process dataList format
            if "dataList" in data:
                for item_idx, item_data in enumerate(data["dataList"]):
                    coords = item_data.get("coordinates", [])
                    if len(coords) < 2:
                        raise ValueError(
                            f"Invalid coordinates in dataList item {item_idx} for {input_json_file_abs_path}"
                        )
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    objects_bbox.append(
                        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    )

                    props = item_data.get("properties", {}) or {}
                    content_string = extract_and_process_fields(props)

                    allowed_props_keys = {"question", "question_ex", "label"}
                    # Allow 'contentZh' and 'content' in properties
                    props_keys_to_check = {
                        k for k in props.keys() if k not in ["contentZh", "content"]
                    }
                    for key in props_keys_to_check:
                        if key not in allowed_props_keys:
                            raise ValueError(
                                f"Unexpected key '{key}' in properties for dataList item {item_idx} in file {input_json_file_abs_path}. Allowed keys: {allowed_props_keys}"
                            )
                    objects_ref.append(content_string)

            # Process markResult format
            elif "markResult" in data and isinstance(
                data.get("markResult", {}).get("features"), list
            ):
                for feature_idx, feature_data in enumerate(
                    data["markResult"]["features"]
                ):
                    geometry = feature_data.get("geometry", {})
                    coords = geometry.get("coordinates", [])

                    # Ensure coordinates are in a list of lists and have at least one point
                    if (
                        not isinstance(coords, list)
                        or len(coords) == 0
                        or not isinstance(coords[0], list)
                        or len(coords[0]) < 1
                    ):
                        raise ValueError(
                            f"Invalid coordinates in markResult feature {feature_idx} for {input_json_file_abs_path}"
                        )

                    # For polygons, coords can be a list of points or a list containing a list of points.
                    # This handles both `[[x,y], ...]` and `[[[x,y], ...]]` structures.
                    points = coords
                    if (
                        points
                        and isinstance(points[0], list)
                        and points[0]
                        and isinstance(points[0][0], list)
                    ):
                        points = points[0]

                    # Extract min/max to form the bounding box
                    if not points or any(len(p) != 2 for p in points):
                        raise ValueError(
                            f"Invalid points list in markResult feature {feature_idx} for {input_json_file_abs_path}"
                        )

                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]

                    objects_bbox.append(
                        [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    )

                    properties = feature_data.get("properties", {})
                    content_string = extract_and_process_fields(properties)

                    content_from_feature = properties.get("content", {})
                    allowed_content_keys = {"label", "question", "question_ex"}
                    for key in content_from_feature.keys():
                        if key not in allowed_content_keys:
                            raise ValueError(
                                f"Unexpected key '{key}' in content for markResult feature {feature_idx} in file {input_json_file_abs_path}. Allowed keys: {allowed_content_keys}"
                            )
                    objects_ref.append(content_string)
            else:
                logger.warning(
                    f"No annotation entries (dataList or markResult.features) found in {input_json_file_abs_path}, skipping file."
                )
                continue

            # Sort objects by bounding box coordinates using ObjectProcessor
            objects_ref, objects_bbox = ObjectProcessor.sort_objects_by_position(
                objects_ref, objects_bbox
            )

            logger.info(f"  Found {len(objects_bbox)} objects.")
            for i, bbox in enumerate(objects_bbox):
                logger.debug(f"    Original bbox {i}: {bbox}")

            # Process image resizing
            if args.resize:
                try:
                    output_image_abs_path = (
                        output_image_folder_path / original_image_abs_path.name
                    )
                    new_height, new_width = smart_resize(
                        height=real_height, width=real_width
                    )

                    # Resize and save the image
                    with Image.open(original_image_abs_path) as img:
                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        resized_img.save(output_image_abs_path)

                    logger.info(f"  Resized image to: {new_width}x{new_height}")

                    # Scale bounding boxes using JSON dimensions
                    scaled_objects_bbox = []
                    for bbox in objects_bbox:
                        try:
                            scaled_bbox = ObjectProcessor.scale_bbox(
                                bbox,
                                original_width=width,  # Use JSON width
                                original_height=height,  # Use JSON height
                                new_width=new_width,
                                new_height=new_height,
                            )
                            scaled_objects_bbox.append(scaled_bbox)
                        except ValueError as e:
                            logger.error(
                                f"Error scaling bounding box for file: {original_image_abs_path.name}"
                            )
                            logger.error(f"  Problematic BBox: {bbox}")
                            logger.error(f"  JSON dimensions: {width}x{height}")
                            logger.error(
                                f"  Actual image dimensions: {real_width}x{real_height}"
                            )
                            logger.error(
                                f"  Target resize dimensions: {new_width}x{new_height}"
                            )
                            raise e

                    for i, bbox in enumerate(scaled_objects_bbox):
                        logger.debug(f"    Scaled bbox {i}: {bbox}")

                    objects_bbox = scaled_objects_bbox
                    width, height = new_width, new_height

                except Exception as e:
                    print(f"Error processing file: {input_json_file_abs_path}")
                    raise e
            else:
                # If not resizing, still copy the image to the output directory
                output_image_abs_path = (
                    output_image_folder_path / original_image_abs_path.name
                )
                if not output_image_abs_path.exists():
                    shutil.copy(original_image_abs_path, output_image_abs_path)

            # Build image path for JSONL relative to the script's execution location
            try:
                final_image_path_for_jsonl = str(
                    output_image_abs_path.relative_to(output_jsonl_path.parent.parent)
                )
            except ValueError:
                final_image_path_for_jsonl = str(output_image_abs_path)

            sample = {
                "images": [final_image_path_for_jsonl],
                "objects": {"ref": objects_ref, "bbox": objects_bbox},
                "height": height,
                "width": width,
            }
            processed_samples.append(sample)
        except Exception as e:
            print(f"Error processing file: {input_json_file_abs_path}")
            raise e

    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"✅ Successfully converted {len(processed_samples)} files.")
    logger.info(f"Intermediate JSONL file saved to: {output_jsonl_path}")

    # Report missing tokens if applicable
    if args.language == "english" and token_mapper:
        missing_tokens = token_mapper.get_missing_tokens()
        if missing_tokens:
            logger.warning(
                f"Found {len(missing_tokens)} tokens in data that are not in the token map:"
            )
            for token in sorted(missing_tokens):
                logger.warning(f"  - '{token}'")
        else:
            logger.info("All tokens found in the token map.")


if __name__ == "__main__":
    main()
