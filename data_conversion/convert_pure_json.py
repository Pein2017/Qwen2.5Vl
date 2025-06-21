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
    ObjectProcessor,
    ResponseFormatter,
    TokenMapper,
)
from PIL import Image

from vision_process import smart_resize

# Configure logging to file
LOG_FILE = Path(__file__).parent / "convert.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=str(LOG_FILE),
    filemode="a",
)

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        type=str2bool,
        required=True,
        help="Enable image resizing and bounding box scaling (True/False)",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=["english", "chinese"],
        help="Specify the language for label extraction.",
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
        type=str,
        default="object_type property",
        help='A space-separated string of response types to include. Example: "object_type property extra_info".',
    )
    args = parser.parse_args()

    # Further processing
    input_folder_path = Path(args.input_folder)
    output_image_folder_path = Path(args.output_image_folder).resolve()
    output_jsonl_path = Path(args.output_jsonl).resolve()

    # Parse response types from CLI and convert to a set
    response_types = set(args.response_types.split())
    logger.info(f"Using response types: {sorted(response_types)}")
    logger.info(f"Language mode: {args.language}")

    # Initialize core modules
    token_mapper = None
    if args.map_file:
        token_map_path = Path(args.map_file).resolve()
        token_mapper = TokenMapper(token_map_path)
    elif args.language == "english":
        raise ValueError("--map_file is required for English language mode.")

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

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

            # Define the output path for the potentially rescaled image
            output_image_rel_path = original_image_abs_path.relative_to(
                input_folder_path
            )
            output_image_abs_path = output_image_folder_path / output_image_rel_path
            output_image_abs_path.parent.mkdir(parents=True, exist_ok=True)

            # Obtain the actual pixel dimensions directly from the image. All
            # images are expected to have been normalised (EXIF orientation
            # removed) beforehand by `strip_exif_orientation.py`. Any mismatch
            # between these dimensions and those recorded in the JSON is an
            # error.
            with Image.open(original_image_abs_path) as img:
                real_width, real_height = img.size

            # Log the findings
            logger.debug(f"Processing {input_json_file_abs_path.name}:")
            # Fail-fast on dimension mismatch between JSON and actual image
            if not (width == real_width and height == real_height):
                raise ValueError(
                    f"Dimension mismatch for {input_json_file_abs_path.name}: JSON reports {width}x{height} but image is {real_width}x{real_height}."
                )

            objects_ref: list[str] = []
            objects_bbox: list[list[float]] = []

            def extract_and_process_fields(source_dict):
                """Extract and process fields based on language."""
                if args.language == "chinese":
                    content_zh = source_dict.get("contentZh", {})
                    if not content_zh:
                        return ""
                    # Extract label entries from contentZh: keys containing '标签'
                    label_values = []
                    for key, v in content_zh.items():
                        if "标签" in key:
                            if isinstance(v, list):
                                label_values.append(", ".join(map(str, v)))
                            elif v:
                                label_values.append(str(v))
                    if not label_values:
                        return ""
                    # Use the first label entry
                    label_string = label_values[0]
                    # Split into object_type/property/extra_info
                    parts = [p.strip() for p in label_string.split("/")]
                    object_type = parts[0] if len(parts) >= 1 else ""
                    property_value = parts[1] if len(parts) >= 2 else ""
                    extra_info = parts[2] if len(parts) >= 3 else ""
                    # Map tokens if mapper is available
                    content_dict = {
                        "object_type": token_mapper.map_token(object_type)
                        if token_mapper
                        else object_type,
                        "property": token_mapper.map_token(property_value)
                        if token_mapper
                        else property_value,
                        "extra_info": token_mapper.map_token(extra_info)
                        if token_mapper
                        else extra_info,
                    }
                    # For Chinese mode, preserve grouping with slash
                    group_parts = []
                    if content_dict["object_type"]:
                        group_parts.append(content_dict["object_type"])
                    if content_dict["property"]:
                        group_parts.append(content_dict["property"])
                    if content_dict["extra_info"]:
                        group_parts.append(content_dict["extra_info"])
                    # Normalize separators: replace commas with slashes
                    content_string = "/".join(group_parts)
                    content_string = content_string.replace(", ", "/").replace(",", "/")
                    return content_string
                else:  # English
                    # Extract fields directly with unified naming (fallback to old names)
                    content_dict_raw = {
                        "object_type": source_dict.get("object_type")
                        or source_dict.get("label", ""),
                        "property": source_dict.get("property")
                        or source_dict.get("question", ""),
                        "extra_info": source_dict.get("extra_info")
                        or source_dict.get("question_ex", ""),
                    }

                    # Apply token mapping to each field (English mode only)
                    content_dict = {
                        k: token_mapper.map_token(v)
                        if args.language == "english"
                        else v
                        for k, v in content_dict_raw.items()
                    }
                    # Convert to string format using ResponseFormatter and normalize separators
                    content_string = ResponseFormatter.format_to_string(
                        content_dict, response_types
                    )
                    content_string = content_string.replace(", ", "/").replace(",", "/")
                    return content_string

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

                    allowed_props_keys = {
                        "object_type",
                        "property",
                        "extra_info",
                        "label",
                        "question",
                        "question_ex",
                    }
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
                    allowed_content_keys = {
                        "object_type",
                        "property",
                        "extra_info",
                        "label",
                        "question",
                        "question_ex",
                    }
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

            # Sort objects by bounding box coordinates (no extra orientation
            # transform needed because we already applied EXIF transpose to the
            # image; the annotation coordinates are defined in that oriented
            # space).

            objects_ref, objects_bbox = ObjectProcessor.sort_objects_by_position(
                objects_ref, objects_bbox
            )

            logger.debug(f"  Found {len(objects_bbox)} objects.")
            for i, bbox in enumerate(objects_bbox):
                logger.debug(f"    Original bbox {i}: {bbox}")

            # Process image resizing
            if args.resize:
                try:
                    # Preserve the original directory structure **inside** the
                    # rescaled folder so that downstream components (e.g.
                    # visualisation scripts) can locate the image using the
                    # same relative path regardless of whether `--resize` is
                    # enabled.
                    output_image_abs_path = (
                        output_image_folder_path / output_image_rel_path
                    )
                    output_image_abs_path.parent.mkdir(parents=True, exist_ok=True)
                    # Important: use the oriented dimensions here.  We reuse
                    # `real_height` / `real_width` which already reflect EXIF
                    # orientation.
                    new_height, new_width = smart_resize(
                        height=real_height, width=real_width
                    )

                    # Resize and save the image
                    with Image.open(original_image_abs_path) as img:
                        resized_img = img.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        resized_img.save(output_image_abs_path)

                    logger.debug(f"  Resized image to: {new_width}x{new_height}")

                    # Scale bounding boxes using JSON dimensions
                    scaled_objects_bbox = []
                    for bbox in objects_bbox:
                        try:
                            scaled_bbox = ObjectProcessor.scale_bbox(
                                bbox,
                                original_width=real_width,
                                original_height=real_height,
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
                # Keep the directory structure when simply copying the image
                # (no resize).  This guarantees path consistency regardless of
                # whether --resize is enabled.
                output_image_abs_path = output_image_folder_path / output_image_rel_path
                output_image_abs_path.parent.mkdir(parents=True, exist_ok=True)
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
