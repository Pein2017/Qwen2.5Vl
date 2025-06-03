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
        required=True,
        help="Absolute path to the token map JSON file (e.g., /data4/swift/data_conversion/token_map.json).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=False,
        help="Enable image resizing and bounding box scaling",
    )
    parser.add_argument(
        "--response_types",
        nargs="+",
        choices=["object_type", "property", "extra_info"],
        default=["object_type", "property", "extra_info"],
        help="Response types to include in output (default: all types)",
    )

    args = parser.parse_args()
    input_folder_path = Path(args.input_folder).resolve()
    output_image_folder_path = Path(args.output_image_folder).resolve()
    output_jsonl_path = Path(args.output_jsonl).resolve()
    token_map_path = Path(args.map_file).resolve()

    # Convert response types to set
    response_types = set(args.response_types)
    logger.info(f"Using response types: {sorted(response_types)}")

    # Initialize core modules
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
        data = json.load(input_json_file_abs_path.open("r", encoding="utf-8"))

        # Extract original annotation dimensions
        height = data.get("info", {}).get("height")
        width = data.get("info", {}).get("width")

        # Derive image path directly from JSON filename
        base = input_json_file_abs_path.stem
        jpeg_path = input_folder_path / f"{base}.jpeg"
        jpg_path = input_folder_path / f"{base}.jpg"
        if jpeg_path.is_file():
            original_image_abs_path = jpeg_path
        elif jpg_path.is_file():
            original_image_abs_path = jpg_path
        else:
            logger.error(f"Image file not found for {input_json_file_abs_path}")
            raise FileNotFoundError(
                f"Image file not found for {input_json_file_abs_path}"
            )

        objects_ref = []
        objects_bbox = []

        def extract_and_process_fields(source_dict):
            """Extract and process fields using core modules."""
            # Use FieldStandardizer to extract content
            content_dict = FieldStandardizer.extract_content_dict(
                source_dict, token_mapper
            )

            # Convert to string format using ResponseFormatter
            return ResponseFormatter.format_to_string(content_dict, response_types)

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
                objects_bbox.append([x1, y1, x2, y2])

                props = item_data.get("properties", {}) or {}
                content_string = extract_and_process_fields(props)

                allowed_props_keys = {"question", "question_ex", "label"}
                for key in props.keys():
                    if key not in allowed_props_keys:
                        raise ValueError(
                            f"Unexpected key '{key}' in properties for dataList item {item_idx} in file {input_json_file_abs_path}. Allowed keys: {allowed_props_keys}"
                        )
                objects_ref.append(content_string)

        # Process markResult format
        elif "markResult" in data and isinstance(
            data.get("markResult", {}).get("features"), list
        ):
            for feature_idx, feature_data in enumerate(data["markResult"]["features"]):
                coords = feature_data.get("geometry", {}).get("coordinates", [])
                if not isinstance(coords, list) or len(coords) < 3:
                    raise ValueError(
                        f"Invalid coordinates in markResult feature {feature_idx} for {input_json_file_abs_path}"
                    )
                x1, y1 = coords[0]
                x2, y2 = coords[2]
                objects_bbox.append([x1, y1, x2, y2])

                content_from_feature = feature_data.get("properties", {}).get(
                    "content", {}
                )
                content_string = extract_and_process_fields(content_from_feature)

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

        # Process image resizing
        orig_img_pil = Image.open(original_image_abs_path)

        # Determine relative path of the image within the input folder
        relative_image_path_to_input_dir = original_image_abs_path.relative_to(
            input_folder_path
        )

        # Always ensure the output image path exists
        rescaled_image_abs_path = (
            output_image_folder_path / relative_image_path_to_input_dir
        ).resolve()
        rescaled_image_abs_path.parent.mkdir(parents=True, exist_ok=True)

        if args.resize:
            # Perform image resizing and adjust bounding boxes
            orig_width, orig_height = orig_img_pil.size
            new_height, new_width = smart_resize(orig_height, orig_width)

            if new_height != orig_height or new_width != orig_width:
                logger.debug(
                    f"Resized image from {orig_height}x{orig_width} to {new_height}x{new_width}"
                )
                # Save resized image
                resized_img_pil = orig_img_pil.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                resized_img_pil.save(str(rescaled_image_abs_path))
            else:
                # No actual resizing needed, but still copy the image
                logger.debug(
                    f"No resizing needed for {original_image_abs_path}, copying original"
                )
                orig_img_pil.save(str(rescaled_image_abs_path))

            # Adjust bounding boxes according to scale using ObjectProcessor
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height
            scaled_objects_bbox = [
                ObjectProcessor.scale_bbox(bbox, scale_x, scale_y)
                for bbox in objects_bbox
            ]
        else:
            # No resizing: copy original image and keep original bounding boxes
            logger.debug(
                f"Copying original image without resizing: {original_image_abs_path}"
            )
            orig_img_pil.save(str(rescaled_image_abs_path))
            scaled_objects_bbox = objects_bbox

        # Build image path prefix from the output_image_folder argument
        final_image_path_for_jsonl = str(
            Path(args.output_image_folder) / relative_image_path_to_input_dir
        )

        # Write output sample
        sample = {
            "images": [final_image_path_for_jsonl],
            "objects": {"ref": objects_ref, "bbox": scaled_objects_bbox},
            "height": height,
            "width": width,
        }
        processed_samples.append(sample)

    # Check for missing tokens
    missing_tokens = token_mapper.get_missing_tokens()
    if missing_tokens:
        logger.error(
            f"Found {len(missing_tokens)} token(s) that are not in the token_map and are not empty strings."
        )
        logger.error(
            "Please add them to your token_map.json or ensure they are empty strings. Missing tokens:"
        )
        for token in sorted(list(missing_tokens)):
            logger.error(token)
        logger.error("Output JSONL file will NOT be created due to missing tokens.")
        return

    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for sample in processed_samples:
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(
        f"Successfully wrote {len(processed_samples)} samples to JSONL: {output_jsonl_path}"
    )


if __name__ == "__main__":
    main()
