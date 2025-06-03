import argparse
import json
import logging
import re
from pathlib import Path
from typing import Set

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Regex to detect Chinese characters
CHINESE_CHAR_REGEX = re.compile(r"[\u4e00-\u9fff]")


def has_chinese_chars(text: str) -> bool:
    """Check if the given text contains any Chinese characters."""
    if not isinstance(text, str):
        return False
    return bool(CHINESE_CHAR_REGEX.search(text))


def main():
    parser = argparse.ArgumentParser(
        description="Extract Chinese captions from 'q' and 'qe' fields in JSON files within a folder."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Absolute path to the folder containing input JSON files.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Absolute path to the output JSON file for Chinese captions (e.g., /data4/swift/data_conversion/chinese_captions.json).",
    )

    args = parser.parse_args()
    input_folder_path = Path(args.input_folder).resolve()
    output_json_path = Path(args.output_json).resolve()

    if not input_folder_path.is_dir():
        logger.error(
            f"Input folder not found or is not a directory: {input_folder_path}"
        )
        raise NotADirectoryError(
            f"Input folder not found or is not a directory: {input_folder_path}"
        )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    all_chinese_captions: Set[str] = set()

    json_files = sorted(list(input_folder_path.rglob("*.json")))

    if not json_files:
        logger.warning(
            f"No JSON files found in {input_folder_path}. Nothing to process."
        )
        # Write an empty list to the output file if no JSON files are found
        with open(output_json_path, "w", encoding="utf-8") as f_out:
            json.dump([], f_out, ensure_ascii=False, indent=2)
        logger.info(
            f"Successfully wrote empty list to {output_json_path} as no input files were found."
        )
        return

    for json_file_path in json_files:
        logger.info(f"Processing file: {json_file_path}")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Skipping file {json_file_path} due to JSON decode error: {e}"
            )
            continue
        except Exception as e:
            logger.warning(
                f"Skipping file {json_file_path} due to an unexpected error: {e}"
            )
            continue

        # Old path: objects_data = data.get("objects")
        # Old path: objects_ref = objects_data.get("ref")
        # New path: Iterate through features in markResult
        mark_result = data.get("markResult")
        if not isinstance(mark_result, dict):
            logger.warning(
                f"Skipping file {json_file_path}: 'markResult' field is missing or not a dict."
            )
            continue

        features_list = mark_result.get("features")
        if not isinstance(features_list, list):
            logger.warning(
                f"Skipping file {json_file_path}: 'markResult.features' field is missing or not a list."
            )
            continue

        # Iterate through each feature item instead of ref_item
        for feature_idx, feature_item in enumerate(features_list):
            if not isinstance(feature_item, dict):
                logger.warning(
                    f"Skipping an item (index {feature_idx}) in 'markResult.features' in file {json_file_path}: item is not a dict."
                )
                continue

            properties = feature_item.get("properties")
            if not isinstance(properties, dict):
                logger.warning(
                    f"Skipping feature item (index {feature_idx}) in file {json_file_path}: 'properties' field is missing or not a dict."
                )
                continue

            content = properties.get("content")
            if not isinstance(content, dict):
                logger.warning(
                    f"Skipping feature item (index {feature_idx}) in file {json_file_path}: 'properties.content' field is missing or not a dict."
                )
                continue

            # Updated keys from 'q' to 'question' and 'qe' to 'question_ex'
            question_val = content.get("question")
            question_ex_val = content.get("question_ex")

            # Extract from 'question' field (formerly 'q')
            if isinstance(question_val, str):
                if has_chinese_chars(question_val):
                    all_chinese_captions.add(question_val)
            elif isinstance(question_val, list):
                for q_idx, q_item in enumerate(question_val):
                    if isinstance(q_item, str) and has_chinese_chars(q_item):
                        all_chinese_captions.add(q_item)
                    elif not isinstance(q_item, str):
                        logger.warning(
                            f"Non-string item found in 'question' list (index {q_idx}) in file {json_file_path}, feature_item index {feature_idx}"
                        )
            elif (
                question_val is not None
            ):  # question field exists but is not str or list
                logger.warning(
                    f"Unexpected type for 'question' field: {type(question_val)} in file {json_file_path}, feature_item index {feature_idx}"
                )

            # Extract from 'question_ex' field (formerly 'qe')
            if isinstance(question_ex_val, str):
                if has_chinese_chars(question_ex_val):
                    all_chinese_captions.add(question_ex_val)
            elif question_ex_val is not None:  # question_ex field exists but is not str
                logger.warning(
                    f"Unexpected type for 'question_ex' field: {type(question_ex_val)} in file {json_file_path}, feature_item index {feature_idx}"
                )

    # Convert set to list for JSON serialization
    sorted_chinese_captions = sorted(list(all_chinese_captions))

    with open(output_json_path, "w", encoding="utf-8") as f_out:
        json.dump(sorted_chinese_captions, f_out, ensure_ascii=False, indent=2)

    logger.info(
        f"Successfully extracted {len(sorted_chinese_captions)} unique Chinese captions to {output_json_path}"
    )


if __name__ == "__main__":
    main()
