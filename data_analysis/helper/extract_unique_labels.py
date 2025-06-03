import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_labels_from_json(file_path: Path) -> set[str]:
    labels_found = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Format B: dataList entries
        if "dataList" in data and isinstance(data["dataList"], list):
            for item in data["dataList"]:
                if isinstance(item, dict):
                    label = item.get("label")
                    if label and isinstance(label, str) and label.strip():
                        labels_found.add(label.strip())

        # Format A: markResult.features entries
        elif "markResult" in data and isinstance(data.get("markResult"), dict):
            features = data["markResult"].get("features")
            if isinstance(features, list):
                for feature in features:
                    if isinstance(feature, dict):
                        properties = feature.get("properties")
                        if isinstance(properties, dict):
                            content = properties.get("content")
                            if isinstance(content, dict):
                                label = content.get("label")
                                if label and isinstance(label, str) and label.strip():
                                    labels_found.add(label.strip())
        else:
            logger.debug(
                f"No 'dataList' or 'markResult.features' found in {file_path}, or structure is unexpected."
            )

    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path}: {e}")
    return labels_found


def main():
    parser = argparse.ArgumentParser(
        description="Extract all unique 'label' strings from JSON files in a folder."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Absolute path to the folder containing input JSON files (e.g., /data4/swift/ds).",
    )
    args = parser.parse_args()

    input_folder_path = Path(args.input_folder).resolve()

    if not input_folder_path.is_dir():
        logger.critical(
            f"Input folder not found or is not a directory: {input_folder_path}"
        )
        sys.exit(1)

    all_unique_labels = set()
    json_files = sorted(list(input_folder_path.rglob("*.json")))

    if not json_files:
        logger.info(f"No JSON files found in {input_folder_path}")
        return

    logger.info(
        f"Found {len(json_files)} JSON files to process in {input_folder_path}."
    )

    for file_count, json_file_path in enumerate(json_files, 1):
        logger.info(
            f"Processing file {file_count}/{len(json_files)}: {json_file_path.name}"
        )
        labels_in_file = extract_labels_from_json(json_file_path)
        all_unique_labels.update(labels_in_file)

    if not all_unique_labels:
        logger.info("No unique labels found across all processed files.")
    else:
        logger.info(f"Found {len(all_unique_labels)} unique labels:")
        for label in sorted(list(all_unique_labels)):  # Sort for consistent output
            print(label)


if __name__ == "__main__":
    main()
