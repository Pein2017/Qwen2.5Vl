#!/usr/bin/env python3
"""
Extract Unique Phrases for Reference-Based Grounding

This script extracts all unique description phrases from the intermediate JSONL data
to create candidate lists for reference-based grounding tasks.

Usage:
    python extract_unique_phrases.py --input_jsonl data_conversion/qwen_combined.jsonl --output_phrases candidates_phrases.json
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

from core_modules import ResponseFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_phrases_from_description(description: str) -> List[str]:
    """Extract meaningful phrases from a description string."""
    if not description:
        return []

    # Parse the description to get components
    components = ResponseFormatter.parse_description_string(description)

    phrases = []

    # Extract object_type
    object_type = components.get("object_type", "").strip()
    if object_type and object_type != "none":
        phrases.append(object_type)

    # Extract property
    property_value = components.get("property", "").strip()
    if property_value and property_value != "none":
        # Split on commas for multiple properties
        if "," in property_value:
            phrases.extend([p.strip() for p in property_value.split(",") if p.strip()])
        else:
            phrases.append(property_value)

    # Extract extra_info
    extra_info = components.get("extra_info", "").strip()
    if extra_info and extra_info != "none":
        # Split on commas for multiple extra info items
        if "," in extra_info:
            phrases.extend([e.strip() for e in extra_info.split(",") if e.strip()])
        else:
            phrases.append(extra_info)

    return phrases


def extract_unique_phrases(input_jsonl: str) -> Dict[str, int]:
    """Extract all unique phrases from the JSONL file with frequency counts."""
    phrase_counter = Counter()
    total_samples = 0

    logger.info(f"Reading JSONL file: {input_jsonl}")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_samples += 1

                # Extract objects
                objects = data.get("objects", {})
                ref_items = objects.get("ref", [])

                # Process each reference description
                for ref_desc in ref_items:
                    phrases = extract_phrases_from_description(ref_desc)
                    for phrase in phrases:
                        phrase_counter[phrase] += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                continue

    logger.info(f"Processed {total_samples} samples")
    logger.info(f"Found {len(phrase_counter)} unique phrases")

    return dict(phrase_counter)


def save_phrases(
    phrases_dict: Dict[str, int], output_file: str, min_frequency: int = 1
):
    """Save phrases to text file with optional frequency filtering."""
    # Filter by minimum frequency
    filtered_phrases = {
        phrase: count
        for phrase, count in phrases_dict.items()
        if count >= min_frequency
    }

    # Sort by frequency (descending) then alphabetically
    sorted_phrases = dict(sorted(filtered_phrases.items(), key=lambda x: (-x[1], x[0])))

    # Extract just the phrase list for the candidates file
    phrase_list = list(sorted_phrases.keys())

    # Save to file as plain text (one phrase per line)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(phrase_list))

    logger.info(f"Saved {len(sorted_phrases)} phrases to {output_file}")

    # Also save detailed metadata to a separate JSON file for analysis
    metadata_file = output_path.with_suffix(".metadata.json")
    output_data = {
        "metadata": {
            "total_unique_phrases": len(sorted_phrases),
            "min_frequency_threshold": min_frequency,
            "most_common_phrase": max(sorted_phrases.items(), key=lambda x: x[1])
            if sorted_phrases
            else None,
        },
        "phrases": sorted_phrases,
        "phrase_list": list(sorted_phrases.keys()),  # For easy access
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved detailed metadata to {metadata_file}")

    # Display statistics
    print("\n📊 Phrase Extraction Results:")
    print("===============================")
    print(f"Total unique phrases: {len(sorted_phrases)}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Candidates file: {output_file} (plain text)")
    print(f"Metadata file: {metadata_file} (JSON)")

    if sorted_phrases:
        top_10 = list(sorted_phrases.items())[:10]
        print("\n🔝 Top 10 most frequent phrases:")
        for i, (phrase, count) in enumerate(top_10, 1):
            print(f"  {i:2d}. {phrase} ({count} occurrences)")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique phrases for reference-based grounding"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file (intermediate format)",
    )
    parser.add_argument(
        "--output_phrases",
        type=str,
        required=True,
        help="Path to output phrases JSON file",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=1,
        help="Minimum frequency threshold for including phrases (default: 1)",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_jsonl).exists():
        logger.error(f"Input file not found: {args.input_jsonl}")
        return

    # Extract phrases
    phrases_dict = extract_unique_phrases(args.input_jsonl)

    # Save results
    save_phrases(phrases_dict, args.output_phrases, args.min_frequency)

    print(f"\n✅ Phrase extraction complete! Results saved to: {args.output_phrases}")
    print(
        "📝 You can now use this phrases file with the converter for reference-based grounding."
    )


if __name__ == "__main__":
    main()
