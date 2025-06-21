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
from typing import Dict, List, Set, Union

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


def extract_phrases_from_description(
    description: str, response_types: Set[str]
) -> List[str]:
    """Extract meaningful phrases from a description string, respecting chosen response types."""
    if not description:
        return []

    # Unify separator: convert commas to slashes
    description = description.replace(", ", "/").replace(",", "/")

    # If the description contains semicolons, it's a structured English string
    if ";" in description:
        # Parse the description to get components
        components = ResponseFormatter.parse_description_string(description)
        phrases = []

        # Helper to push phrases conditionally
        def maybe_add(items: Union[str, List[str]], key: str):
            if key not in response_types:
                return
            if not items:
                return
            if isinstance(items, list):
                for item in items:
                    if item and item != "none":
                        phrases.append(item)
            else:
                item_str = str(items).strip()
                if item_str and item_str != "none":
                    if "," in item_str:
                        phrases.extend(
                            [p.strip() for p in item_str.split(",") if p.strip()]
                        )
                    else:
                        phrases.append(item_str)

        maybe_add(components.get("object_type", "").strip(), "object_type")
        maybe_add(components.get("property", "").strip(), "property")
        maybe_add(components.get("extra_info", "").strip(), "extra_info")

        return phrases
    else:
        # Chinese descriptions: return each slash-delimited or plain segment as a whole phrase
        segments = [s.strip() for s in description.split(",") if s.strip()]
        return segments


def extract_unique_phrases(
    input_jsonl: str, response_types: Set[str]
) -> Dict[str, int]:
    """Extract all unique phrases from the JSONL file with frequency counts, using selected response types."""
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
                    phrases = extract_phrases_from_description(ref_desc, response_types)
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
    """Save phrases to a single JSON file with metadata."""
    # Filter by minimum frequency
    filtered_phrases = {
        phrase: count
        for phrase, count in phrases_dict.items()
        if count >= min_frequency
    }

    # Sort by frequency (descending) then alphabetically
    sorted_phrases = dict(sorted(filtered_phrases.items(), key=lambda x: (-x[1], x[0])))

    # Create output data structure
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

    # Save to single JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(sorted_phrases)} phrases to {output_file}")

    # Display statistics
    print("\n📊 Phrase Extraction Results:")
    print("===============================")
    print(f"Total unique phrases: {len(sorted_phrases)}")
    print(f"Minimum frequency: {min_frequency}")
    print(f"Candidates file: {output_file} (JSON)")

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
    parser.add_argument(
        "--response_types",
        default="object_type property extra_info",
        help="Space-separated response types to consider when extracting phrases.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Validate input file
    if not Path(args.input_jsonl).exists():
        logger.error(f"Input file not found: {args.input_jsonl}")
        return

    # Extract phrases
    response_types = set(args.response_types.split())
    phrases_dict = extract_unique_phrases(args.input_jsonl, response_types)

    # Save results
    save_phrases(phrases_dict, args.output_phrases, args.min_frequency)

    print(f"\n✅ Phrase extraction complete! Results saved to: {args.output_phrases}")
    print(
        "📝 You can now use this phrases file with the converter for reference-based grounding."
    )


if __name__ == "__main__":
    main()
