#!/usr/bin/env python3
"""
Invalid Objects Validator

This script validates the objects in the invalid_objects.jsonl file
and provides detailed error reports using the UnifiedDatasetAnalyzer.
It also attempts to fix invalid objects when possible.

Usage:
    python validate_invalid_objects.py [--input_file PATH] [--output_dir PATH] [--attempt_fix]
"""

import argparse
import logging
import sys
from pathlib import Path


# Ensure project root is on module search path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from data_analysis.unified_analyzer import UnifiedDatasetAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Validate invalid objects, attempt fixes, and generate detailed reports."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Invalid Objects Validator")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/ds_v2_full/invalid_objects.jsonl",
        help="Path to the invalid objects JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_analysis/validation_results",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--attempt_fix",
        action="store_true",
        help="Attempt to fix invalid objects",
    )

    args = parser.parse_args()

    # Ensure input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer with the invalid objects file
    logger.info(f"Validating objects from {input_path}")
    analyzer = UnifiedDatasetAnalyzer(str(input_path))
    analyzer.load_data()

    # Skip analysis and run validation directly
    analyzer.validate_objects()

    # Export results
    validation_output = output_dir / "validation_results.json"
    invalid_objects_output = output_dir / "invalid_objects.jsonl"
    invalid_samples_output = output_dir / "invalid_samples.jsonl"

    analyzer.export_validation_results(str(validation_output))
    analyzer.export_invalid_objects(str(invalid_objects_output))
    analyzer.export_invalid_samples(str(invalid_samples_output))

    logger.info(
        f"Exported visualization-friendly invalid samples to {invalid_samples_output}"
    )

    # Print summary
    analyzer.print_validation_summary()

    # Attempt to fix invalid objects if requested or if there are any invalid objects
    if args.attempt_fix and analyzer.validation_results["invalid_objects"] > 0:
        logger.info("Attempting to fix invalid objects...")
        fix_stats = analyzer.attempt_fix_invalid_objects()

        if fix_stats["fixed_objects"] > 0:
            fixed_objects_output = output_dir / "fixed_objects.jsonl"
            analyzer.export_fixed_objects(fix_stats, str(fixed_objects_output))
            analyzer.print_fix_summary(fix_stats)

            logger.info(
                f"Fixed {fix_stats['fixed_objects']} objects, "
                f"{fix_stats['unfixable_objects']} remain unfixable"
            )
            logger.info(f"Fixed objects saved to {fixed_objects_output}")
        else:
            logger.info("No objects could be fixed automatically.")

    logger.info(f"Validation results saved to {output_dir}")

    # Print error counts by category
    if analyzer.validation_results["error_categories"]:
        print("\nError categories:")
        for category, count in sorted(
            analyzer.validation_results["error_categories"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
