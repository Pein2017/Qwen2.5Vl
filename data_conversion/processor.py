#!/usr/bin/env python3
"""
Data Processor - Unified Entry Point

This module provides backward compatibility while routing to the new
unified processor for improved functionality and maintainability.
"""

import argparse
import logging
import sys

from unified_processor import UnifiedProcessor

from config import DataConversionConfig, setup_logging, validate_config


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def main():
    """Main entry point with backward compatibility."""
    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")

    # Required arguments
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with JSON/image files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--language",
        choices=["chinese", "english"],
        required=True,
        help="Language mode",
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset name for organized output (auto-detected from input_dir if not provided)",
    )

    # Processing arguments - REQUIRED
    parser.add_argument(
        "--response_types",
        nargs="+",
        required=True,
        help="Response types to include (e.g., object_type property)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="Validation split ratio (e.g., 0.1)",
    )
    parser.add_argument(
        "--max_teachers",
        type=int,
        required=True,
        help="Maximum teacher samples (e.g., 10)",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed (e.g., 42)"
    )

    # Processing options - OPTIONAL
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Advanced processing options
    parser.add_argument(
        "--geometry_diversity_weight",
        type=float,
        default=4.0,
        help="Weight for geometry diversity in teacher selection",
    )

    args = parser.parse_args()

    # Create configuration from arguments
    config = DataConversionConfig.from_args(args)

    # Setup logging
    setup_logging(config)

    # Validate configuration
    validate_config(config)

    # Create and run unified processor
    processor = UnifiedProcessor(config)
    result = processor.process()

    # Print result for compatibility
    print(f"\n✅ Processing complete: {result}")


if __name__ == "__main__":
    main()
