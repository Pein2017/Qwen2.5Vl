#!/usr/bin/env python3
"""
Data Processor - Unified Entry Point

This module provides backward compatibility while routing to the new
unified processor for improved functionality and maintainability.
"""

import argparse
import logging
import sys

from config import DataConversionConfig, setup_logging, validate_config
from unified_processor import UnifiedProcessor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def main():
    """Main entry point with backward compatibility."""
    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, help="Input directory with JSON/image files")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--language", choices=["chinese", "english"], required=True, help="Language mode")
    
    # Optional arguments
    parser.add_argument("--output_image_dir", help="Output directory for processed images")
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument("--response_types", nargs="+", default=["object_type", "property"],
                       help="Response types to include")
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max_teachers", type=int, default=10, help="Maximum teacher samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
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