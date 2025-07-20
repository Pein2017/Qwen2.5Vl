#!/usr/bin/env python3
"""
Test script to verify rank-aware logging implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger_utils import configure_global_logging, get_logger

def test_rank_aware_logging():
    """Test the rank-aware logging functionality."""
    print("🧪 Testing rank-aware logging system...")
    
    # Configure logging
    configure_global_logging(
        log_dir="temporal",
        log_file="test_logging.log", 
        log_level="DEBUG",
        verbose=True,
        overwrite=True
    )
    
    # Get loggers
    logger = get_logger("test")
    config_logger = get_logger("config")
    training_logger = get_logger("training")
    
    # Test different log levels
    logger.debug("This is a DEBUG message (should only show on rank 0)")
    logger.info("This is an INFO message (should only show on rank 0)")
    logger.warning("This is a WARNING message (should show on all ranks)")
    logger.error("This is an ERROR message (should show on all ranks)")
    
    # Test different loggers
    config_logger.info("Config logger test message")
    training_logger.info("Training logger test message")
    
    print("✅ Rank-aware logging test completed")
    print("📄 Check temporal/test_logging.log for logged messages")

if __name__ == "__main__":
    test_rank_aware_logging()