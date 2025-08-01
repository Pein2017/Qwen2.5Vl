"""
Test fixtures and utilities for src_new testing.

This module provides reusable test data, mock objects, and utility functions
used across different test modules.
"""

from .test_data import *
from .mock_objects import *
from .test_utils import *

__all__ = [
    # Test data
    "create_sample_config",
    "create_sample_jsonl_data", 
    "create_sample_conversation",
    "create_sample_batch",
    
    # Mock objects
    "MockTokenizer",
    "MockImageProcessor",
    "MockModel",
    "MockTrainer",
    
    # Test utilities
    "compare_configs",
    "validate_batch_structure",
    "assert_tensor_shapes",
    "create_temp_files",
]