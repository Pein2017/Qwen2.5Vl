"""
Test Suite for Qwen2.5-VL Simplified Training Pipeline

This test package provides comprehensive validation of the new src_new/ implementation,
ensuring all functionality works correctly and maintains backward compatibility.

Test Categories:
- Unit Tests: Individual module validation
- Integration Tests: End-to-end pipeline validation
- Compatibility Tests: Backward compatibility with existing configs
- Performance Tests: Speed and memory benchmarking
- Regression Tests: Validation against original implementation
"""

import sys
import os
from pathlib import Path

# Add src_new to Python path for testing
test_dir = Path(__file__).parent
src_new_dir = test_dir.parent
project_root = src_new_dir.parent

if str(src_new_dir) not in sys.path:
    sys.path.insert(0, str(src_new_dir))

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "fixtures"
TEST_CONFIG_DIR = project_root / "configs"
TEST_OUTPUT_DIR = project_root / "test_outputs"

# Ensure test output directory exists
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"
__all__ = ["TEST_DATA_DIR", "TEST_CONFIG_DIR", "TEST_OUTPUT_DIR"]