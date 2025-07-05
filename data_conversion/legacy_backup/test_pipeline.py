#!/usr/bin/env python3
"""
Test Script for Modular Data Pipeline

Quick validation that all components work together correctly.
"""

import json
import logging
import tempfile
from pathlib import Path

from data_loader import SampleLoader
from data_splitter import DataSplitter
from teacher_selector import TeacherSelector
from processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_component_imports():
    """Test that all modules can be imported without errors."""
    logger.info("✅ All modules imported successfully")


def test_data_splitter():
    """Test DataSplitter with mock data."""
    samples = [{"id": i, "data": f"sample_{i}"} for i in range(10)]
    
    splitter = DataSplitter(val_ratio=0.2, seed=42)
    train, val = splitter.split(samples)
    
    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(samples)
    
    logger.info("✅ DataSplitter test passed")


def test_teacher_selector():
    """Test TeacherSelector with mock data."""
    mock_samples = []
    for i in range(5):
        mock_samples.append({
            "images": [f"image_{i}.jpg"],
            "objects": [
                {"bbox_2d": [10, 10, 50, 50], "desc": f"object_type_A/property_{i % 3}"},
                {"bbox_2d": [60, 60, 100, 100], "desc": f"object_type_B/property_{i % 2}"}
            ],
            "width": 200,
            "height": 200
        })
    
    # Mock hierarchy
    hierarchy = {
        "object_type_A": ["property_0", "property_1", "property_2"],
        "object_type_B": ["property_0", "property_1"]
    }
    
    selector = TeacherSelector(hierarchy, max_teachers=3, seed=42)
    teachers, indices = selector.select_teachers(mock_samples)
    
    assert len(teachers) <= 3
    assert len(indices) <= 3
    assert all(isinstance(idx, int) for idx in indices)
    
    logger.info("✅ TeacherSelector test passed")


def test_processor_config_validation():
    """Test DataProcessor configuration validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create minimal test directory structure
        input_dir = temp_path / "input"
        input_dir.mkdir()
        
        # Valid config
        valid_config = {
            "input_dir": str(input_dir),
            "output_dir": str(temp_path / "output"),
            "language": "chinese"
        }
        
        # Should not raise exception
        processor = DataProcessor(valid_config)
        assert processor.config["language"] == "chinese"
        
        # Invalid config - missing required field
        invalid_config = {
            "input_dir": str(input_dir),
            "language": "chinese"
            # Missing output_dir
        }
        
        try:
            DataProcessor(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        logger.info("✅ DataProcessor config validation test passed")


def run_all_tests():
    """Run all component tests."""
    logger.info("🧪 Starting component tests...")
    
    test_component_imports()
    test_data_splitter()
    test_teacher_selector()
    test_processor_config_validation()
    
    logger.info("🎉 All tests passed!")


if __name__ == "__main__":
    run_all_tests()