#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the error formatting utilities module.

Tests the centralized error message formatting functionality with various
scenarios including file errors, validation errors, and tensor errors.
"""

import pytest
from pathlib import Path

from src_new.utils.error_formatting import (
    ErrorMessageBuilder,
    format_file_not_found,
    format_validation_error,
    format_tensor_error,
)


class TestErrorMessageBuilder:
    """Test cases for ErrorMessageBuilder class."""
    
    def test_build_file_not_found_error_basic(self):
        """Test basic file not found error message."""
        error_msg = ErrorMessageBuilder.build_file_not_found_error(
            missing_path="/path/to/missing.txt"
        )
        
        assert "File not found: /path/to/missing.txt" in error_msg
    
    def test_build_file_not_found_error_with_available_files(self):
        """Test file not found error with available files."""
        error_msg = ErrorMessageBuilder.build_file_not_found_error(
            missing_path="/path/to/missing.txt",
            available_files=["file1.txt", "file2.txt"],
            search_location="/path/to"
        )
        
        assert "File not found: /path/to/missing.txt" in error_msg
        assert "Available files in /path/to:" in error_msg
        assert "['file1.txt', 'file2.txt']" in error_msg
    
    def test_build_file_not_found_error_with_suggestions(self):
        """Test file not found error with suggestions."""
        error_msg = ErrorMessageBuilder.build_file_not_found_error(
            missing_path="/path/to/missing.txt",
            suggestions=["Check file permissions", "Verify file path"]
        )
        
        assert "File not found: /path/to/missing.txt" in error_msg
        assert "Suggestions:" in error_msg
        assert "• Check file permissions" in error_msg
        assert "• Verify file path" in error_msg
    
    def test_build_file_not_found_error_custom_type(self):
        """Test file not found error with custom file type."""
        error_msg = ErrorMessageBuilder.build_file_not_found_error(
            missing_path="/path/to/missing",
            file_type="dataset"
        )
        
        assert "Dataset not found: /path/to/missing" in error_msg
    
    def test_build_directory_structure_error(self):
        """Test directory structure error message."""
        error_msg = ErrorMessageBuilder.build_directory_structure_error(
            root_path="/data/root",
            missing_items=["train.jsonl", "images directory"],
            available_files=["config.yaml"],
            available_dirs=["logs"]
        )
        
        assert "Directory structure validation failed: /data/root" in error_msg
        assert "Missing required items:" in error_msg
        assert "• train.jsonl" in error_msg
        assert "• images directory" in error_msg
        assert "Available files:" in error_msg
        assert "['config.yaml']" in error_msg
        assert "Available directories:" in error_msg
        assert "['logs']" in error_msg
    
    def test_build_directory_structure_error_with_expected(self):
        """Test directory structure error with expected structure."""
        expected_structure = {
            "train.jsonl": "Training data file",
            "images/": "Directory containing image files"
        }
        
        error_msg = ErrorMessageBuilder.build_directory_structure_error(
            root_path="/data/root",
            missing_items=["train.jsonl"],
            expected_structure=expected_structure
        )
        
        assert "Expected structure:" in error_msg
        assert "train.jsonl: Training data file" in error_msg
        assert "images/: Directory containing image files" in error_msg
    
    def test_build_validation_error_basic(self):
        """Test basic validation error message."""
        error_msg = ErrorMessageBuilder.build_validation_error(
            item="tensor shape",
            expected=(224, 224, 3),
            actual=(256, 256, 3)
        )
        
        assert "Validation failed for tensor shape:" in error_msg
        assert "Expected: (224, 224, 3)" in error_msg
        assert "Actual: (256, 256, 3)" in error_msg
    
    def test_build_validation_error_with_context(self):
        """Test validation error with context."""
        error_msg = ErrorMessageBuilder.build_validation_error(
            item="batch size",
            expected=32,
            actual=16,
            context="image preprocessing pipeline"
        )
        
        assert "Validation failed for batch size:" in error_msg
        assert "Context: image preprocessing pipeline" in error_msg
        assert "Expected: 32" in error_msg
        assert "Actual: 16" in error_msg
    
    def test_build_validation_error_with_suggestions(self):
        """Test validation error with suggestions."""
        error_msg = ErrorMessageBuilder.build_validation_error(
            item="input format",
            expected="JSONL",
            actual="JSON",
            suggestions=["Convert file to JSONL format", "Check file extension"]
        )
        
        assert "Validation failed for input format:" in error_msg
        assert "Suggestions:" in error_msg
        assert "• Convert file to JSONL format" in error_msg
        assert "• Check file extension" in error_msg
    
    def test_build_tensor_shape_error_basic(self):
        """Test basic tensor shape error message."""
        error_msg = ErrorMessageBuilder.build_tensor_shape_error(
            tensor_name="pixel_values",
            expected_shape=(1024, 768),
            actual_shape=(512, 768)
        )
        
        assert "Tensor shape mismatch for pixel_values:" in error_msg
        assert "Expected shape: (1024, 768)" in error_msg
        assert "Actual shape: (512, 768)" in error_msg
        assert "Shape difference: 0 dimensions" in error_msg
    
    def test_build_tensor_shape_error_with_operation(self):
        """Test tensor shape error with operation context."""
        error_msg = ErrorMessageBuilder.build_tensor_shape_error(
            tensor_name="input_tensor",
            expected_shape=(32, 224, 224, 3),
            actual_shape=(16, 224, 224, 3),
            operation="batch processing",
            context="image preprocessing"
        )
        
        assert "Tensor shape mismatch for input_tensor:" in error_msg
        assert "Operation: batch processing" in error_msg
        assert "Context: image preprocessing" in error_msg
    
    def test_build_tensor_shape_error_dimension_suggestions(self):
        """Test tensor shape error with automatic suggestions."""
        # Test case where actual has more dimensions
        error_msg = ErrorMessageBuilder.build_tensor_shape_error(
            tensor_name="test_tensor",
            expected_shape=(224, 224),
            actual_shape=(1, 224, 224)
        )
        
        assert "Consider using squeeze() to remove singleton dimensions" in error_msg
        
        # Test case where actual has fewer dimensions
        error_msg = ErrorMessageBuilder.build_tensor_shape_error(
            tensor_name="test_tensor",
            expected_shape=(1, 224, 224),
            actual_shape=(224, 224)
        )
        
        assert "Consider using unsqueeze() to add missing dimensions" in error_msg
    
    def test_build_configuration_error_basic(self):
        """Test basic configuration error message."""
        error_msg = ErrorMessageBuilder.build_configuration_error(
            config_item="batch_size",
            issue="must be positive integer"
        )
        
        assert "Configuration error for 'batch_size': must be positive integer" in error_msg
    
    def test_build_configuration_error_with_details(self):
        """Test configuration error with full details."""
        error_msg = ErrorMessageBuilder.build_configuration_error(
            config_item="model_type",
            issue="invalid value",
            current_value="unknown_model",
            valid_options=["qwen2.5-vl", "llama", "bert"],
            config_file="/path/to/config.yaml"
        )
        
        assert "Configuration error for 'model_type': invalid value" in error_msg
        assert "Current value: unknown_model" in error_msg
        assert "Valid options: ['qwen2.5-vl', 'llama', 'bert']" in error_msg
        assert "Configuration file: /path/to/config.yaml" in error_msg
    
    def test_build_import_error_basic(self):
        """Test basic import error message."""
        import_error = ImportError("No module named 'missing_module'")
        
        error_msg = ErrorMessageBuilder.build_import_error(
            module_name="missing_module",
            import_error=import_error
        )
        
        assert "Failed to import required module: missing_module" in error_msg
        assert "Error: No module named 'missing_module'" in error_msg
        assert "pip install missing_module" in error_msg
    
    def test_build_import_error_with_suggestions(self):
        """Test import error with custom suggestions."""
        import_error = ImportError("No module named 'torch'")
        
        error_msg = ErrorMessageBuilder.build_import_error(
            module_name="torch",
            import_error=import_error,
            suggestions=["pip install torch", "conda install pytorch"],
            required_for="tensor operations"
        )
        
        assert "Failed to import required module: torch" in error_msg
        assert "Required for: tensor operations" in error_msg
        assert "Resolution suggestions:" in error_msg
        assert "• pip install torch" in error_msg
        assert "• conda install pytorch" in error_msg
    
    def test_build_context_error(self):
        """Test context error message."""
        context = {
            "operation": "model_loading",
            "model_path": "/path/to/model",
            "device": "cuda:0"
        }
        error = RuntimeError("CUDA out of memory")
        
        error_msg = ErrorMessageBuilder.build_context_error(
            operation="load model",
            error=error,
            context=context,
            suggestions=["Reduce batch size", "Use CPU instead"]
        )
        
        assert "Operation failed: load model" in error_msg
        assert "Error: CUDA out of memory" in error_msg
        assert "Context:" in error_msg
        assert "operation: model_loading" in error_msg
        assert "model_path: /path/to/model" in error_msg
        assert "device: cuda:0" in error_msg
        assert "Suggestions:" in error_msg
        assert "• Reduce batch size" in error_msg
        assert "• Use CPU instead" in error_msg


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_format_file_not_found(self):
        """Test format_file_not_found convenience function."""
        error_msg = format_file_not_found(
            path="/missing/file.txt",
            available=["file1.txt", "file2.txt"]
        )
        
        assert "File not found: /missing/file.txt" in error_msg
        assert "['file1.txt', 'file2.txt']" in error_msg
    
    def test_format_validation_error(self):
        """Test format_validation_error convenience function."""
        error_msg = format_validation_error(
            item="input size",
            expected=1024,
            actual=512
        )
        
        assert "Validation failed for input size:" in error_msg
        assert "Expected: 1024" in error_msg
        assert "Actual: 512" in error_msg
    
    def test_format_tensor_error(self):
        """Test format_tensor_error convenience function."""
        error_msg = format_tensor_error(
            name="input_tensor",
            expected_shape=(32, 224, 224),
            actual_shape=(16, 224, 224)
        )
        
        assert "Tensor shape mismatch for input_tensor:" in error_msg
        assert "Expected shape: (32, 224, 224)" in error_msg
        assert "Actual shape: (16, 224, 224)" in error_msg
