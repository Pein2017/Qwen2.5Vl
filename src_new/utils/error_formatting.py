#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized error message formatting utilities for Qwen2.5-VL project.

This module provides consistent error message construction patterns used
throughout the codebase, eliminating duplicate error formatting logic and
ensuring consistent, helpful error messages.

Key Features:
- Standardized error message formats with context
- File and directory error messages with available alternatives
- Validation error messages with expected vs actual comparisons
- Tensor shape and dimension error messages
- Contextual debugging information in error messages

Usage:
    from src_new.utils.error_formatting import ErrorMessageBuilder
    
    # File not found with context
    error_msg = ErrorMessageBuilder.build_file_not_found_error(
        missing_path="/path/to/missing.txt",
        available_files=["file1.txt", "file2.txt"],
        search_location="/path/to"
    )
    
    # Validation error with comparison
    error_msg = ErrorMessageBuilder.build_validation_error(
        item="tensor shape",
        expected=(224, 224, 3),
        actual=(256, 256, 3),
        context="image preprocessing"
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .logger_factory import get_module_logger

logger = get_module_logger(__name__)


class ErrorMessageBuilder:
    """
    Centralized error message builder for consistent error formatting.
    
    Provides standardized error message construction patterns used throughout
    the codebase, ensuring consistent and helpful error messages with context.
    """
    
    @staticmethod
    def build_file_not_found_error(
        missing_path: Union[str, Path],
        available_files: Optional[List[str]] = None,
        search_location: Optional[Union[str, Path]] = None,
        file_type: str = "file",
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Build detailed file not found error message with context.
        
        Args:
            missing_path: Path that was not found
            available_files: List of available files in the location
            search_location: Directory where the search was performed
            file_type: Type of file for error message ("file", "directory", "dataset", etc.)
            suggestions: Optional suggestions for resolution
            
        Returns:
            Formatted error message with context
        """
        missing_path = Path(missing_path)
        search_location = Path(search_location) if search_location else missing_path.parent
        
        error_lines = [
            f"{file_type.capitalize()} not found: {missing_path}"
        ]
        
        if available_files:
            error_lines.extend([
                f"",
                f"Available {file_type}s in {search_location}:",
                f"  {available_files}"
            ])
        
        if suggestions:
            error_lines.extend([
                f"",
                f"Suggestions:",
                *[f"  • {suggestion}" for suggestion in suggestions]
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_directory_structure_error(
        root_path: Union[str, Path],
        missing_items: List[str],
        available_files: Optional[List[str]] = None,
        available_dirs: Optional[List[str]] = None,
        expected_structure: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build directory structure validation error message.
        
        Args:
            root_path: Root directory that was validated
            missing_items: List of missing required items
            available_files: List of available files
            available_dirs: List of available directories
            expected_structure: Expected directory structure description
            
        Returns:
            Formatted error message with structure information
        """
        root_path = Path(root_path)
        
        error_lines = [
            f"Directory structure validation failed: {root_path}",
            f"",
            f"Missing required items:",
            *[f"  • {item}" for item in missing_items]
        ]
        
        if available_files:
            error_lines.extend([
                f"",
                f"Available files:",
                f"  {available_files}"
            ])
        
        if available_dirs:
            error_lines.extend([
                f"",
                f"Available directories:",
                f"  {available_dirs}"
            ])
        
        if expected_structure:
            error_lines.extend([
                f"",
                f"Expected structure:",
                *[f"  {key}: {value}" for key, value in expected_structure.items()]
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_validation_error(
        item: str,
        expected: Any,
        actual: Any,
        context: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Build validation error message with expected vs actual comparison.
        
        Args:
            item: Name of the item being validated
            expected: Expected value
            actual: Actual value
            context: Optional context description
            suggestions: Optional suggestions for resolution
            
        Returns:
            Formatted validation error message
        """
        error_lines = [
            f"Validation failed for {item}:"
        ]
        
        if context:
            error_lines.append(f"Context: {context}")
        
        error_lines.extend([
            f"  Expected: {expected}",
            f"  Actual: {actual}"
        ])
        
        if suggestions:
            error_lines.extend([
                f"",
                f"Suggestions:",
                *[f"  • {suggestion}" for suggestion in suggestions]
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_tensor_shape_error(
        tensor_name: str,
        expected_shape: tuple,
        actual_shape: tuple,
        operation: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Build tensor shape validation error message.
        
        Args:
            tensor_name: Name of the tensor
            expected_shape: Expected tensor shape
            actual_shape: Actual tensor shape
            operation: Operation being performed
            context: Additional context
            
        Returns:
            Formatted tensor shape error message
        """
        error_lines = [
            f"Tensor shape mismatch for {tensor_name}:"
        ]
        
        if operation:
            error_lines.append(f"Operation: {operation}")
        
        if context:
            error_lines.append(f"Context: {context}")
        
        error_lines.extend([
            f"  Expected shape: {expected_shape}",
            f"  Actual shape: {actual_shape}",
            f"  Shape difference: {len(actual_shape) - len(expected_shape)} dimensions"
        ])
        
        # Add helpful suggestions based on common shape issues
        suggestions = []
        if len(actual_shape) != len(expected_shape):
            if len(actual_shape) > len(expected_shape):
                suggestions.append("Consider using squeeze() to remove singleton dimensions")
            else:
                suggestions.append("Consider using unsqueeze() to add missing dimensions")
        
        if actual_shape and expected_shape:
            if actual_shape[0] != expected_shape[0]:
                suggestions.append("Check batch size dimension")
            if len(actual_shape) >= 2 and len(expected_shape) >= 2:
                if actual_shape[-2:] != expected_shape[-2:]:
                    suggestions.append("Check spatial dimensions (height, width)")
        
        if suggestions:
            error_lines.extend([
                f"",
                f"Suggestions:",
                *[f"  • {suggestion}" for suggestion in suggestions]
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_configuration_error(
        config_item: str,
        issue: str,
        current_value: Any = None,
        valid_options: Optional[List[Any]] = None,
        config_file: Optional[str] = None
    ) -> str:
        """
        Build configuration error message.
        
        Args:
            config_item: Name of the configuration item
            issue: Description of the issue
            current_value: Current value (if applicable)
            valid_options: List of valid options
            config_file: Configuration file path
            
        Returns:
            Formatted configuration error message
        """
        error_lines = [
            f"Configuration error for '{config_item}': {issue}"
        ]
        
        if current_value is not None:
            error_lines.append(f"Current value: {current_value}")
        
        if valid_options:
            error_lines.extend([
                f"Valid options: {valid_options}"
            ])
        
        if config_file:
            error_lines.extend([
                f"",
                f"Configuration file: {config_file}"
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_import_error(
        module_name: str,
        import_error: Exception,
        suggestions: Optional[List[str]] = None,
        required_for: Optional[str] = None
    ) -> str:
        """
        Build import error message with installation suggestions.
        
        Args:
            module_name: Name of the module that failed to import
            import_error: Original import exception
            suggestions: Installation or resolution suggestions
            required_for: What functionality requires this module
            
        Returns:
            Formatted import error message
        """
        error_lines = [
            f"Failed to import required module: {module_name}",
            f"Error: {import_error}"
        ]
        
        if required_for:
            error_lines.append(f"Required for: {required_for}")
        
        if suggestions:
            error_lines.extend([
                f"",
                f"Resolution suggestions:",
                *[f"  • {suggestion}" for suggestion in suggestions]
            ])
        else:
            # Default installation suggestion
            error_lines.extend([
                f"",
                f"Try installing with:",
                f"  pip install {module_name}"
            ])
        
        return "\n".join(error_lines)
    
    @staticmethod
    def build_context_error(
        operation: str,
        error: Exception,
        context: Dict[str, Any],
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Build error message with rich context information.
        
        Args:
            operation: Operation that failed
            error: Original exception
            context: Context dictionary with relevant information
            suggestions: Optional suggestions for resolution
            
        Returns:
            Formatted error message with context
        """
        error_lines = [
            f"Operation failed: {operation}",
            f"Error: {error}",
            f"",
            f"Context:"
        ]
        
        for key, value in context.items():
            error_lines.append(f"  {key}: {value}")
        
        if suggestions:
            error_lines.extend([
                f"",
                f"Suggestions:",
                *[f"  • {suggestion}" for suggestion in suggestions]
            ])
        
        return "\n".join(error_lines)


# Convenience functions for common error patterns
def format_file_not_found(path: Union[str, Path], available: List[str] = None) -> str:
    """Convenience function for file not found errors."""
    return ErrorMessageBuilder.build_file_not_found_error(path, available)


def format_validation_error(item: str, expected: Any, actual: Any) -> str:
    """Convenience function for validation errors."""
    return ErrorMessageBuilder.build_validation_error(item, expected, actual)


def format_tensor_error(name: str, expected_shape: tuple, actual_shape: tuple) -> str:
    """Convenience function for tensor shape errors."""
    return ErrorMessageBuilder.build_tensor_shape_error(name, expected_shape, actual_shape)


# Export public API
__all__ = [
    "ErrorMessageBuilder",
    "format_file_not_found",
    "format_validation_error", 
    "format_tensor_error",
]
