#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the validation utilities module.

Tests the centralized validation functionality with various scenarios
including path validation, directory structure validation, and error handling.
"""

import pytest

from src_new.utils.validation import (
    DirectoryValidationError,
    PathValidationError,
    PathValidator,
    ValidationError,
    validate_dataset_structure,
    validate_directory,
    validate_file,
)


class TestPathValidator:
    """Test cases for PathValidator class."""

    def test_validate_path_exists_valid_file(self, tmp_path):
        """Test validating an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = PathValidator.validate_path_exists(test_file)
        assert result == test_file.resolve()

    def test_validate_path_exists_valid_directory(self, tmp_path):
        """Test validating an existing directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = PathValidator.validate_path_exists(test_dir)
        assert result == test_dir.resolve()

    def test_validate_path_exists_missing_path(self, tmp_path):
        """Test validating a non-existent path."""
        missing_path = tmp_path / "missing.txt"

        with pytest.raises(PathValidationError, match="Path does not exist"):
            PathValidator.validate_path_exists(missing_path)

    def test_validate_path_exists_allow_missing(self, tmp_path):
        """Test validating with allow_missing=True."""
        missing_path = tmp_path / "missing.txt"

        result = PathValidator.validate_path_exists(missing_path, allow_missing=True)
        assert result == missing_path.resolve()

    def test_validate_path_exists_empty_path(self):
        """Test validating empty path."""
        with pytest.raises(ValueError, match="Path path cannot be empty or None"):
            PathValidator.validate_path_exists("")

        with pytest.raises(ValueError, match="Path path cannot be empty or None"):
            PathValidator.validate_path_exists(None)

    def test_validate_path_exists_invalid_type(self):
        """Test validating invalid path type."""
        with pytest.raises(ValueError, match="Path path must be str or Path"):
            PathValidator.validate_path_exists(123)

    def test_validate_file_exists(self, tmp_path):
        """Test file-specific validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = PathValidator.validate_file_exists(test_file)
        assert result == test_file.resolve()

    def test_validate_file_exists_directory(self, tmp_path):
        """Test file validation on directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(PathValidationError, match="Path exists but is not a file"):
            PathValidator.validate_file_exists(test_dir)

    def test_validate_directory_exists(self, tmp_path):
        """Test directory-specific validation."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = PathValidator.validate_directory_exists(test_dir)
        assert result == test_dir.resolve()

    def test_validate_directory_exists_file(self, tmp_path):
        """Test directory validation on file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(
            PathValidationError, match="Path exists but is not a directory"
        ):
            PathValidator.validate_directory_exists(test_file)

    def test_validate_paths_exist_multiple(self, tmp_path):
        """Test validating multiple paths."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        result = PathValidator.validate_paths_exist([file1, file2])
        assert len(result) == 2
        assert result[0] == file1.resolve()
        assert result[1] == file2.resolve()

    def test_validate_paths_exist_empty_list(self):
        """Test validating empty path list."""
        result = PathValidator.validate_paths_exist([])
        assert result == []

    def test_validate_paths_exist_one_missing(self, tmp_path):
        """Test validating paths with one missing."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "missing.txt"
        file1.write_text("content1")

        with pytest.raises(PathValidationError, match="Failed to validate path 1"):
            PathValidator.validate_paths_exist([file1, file2])

    def test_validate_directory_structure_valid(self, tmp_path):
        """Test validating valid directory structure."""
        # Create required files and directories
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").mkdir()

        root, status = PathValidator.validate_directory_structure(
            root_path=tmp_path,
            required_files=["train.jsonl", "val.jsonl"],
            required_dirs=["images"],
        )

        assert root == tmp_path.resolve()
        assert status["file:train.jsonl"] is True
        assert status["file:val.jsonl"] is True
        assert status["dir:images"] is True

    def test_validate_directory_structure_missing_file(self, tmp_path):
        """Test validating directory structure with missing file."""
        # Create only some required files
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").mkdir()

        with pytest.raises(
            DirectoryValidationError, match="Directory structure validation failed"
        ):
            PathValidator.validate_directory_structure(
                root_path=tmp_path,
                required_files=["train.jsonl", "val.jsonl"],
                required_dirs=["images"],
            )

    def test_validate_directory_structure_missing_directory(self, tmp_path):
        """Test validating directory structure with missing directory."""
        # Create files but not directory
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}')

        with pytest.raises(
            DirectoryValidationError, match="Directory structure validation failed"
        ):
            PathValidator.validate_directory_structure(
                root_path=tmp_path,
                required_files=["train.jsonl", "val.jsonl"],
                required_dirs=["images"],
            )

    def test_validate_directory_structure_with_optional(self, tmp_path):
        """Test validating directory structure with optional items."""
        # Create required items
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").mkdir()
        # Create one optional item
        (tmp_path / "config.yaml").write_text("config: value")

        root, status = PathValidator.validate_directory_structure(
            root_path=tmp_path,
            required_files=["train.jsonl"],
            required_dirs=["images"],
            optional_files=["config.yaml", "missing.txt"],
        )

        assert root == tmp_path.resolve()
        assert status["file:train.jsonl"] is True
        assert status["dir:images"] is True
        assert status["optional_file:config.yaml"] is True
        assert status["optional_file:missing.txt"] is False

    def test_get_missing_items_complete(self, tmp_path):
        """Test getting missing items from complete structure."""
        # Create complete structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").mkdir()

        missing = PathValidator.get_missing_items(
            root_path=tmp_path,
            required_files=["train.jsonl", "val.jsonl"],
            required_dirs=["images"],
        )

        assert missing == []

    def test_get_missing_items_partial(self, tmp_path):
        """Test getting missing items from partial structure."""
        # Create only some items
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").write_text("not a directory")  # Wrong type

        missing = PathValidator.get_missing_items(
            root_path=tmp_path,
            required_files=["train.jsonl", "val.jsonl"],
            required_dirs=["images"],
        )

        assert "val.jsonl (file)" in missing
        assert "images (not a directory)" in missing
        assert len(missing) == 2

    def test_get_missing_items_invalid_root(self):
        """Test getting missing items with invalid root."""
        missing = PathValidator.get_missing_items(
            root_path="/nonexistent/path",
            required_files=["train.jsonl"],
            required_dirs=["images"],
        )

        assert len(missing) == 1
        assert "Root directory" in missing[0]


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_validate_file(self, tmp_path):
        """Test validate_file convenience function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validate_file(test_file)
        assert result == test_file.resolve()

    def test_validate_directory(self, tmp_path):
        """Test validate_directory convenience function."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = validate_directory(test_dir)
        assert result == test_dir.resolve()

    def test_validate_dataset_structure(self, tmp_path):
        """Test validate_dataset_structure convenience function."""
        # Create standard dataset structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}')
        (tmp_path / "images").mkdir()

        result = validate_dataset_structure(tmp_path)
        assert result == tmp_path.resolve()

    def test_validate_dataset_structure_custom_files(self, tmp_path):
        """Test validate_dataset_structure with custom file names."""
        # Create custom structure
        (tmp_path / "training.jsonl").write_text('{"test": "data"}')
        (tmp_path / "validation.jsonl").write_text('{"test": "data"}')
        (tmp_path / "pics").mkdir()

        result = validate_dataset_structure(
            data_root=tmp_path,
            train_file="training.jsonl",
            val_file="validation.jsonl",
            images_dir="pics",
        )
        assert result == tmp_path.resolve()


class TestExceptionClasses:
    """Test cases for custom exception classes."""

    def test_validation_error_basic(self):
        """Test basic ValidationError."""
        error = ValidationError("Test error")
        assert str(error) == "Test error"
        assert error.context == {}

    def test_validation_error_with_context(self):
        """Test ValidationError with context."""
        context = {"key": "value", "number": 42}
        error = ValidationError("Test error", context)
        assert str(error) == "Test error"
        assert error.context == context

    def test_path_validation_error(self):
        """Test PathValidationError inheritance."""
        error = PathValidationError("Path error")
        assert isinstance(error, ValidationError)
        assert str(error) == "Path error"

    def test_directory_validation_error(self):
        """Test DirectoryValidationError inheritance."""
        error = DirectoryValidationError("Directory error")
        assert isinstance(error, ValidationError)
        assert str(error) == "Directory error"
