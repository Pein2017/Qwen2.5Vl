"""
Unit tests for the data resolver module.

Tests the centralized data path resolution functionality with various scenarios
including valid structures, missing files, and error conditions.
"""

from pathlib import Path

import pytest

from src_new.utils.data_resolver import (
    DataResolver,
    DatasetPaths,
    resolve_dataset_paths,
)


class TestDataResolver:
    """Test cases for DataResolver class."""

    def test_resolve_valid_dataset_structure(self, tmp_path):
        """Test successful resolution of valid dataset structure."""
        # Create valid dataset structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "test.jpg").write_text("fake image")

        # Test resolution
        paths = DataResolver.resolve_dataset_paths(str(tmp_path))

        # Verify all paths are correct
        assert paths.data_root == tmp_path.resolve()
        assert paths.train_data_path == tmp_path / "train.jsonl"
        assert paths.val_data_path == tmp_path / "val.jsonl"
        assert paths.teacher_pool_file == tmp_path / "teacher_pool.jsonl"
        assert paths.images_dir == tmp_path / "images"

        # Verify all paths exist
        assert paths.train_data_path.exists()
        assert paths.val_data_path.exists()
        assert paths.teacher_pool_file.exists()
        assert paths.images_dir.exists()
        assert paths.images_dir.is_dir()

    def test_resolve_with_path_object(self, tmp_path):
        """Test resolution works with Path object input."""
        # Create valid structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        # Test with Path object
        paths = DataResolver.resolve_dataset_paths(tmp_path)
        assert paths.data_root == tmp_path.resolve()

    def test_empty_data_root_raises_error(self):
        """Test that empty data_root raises ValueError."""
        with pytest.raises(ValueError, match="data_root cannot be empty or None"):
            DataResolver.resolve_dataset_paths("")

        with pytest.raises(ValueError, match="data_root cannot be empty or None"):
            DataResolver.resolve_dataset_paths(None)

    def test_invalid_data_root_type_raises_error(self):
        """Test that invalid data_root type raises TypeError."""
        with pytest.raises(TypeError, match="data_root must be str or Path"):
            DataResolver.resolve_dataset_paths(123)

        with pytest.raises(TypeError, match="data_root must be str or Path"):
            DataResolver.resolve_dataset_paths(["/some/path"])

    def test_nonexistent_data_root_raises_error(self):
        """Test that nonexistent data_root raises FileNotFoundError."""
        with pytest.raises(
            FileNotFoundError, match="Data root directory does not exist"
        ):
            DataResolver.resolve_dataset_paths("/nonexistent/path")

    def test_data_root_not_directory_raises_error(self, tmp_path):
        """Test that data_root pointing to file raises ValueError."""
        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Data root path is not a directory"):
            DataResolver.resolve_dataset_paths(str(file_path))

    def test_missing_train_file_raises_error(self, tmp_path):
        """Test that missing train.jsonl raises FileNotFoundError."""
        # Create incomplete structure (missing train.jsonl)
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        with pytest.raises(
            FileNotFoundError, match="Training data file does not exist"
        ):
            DataResolver.resolve_dataset_paths(str(tmp_path))

    def test_missing_val_file_raises_error(self, tmp_path):
        """Test that missing val.jsonl raises FileNotFoundError."""
        # Create incomplete structure (missing val.jsonl)
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        with pytest.raises(
            FileNotFoundError, match="Validation data file does not exist"
        ):
            DataResolver.resolve_dataset_paths(str(tmp_path))

    def test_missing_teacher_pool_raises_error(self, tmp_path):
        """Test that missing teacher_pool.jsonl raises FileNotFoundError."""
        # Create incomplete structure (missing teacher_pool.jsonl)
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        with pytest.raises(FileNotFoundError, match="Teacher pool file does not exist"):
            DataResolver.resolve_dataset_paths(str(tmp_path))

    def test_missing_images_dir_raises_error(self, tmp_path):
        """Test that missing images directory raises FileNotFoundError."""
        # Create incomplete structure (missing images dir)
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')

        with pytest.raises(FileNotFoundError, match="Images directory does not exist"):
            DataResolver.resolve_dataset_paths(str(tmp_path))

    def test_images_not_directory_raises_error(self, tmp_path):
        """Test that images path pointing to file raises FileNotFoundError (wrapped)."""
        # Create structure with images as file instead of directory
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").write_text("not a directory")

        with pytest.raises(FileNotFoundError, match="Images path is not a directory"):
            DataResolver.resolve_dataset_paths(str(tmp_path))

    def test_enhanced_error_message_shows_available_files(self, tmp_path):
        """Test that error messages include available files for debugging."""
        # Create partial structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "other.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "some_dir").mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            DataResolver.resolve_dataset_paths(str(tmp_path))

        error_msg = str(exc_info.value)
        assert "Available .jsonl files" in error_msg
        assert "train.jsonl" in error_msg
        assert "other.jsonl" in error_msg
        assert "Available directories" in error_msg
        assert "some_dir" in error_msg
        assert "Expected structure" in error_msg

    def test_validate_dataset_structure_valid(self, tmp_path):
        """Test validate_dataset_structure returns True for valid structure."""
        # Create valid structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        assert DataResolver.validate_dataset_structure(str(tmp_path)) is True

    def test_validate_dataset_structure_invalid(self, tmp_path):
        """Test validate_dataset_structure returns False for invalid structure."""
        # Create incomplete structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')

        assert DataResolver.validate_dataset_structure(str(tmp_path)) is False

    def test_get_missing_files_complete_structure(self, tmp_path):
        """Test get_missing_files returns empty list for complete structure."""
        # Create valid structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        missing = DataResolver.get_missing_files(str(tmp_path))
        assert missing == []

    def test_get_missing_files_partial_structure(self, tmp_path):
        """Test get_missing_files returns correct missing files."""
        # Create partial structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").write_text("not a directory")  # Wrong type

        missing = DataResolver.get_missing_files(str(tmp_path))
        assert "val.jsonl (file)" in missing
        assert "teacher_pool.jsonl (file)" in missing
        assert "images (not a directory)" in missing
        assert len(missing) == 3

    def test_get_missing_files_invalid_data_root(self):
        """Test get_missing_files handles invalid data_root gracefully."""
        missing = DataResolver.get_missing_files(None)
        assert "Invalid data_root parameter" in missing

        missing = DataResolver.get_missing_files("/nonexistent")
        assert "Data root directory" in missing[0]


class TestDatasetPaths:
    """Test cases for DatasetPaths dataclass."""

    def test_dataset_paths_immutable(self, tmp_path):
        """Test that DatasetPaths is immutable (frozen)."""
        # Create valid structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        paths = DataResolver.resolve_dataset_paths(str(tmp_path))

        # Test immutability
        with pytest.raises(AttributeError):
            paths.data_root = Path("/other/path")


class TestConvenienceFunction:
    """Test cases for convenience functions."""

    def test_resolve_dataset_paths_function(self, tmp_path):
        """Test that convenience function works correctly."""
        # Create valid structure
        (tmp_path / "train.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "val.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "teacher_pool.jsonl").write_text('{"test": "data"}\n')
        (tmp_path / "images").mkdir()

        # Test convenience function
        paths = resolve_dataset_paths(str(tmp_path))
        assert isinstance(paths, DatasetPaths)
        assert paths.data_root == tmp_path.resolve()
