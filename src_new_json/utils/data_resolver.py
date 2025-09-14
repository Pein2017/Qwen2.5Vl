"""
Centralized data path resolution module for Qwen2.5-VL training and inference.

This module provides automatic discovery and validation of dataset files based on
a standardized directory structure, requiring only a single data_root parameter.

Expected directory structure:
    data_root/
    ├── train.jsonl
    ├── val.jsonl
    ├── teacher_pool.jsonl
    └── images/
        └── [image files]

Follows the project's fail-fast development approach with strict validation.
"""

from dataclasses import dataclass
from pathlib import Path

from .rank_aware_logging import get_rank_aware_logger
from .validation import DirectoryValidationError, PathValidator


logger = get_rank_aware_logger("data_resolver")


@dataclass(frozen=True)
class DatasetPaths:
    """Immutable container for resolved dataset file paths."""

    data_root: Path
    train_data_path: Path
    val_data_path: Path
    teacher_pool_file: Path
    images_dir: Path

    def __post_init__(self):
        """Validate all paths exist after initialization (using centralized validation)."""
        # Note: Validation is now handled by PathValidator.validate_directory_structure
        # before DatasetPaths creation, so this is just a safety check
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )
        if not self.images_dir.is_dir():
            raise ValueError(f"Images path is not a directory: {self.images_dir}")


class DataResolver:
    """
    Centralized data path resolver for automatic dataset file discovery.

    Follows fail-fast principles with strict validation and clear error messages.
    """

    # Standard file names expected in data_root
    TRAIN_FILE = "train.jsonl"
    VAL_FILE = "val.jsonl"
    TEACHER_POOL_FILE = "teacher_pool.jsonl"
    IMAGES_DIR = "images"

    @classmethod
    def resolve_dataset_paths(cls, data_root: str) -> DatasetPaths:
        """
        Resolve and validate all dataset file paths from data_root.

        Args:
            data_root: Root directory containing dataset files

        Returns:
            DatasetPaths object with resolved paths (relative preserved if input is relative)

        Raises:
            ValueError: If data_root is invalid or empty
            FileNotFoundError: If any required files are missing
            TypeError: If data_root is not a string or Path-like object
        """
        # Validate input parameter and directory structure using centralized validation
        try:
            data_root_path, _ = PathValidator.validate_directory_structure(
                root_path=data_root,
                required_files=[cls.TRAIN_FILE, cls.VAL_FILE, cls.TEACHER_POOL_FILE],
                required_dirs=[cls.IMAGES_DIR],
            )
        except DirectoryValidationError as e:
            # Re-raise with enhanced error message for debugging
            raise FileNotFoundError(str(e)) from e

        # Resolve all required file paths
        train_path = data_root_path / cls.TRAIN_FILE
        val_path = data_root_path / cls.VAL_FILE
        teacher_pool_path = data_root_path / cls.TEACHER_POOL_FILE
        images_path = data_root_path / cls.IMAGES_DIR

        # Log discovered paths for debugging
        logger.debug(f"🔍 Data resolver scanning: {data_root_path}")
        logger.debug(f"   Expected train file: {train_path}")
        logger.debug(f"   Expected val file: {val_path}")
        logger.debug(f"   Expected teacher pool: {teacher_pool_path}")
        logger.debug(f"   Expected images dir: {images_path}")

        # Create DatasetPaths object (validation happens in __post_init__)
        try:
            dataset_paths = DatasetPaths(
                data_root=data_root_path,
                train_data_path=train_path,
                val_data_path=val_path,
                teacher_pool_file=teacher_pool_path,
                images_dir=images_path,
            )
        except (FileNotFoundError, ValueError) as e:
            # Enhance error message with available files for debugging
            available_files = list(data_root_path.glob("*.jsonl"))
            available_dirs = [p for p in data_root_path.iterdir() if p.is_dir()]

            error_msg = (
                f"Dataset validation failed: {e}\n"
                f"Available .jsonl files in {data_root_path}:\n"
                f"  {[f.name for f in available_files]}\n"
                f"Available directories:\n"
                f"  {[d.name for d in available_dirs]}\n"
                f"Expected structure:\n"
                f"  {data_root_path}/\n"
                f"  ├── {cls.TRAIN_FILE}\n"
                f"  ├── {cls.VAL_FILE}\n"
                f"  ├── {cls.TEACHER_POOL_FILE}\n"
                f"  └── {cls.IMAGES_DIR}/\n"
            )
            raise FileNotFoundError(error_msg) from e

        # Log successful resolution
        logger.info(f"✅ Dataset paths resolved successfully from: {data_root_path}")
        logger.info(f"📁 Train: {dataset_paths.train_data_path}")
        logger.info(f"📁 Val: {dataset_paths.val_data_path}")
        logger.info(f"📁 Teacher pool: {dataset_paths.teacher_pool_file}")
        logger.info(f"📁 Images: {dataset_paths.images_dir}")

        return dataset_paths

    @classmethod
    def validate_dataset_structure(cls, data_root: str) -> bool:
        """
        Validate dataset structure without raising exceptions.

        Args:
            data_root: Root directory to validate

        Returns:
            True if structure is valid, False otherwise
        """
        try:
            cls.resolve_dataset_paths(data_root)
            return True
        except (ValueError, FileNotFoundError, TypeError):
            return False

    @classmethod
    def get_missing_files(cls, data_root: str) -> list[str]:
        """
        Get list of missing required files in data_root.

        Args:
            data_root: Root directory to check

        Returns:
            List of missing file names (empty if all files exist)
        """
        return PathValidator.get_missing_items(
            root_path=data_root,
            required_files=[cls.TRAIN_FILE, cls.VAL_FILE, cls.TEACHER_POOL_FILE],
            required_dirs=[cls.IMAGES_DIR],
        )


# Convenience function for backward compatibility
def resolve_dataset_paths(data_root: str) -> DatasetPaths:
    """Convenience function that delegates to DataResolver.resolve_dataset_paths."""
    return DataResolver.resolve_dataset_paths(data_root)


# Export public API
__all__ = ["DatasetPaths", "DataResolver", "resolve_dataset_paths"]
