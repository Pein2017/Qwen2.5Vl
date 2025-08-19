#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized validation utilities for Qwen2.5-VL project.

This module provides consistent validation patterns for paths, files, directories,
and other common validation needs across the codebase. It eliminates duplicate
validation logic and ensures consistent error handling.

Key Features:
- Centralized path existence and type validation
- Directory structure validation
- Consistent error messages with context
- Fail-fast validation following project principles
- Type-safe validation with clear return types

Usage:
    from src_new.utils.validation import PathValidator, ValidationError

    # Validate single path
    validated_path = PathValidator.validate_path_exists("/path/to/file")

    # Validate directory structure
    PathValidator.validate_directory_structure(
        root_path="/data/root",
        required_files=["train.jsonl", "val.jsonl"],
        required_dirs=["images"]
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("validation")


# --- Path alias & normalization helpers ---
def _get_src_new_dir() -> Path:
    """Return absolute path to the src_new directory."""
    # This file is located at src_new/utils/validation.py
    return Path(__file__).resolve().parents[1]  # .../src_new


def expand_path_aliases(path: Union[str, Path]) -> str:
    """Expand supported path aliases in the given path string.

    Supported aliases:
    - "@src_new" or "@src_new/" → absolute path to the repository's src_new directory
    """
    if isinstance(path, Path):
        path_str = str(path)
    else:
        path_str = path

    if not isinstance(path_str, str):
        return path  # type: ignore[return-value]

    if path_str.startswith("@src_new/") or path_str == "@src_new":
        src_new_dir = _get_src_new_dir()
        remainder = path_str[len("@src_new/") :] if path_str != "@src_new" else ""
        expanded = str(src_new_dir / remainder)
        logger.debug(f"Expanded '@src_new' alias: {path_str} -> {expanded}")
        return expanded

    return path_str


def normalize_path_input(
    path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Expand aliases and convert to a Path, preserving relativity.

    If the provided path is relative, it is left relative to the current working
    directory by default, or joined with base_dir if provided. No filesystem
    resolution is performed here.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError(f"Path must be str or Path, got {type(path)}")

    expanded = expand_path_aliases(path)
    candidate = Path(expanded)

    if candidate.is_absolute():
        return candidate

    base = Path(base_dir) if base_dir is not None else None
    if base is None:
        return candidate
    joined = base / candidate
    return joined


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class PathValidationError(ValidationError):
    """Specific exception for path validation errors."""

    pass


class DirectoryValidationError(ValidationError):
    """Specific exception for directory validation errors."""

    pass


class PathValidator:
    """
    Centralized path validation utilities.

    Provides consistent path validation patterns used throughout the codebase,
    eliminating duplicate validation logic and ensuring fail-fast behavior.
    """

    @staticmethod
    def validate_path_exists(
        path: Union[str, Path], path_type: str = "path", allow_missing: bool = False
    ) -> Path:
        """
        Validate that a path exists and return it as a Path object.

        Args:
            path: Path to validate (string or Path object)
            path_type: Type description for error messages ("file", "directory", "path")
            allow_missing: If True, don't raise error for missing paths

        Returns:
            Validated Path object (preserving relative paths if provided)

        Raises:
            PathValidationError: If path is invalid or doesn't exist
            ValueError: If path is empty or None
        """
        if not path:
            raise ValueError(f"{path_type.capitalize()} path cannot be empty or None")

        if not isinstance(path, (str, Path)):
            raise ValueError(
                f"{path_type.capitalize()} path must be str or Path, got {type(path)}"
            )

        expanded = expand_path_aliases(path)
        path_obj = Path(expanded)

        if not allow_missing and not path_obj.exists():
            raise PathValidationError(
                f"{path_type.capitalize()} does not exist: {path_obj}",
                context={"path": str(path_obj), "path_type": path_type},
            )

        logger.debug(f"✅ Validated {path_type}: {path_obj}")
        return path_obj

    @staticmethod
    def validate_file_exists(
        path: Union[str, Path], allow_missing: bool = False
    ) -> Path:
        """
        Validate that a file exists.

        Args:
            path: File path to validate
            allow_missing: If True, don't raise error for missing files

        Returns:
            Validated Path object

        Raises:
            PathValidationError: If file doesn't exist or is not a file
        """
        validated_path = PathValidator.validate_path_exists(path, "file", allow_missing)

        if (
            not allow_missing
            and validated_path.exists()
            and not validated_path.is_file()
        ):
            raise PathValidationError(
                f"Path exists but is not a file: {validated_path}",
                context={
                    "path": str(validated_path),
                    "is_dir": validated_path.is_dir(),
                },
            )

        return validated_path

    @staticmethod
    def validate_directory_exists(
        path: Union[str, Path], allow_missing: bool = False
    ) -> Path:
        """
        Validate that a directory exists.

        Args:
            path: Directory path to validate
            allow_missing: If True, don't raise error for missing directories

        Returns:
            Validated Path object

        Raises:
            PathValidationError: If directory doesn't exist or is not a directory
        """
        validated_path = PathValidator.validate_path_exists(
            path, "directory", allow_missing
        )

        if (
            not allow_missing
            and validated_path.exists()
            and not validated_path.is_dir()
        ):
            raise PathValidationError(
                f"Path exists but is not a directory: {validated_path}",
                context={
                    "path": str(validated_path),
                    "is_file": validated_path.is_file(),
                },
            )

        return validated_path

    @staticmethod
    def validate_paths_exist(
        paths: List[Union[str, Path]],
        path_type: str = "path",
        allow_missing: bool = False,
    ) -> List[Path]:
        """
        Validate multiple paths exist.

        Args:
            paths: List of paths to validate
            path_type: Type description for error messages
            allow_missing: If True, don't raise error for missing paths

        Returns:
            List of validated Path objects

        Raises:
            PathValidationError: If any path is invalid
        """
        if not paths:
            return []

        validated_paths = []
        for i, path in enumerate(paths):
            try:
                validated = PathValidator.validate_path_exists(
                    path, path_type, allow_missing
                )
                validated_paths.append(validated)
            except (ValueError, PathValidationError) as e:
                raise PathValidationError(
                    f"Failed to validate {path_type} {i}: {e}",
                    context={"index": i, "path": str(path), "original_error": str(e)},
                ) from e

        logger.debug(f"✅ Validated {len(validated_paths)} {path_type}s")
        return validated_paths

    @staticmethod
    def validate_directory_structure(
        root_path: Union[str, Path],
        required_files: Optional[List[str]] = None,
        required_dirs: Optional[List[str]] = None,
        optional_files: Optional[List[str]] = None,
        optional_dirs: Optional[List[str]] = None,
    ) -> Tuple[Path, Dict[str, bool]]:
        """
        Validate directory structure contains required files and directories.

        Args:
            root_path: Root directory to validate
            required_files: List of required file names
            required_dirs: List of required directory names
            optional_files: List of optional file names (for reporting)
            optional_dirs: List of optional directory names (for reporting)

        Returns:
            Tuple of (validated_root_path, status_dict)

        Raises:
            DirectoryValidationError: If structure is invalid
        """
        # Expand aliases for the root
        expanded_root = expand_path_aliases(root_path)
        root = PathValidator.validate_directory_exists(expanded_root)

        required_files = required_files or []
        required_dirs = required_dirs or []
        optional_files = optional_files or []
        optional_dirs = optional_dirs or []

        status = {}
        missing_required = []

        # Check required files
        for filename in required_files:
            file_path = root / filename
            exists = file_path.exists() and file_path.is_file()
            status[f"file:{filename}"] = exists
            if not exists:
                missing_required.append(f"file: {filename}")

        # Check required directories
        for dirname in required_dirs:
            dir_path = root / dirname
            exists = dir_path.exists() and dir_path.is_dir()
            status[f"dir:{dirname}"] = exists
            if not exists:
                missing_required.append(f"directory: {dirname}")

        # Check optional files (for reporting only)
        for filename in optional_files:
            file_path = root / filename
            exists = file_path.exists() and file_path.is_file()
            status[f"optional_file:{filename}"] = exists

        # Check optional directories (for reporting only)
        for dirname in optional_dirs:
            dir_path = root / dirname
            exists = dir_path.exists() and dir_path.is_dir()
            status[f"optional_dir:{dirname}"] = exists

        # Raise error if any required items are missing
        if missing_required:
            # Get available files and directories for context
            available_files = [f.name for f in root.glob("*") if f.is_file()]
            available_dirs = [d.name for d in root.glob("*") if d.is_dir()]

            error_msg = (
                f"Directory structure validation failed for: {root}\n"
                f"Missing required items: {missing_required}\n"
                f"Available files: {available_files}\n"
                f"Available directories: {available_dirs}"
            )

            raise DirectoryValidationError(
                error_msg,
                context={
                    "root_path": str(root),
                    "missing_required": missing_required,
                    "available_files": available_files,
                    "available_dirs": available_dirs,
                    "status": status,
                },
            )

        logger.info(f"✅ Directory structure validated: {root}")
        logger.debug(f"Structure status: {status}")

        return root, status

    @staticmethod
    def get_missing_items(
        root_path: Union[str, Path],
        required_files: Optional[List[str]] = None,
        required_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get list of missing required items without raising exceptions.

        Args:
            root_path: Root directory to check
            required_files: List of required file names
            required_dirs: List of required directory names

        Returns:
            List of missing item descriptions
        """
        try:
            root = PathValidator.validate_directory_exists(root_path)
        except (ValueError, PathValidationError):
            return [f"Root directory: {root_path}"]

        missing = []

        # Check required files
        for filename in required_files or []:
            file_path = root / filename
            if not file_path.exists():
                missing.append(f"{filename} (file)")
            elif not file_path.is_file():
                missing.append(f"{filename} (not a file)")

        # Check required directories
        for dirname in required_dirs or []:
            dir_path = root / dirname
            if not dir_path.exists():
                missing.append(f"{dirname} (directory)")
            elif not dir_path.is_dir():
                missing.append(f"{dirname} (not a directory)")

        return missing


# Convenience functions for common validation patterns
def validate_file(path: Union[str, Path]) -> Path:
    """Convenience function to validate a single file."""
    return PathValidator.validate_file_exists(path)


def validate_directory(path: Union[str, Path]) -> Path:
    """Convenience function to validate a single directory."""
    return PathValidator.validate_directory_exists(path)


def validate_dataset_structure(
    data_root: Union[str, Path],
    train_file: str = "train.jsonl",
    val_file: str = "val.jsonl",
    images_dir: str = "images",
) -> Path:
    """Convenience function to validate standard dataset structure."""
    root, _ = PathValidator.validate_directory_structure(
        root_path=data_root,
        required_files=[train_file, val_file],
        required_dirs=[images_dir],
    )
    return root


# Export public API
__all__ = [
    "ValidationError",
    "PathValidationError",
    "DirectoryValidationError",
    "PathValidator",
    "validate_file",
    "validate_directory",
    "validate_dataset_structure",
    "expand_path_aliases",
    "normalize_path_input",
]
