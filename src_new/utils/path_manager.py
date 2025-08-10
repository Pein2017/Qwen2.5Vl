"""Unified path management system for consistent image path resolution.

This module provides the PathManager class to handle path resolution
consistently across the inference pipeline, preventing double-prefixing
issues and providing robust error handling for both teacher and student
image paths.
"""

from pathlib import Path
from typing import List, Optional, Union

from .rank_aware_logging import get_logger


# Path management logger
logger = get_logger("path_manager")


class PathManager:
    """Unified path management system for consistent image path resolution.

    Prevents double-prefixing issues and provides consistent path handling
    for both teacher and student images in the inference pipeline.
    """

    def __init__(self, data_root: Optional[str] = None):
        """Initialize PathManager with optional data root.

        Args:
            data_root: Base directory for relative path resolution
        """
        self.data_root = Path(data_root) if data_root else None
        if self.data_root:
            logger.debug(f"PathManager initialized with data_root: {self.data_root}")
        else:
            logger.debug("PathManager initialized without data_root")

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a single path, preventing double-prefixing.

        Args:
            path: Path to resolve (absolute, relative, or already resolved)

        Returns:
            Resolved absolute Path object

        Raises:
            FileNotFoundError: If resolved path doesn't exist
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Path cannot be empty or None")

        path_obj = Path(path)

        # If already absolute, return as-is (after validation)
        if path_obj.is_absolute():
            if not path_obj.exists():
                raise FileNotFoundError(f"Absolute path does not exist: {path_obj}")
            logger.debug(f"Using absolute path: {path_obj}")
            return path_obj

        # Handle relative paths
        if not self.data_root:
            # Try to resolve relative to current working directory
            resolved = Path.cwd() / path_obj
            if not resolved.exists():
                raise FileNotFoundError(
                    f"No data_root provided and relative path not found: {resolved}"
                )
            logger.debug(
                f"Resolved relative path (no data_root): {path_obj} -> {resolved}"
            )
            return resolved

        # Check if path already contains data_root to prevent double-prefixing
        path_str = str(path_obj)
        data_root_str = str(self.data_root)

        if path_str.startswith(data_root_str):
            # Path already contains data_root, treat as absolute
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"Path with embedded data_root does not exist: {path_obj}"
                )
            logger.debug(f"Path already contains data_root: {path_obj}")
            return path_obj

        # Resolve relative path against data_root
        resolved = self.data_root / path_obj
        if not resolved.exists():
            raise FileNotFoundError(f"Resolved path does not exist: {resolved}")

        logger.debug(f"Resolved relative path: {path_obj} -> {resolved}")
        return resolved

    def resolve_paths(self, paths: List[Union[str, Path]]) -> List[Path]:
        """Resolve multiple paths consistently.

        Args:
            paths: List of paths to resolve

        Returns:
            List of resolved Path objects

        Raises:
            ValueError: If any path is invalid
            FileNotFoundError: If any resolved path doesn't exist
        """
        if not paths:
            return []

        resolved_paths = []
        for i, path in enumerate(paths):
            try:
                resolved = self.resolve_path(path)
                resolved_paths.append(resolved)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Failed to resolve path {i}: {path}")
                raise e

        logger.debug(f"Successfully resolved {len(resolved_paths)} paths")
        return resolved_paths

    def resolve_paths_safe(self, paths: List[Union[str, Path]]) -> List[Optional[Path]]:
        """Resolve multiple paths without raising exceptions on individual failures.

        Args:
            paths: List of paths to resolve

        Returns:
            List of resolved Path objects (None for failed resolutions)
        """
        if not paths:
            return []

        resolved_paths = []
        for i, path in enumerate(paths):
            try:
                resolved = self.resolve_path(path)
                resolved_paths.append(resolved)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Failed to resolve path {i}: {path} - {e}")
                resolved_paths.append(None)

        successful_count = sum(1 for p in resolved_paths if p is not None)
        logger.debug(f"Successfully resolved {successful_count}/{len(paths)} paths")
        return resolved_paths

    def validate_paths_exist(self, paths: List[Union[str, Path]]) -> List[bool]:
        """Check which paths exist without raising exceptions.

        Args:
            paths: List of paths to check

        Returns:
            List of boolean existence flags
        """
        existence_flags = []
        for path in paths:
            try:
                self.resolve_path(path)  # Just check if resolution works
                existence_flags.append(True)
            except (ValueError, FileNotFoundError):
                existence_flags.append(False)

        return existence_flags

    def get_relative_path(self, path: Union[str, Path]) -> Optional[Path]:
        """Get relative path from data_root if applicable.

        Args:
            path: Path to make relative

        Returns:
            Relative path if possible, None otherwise
        """
        if not self.data_root:
            return None

        path_obj = Path(path)
        try:
            return path_obj.relative_to(self.data_root)
        except ValueError:
            # Path is not under data_root
            return None

    def update_data_root(self, new_data_root: Optional[str]) -> None:
        """Update data_root for future path resolutions.

        Args:
            new_data_root: New data root directory
        """
        old_root = self.data_root
        self.data_root = Path(new_data_root) if new_data_root else None
        logger.debug(f"Updated data_root: {old_root} -> {self.data_root}")

    def is_double_prefixed(self, path: Union[str, Path]) -> bool:
        """Check if a path contains double-prefixing with data_root.

        Args:
            path: Path to check

        Returns:
            True if path appears to be double-prefixed
        """
        if not self.data_root:
            return False

        path_str = str(path)
        data_root_str = str(self.data_root)

        # Check for patterns like /data_root/data_root/...
        double_prefix_pattern = f"{data_root_str}/{data_root_str.split('/')[-1]}"
        return double_prefix_pattern in path_str

    def fix_double_prefix(self, path: Union[str, Path]) -> Path:
        """Attempt to fix double-prefixed paths.

        Args:
            path: Potentially double-prefixed path

        Returns:
            Corrected path
        """
        path_obj = Path(path)

        if not self.is_double_prefixed(path):
            return path_obj

        path_str = str(path_obj)
        data_root_str = str(self.data_root)

        # Remove the first occurrence of data_root if it appears twice
        if path_str.count(data_root_str) >= 2:
            corrected_str = path_str.replace(data_root_str + "/", "", 1)
            corrected = Path(corrected_str)
            logger.debug(f"Fixed double-prefixed path: {path_obj} -> {corrected}")
            return corrected

        return path_obj


def create_path_manager(data_root: Optional[str] = None) -> PathManager:
    """Factory function to create PathManager instance.

    Args:
        data_root: Base directory for relative path resolution

    Returns:
        Configured PathManager instance
    """
    return PathManager(data_root=data_root)


def resolve_image_paths(
    image_paths: List[Union[str, Path]], data_root: Optional[str] = None
) -> List[Path]:
    """Convenience function to resolve image paths with logging.

    Args:
        image_paths: List of image paths to resolve
        data_root: Base directory for relative paths

    Returns:
        List of resolved Path objects

    Raises:
        FileNotFoundError: If any image path cannot be resolved
    """
    if not image_paths:
        return []

    path_manager = create_path_manager(data_root)
    resolved = path_manager.resolve_paths(image_paths)

    logger.debug(f"Resolved {len(resolved)} image paths")
    return resolved


def safe_resolve_image_paths(
    image_paths: List[Union[str, Path]], data_root: Optional[str] = None
) -> List[Optional[Path]]:
    """Convenience function to safely resolve image paths without exceptions.

    Args:
        image_paths: List of image paths to resolve
        data_root: Base directory for relative paths

    Returns:
        List of resolved Path objects (None for failed resolutions)
    """
    if not image_paths:
        return []

    path_manager = create_path_manager(data_root)
    resolved = path_manager.resolve_paths_safe(image_paths)

    successful_count = sum(1 for p in resolved if p is not None)
    logger.debug(f"Safely resolved {successful_count}/{len(image_paths)} image paths")
    return resolved
