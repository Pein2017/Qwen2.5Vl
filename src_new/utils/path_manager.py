"""Unified path management system for consistent image path resolution.

This module provides the PathManager class to handle path resolution
consistently across the inference pipeline, preventing double-prefixing
issues and providing robust error handling for both teacher and student
image paths.
"""

from pathlib import Path
from typing import List, Optional, Union

from .logger_factory import get_module_logger
from .validation import PathValidationError, PathValidator


# Path management logger
logger = get_module_logger(__name__)


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
        # Avoid premature validation on raw relative paths. Validate only after full resolution.
        if not path:
            raise ValueError("path cannot be empty or None")

        raw_path = Path(path)

        # Absolute path: validate and return as-is. In safe resolution mode, callers catch exceptions.
        if raw_path.is_absolute():
            validated = PathValidator.validate_path_exists(raw_path, "path")
            logger.debug(f"Using absolute path: {validated}")
            return validated

        # Relative path: prefer data_root if provided
        if self.data_root is not None:
            # If caller accidentally included data_root inside the relative path, avoid double prefix
            # Example: data_root=/abs/root and path='images/..' (OK) or 'data/ds/.../images/..' (avoid /abs/root/data/ds/...)
            candidate = (self.data_root / raw_path).resolve()
            if candidate.exists():
                logger.debug(
                    f"Resolved relative path against data_root: {raw_path} -> {candidate}"
                )
                return candidate

            # As a fallback, if the raw string already starts with the data_root name, try trimming it once
            try:
                data_root_name = self.data_root.name
                parts = list(raw_path.parts)
                if parts and parts[0] == data_root_name:
                    trimmed = Path(*parts[1:])
                    candidate2 = (self.data_root / trimmed).resolve()
                    if candidate2.exists():
                        logger.debug(
                            f"Resolved by trimming embedded data_root name: {raw_path} -> {candidate2}"
                        )
                        return candidate2
            except Exception:
                pass

            # If still not found, raise with actionable message
            raise FileNotFoundError(
                f"Resolved path does not exist: {(self.data_root / raw_path).resolve()}"
            )

        # No data_root: resolve relative to CWD
        candidate = (Path.cwd() / raw_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"No data_root provided and relative path not found: {candidate}"
            )
        logger.debug(
            f"Resolved relative path (no data_root): {raw_path} -> {candidate}"
        )
        return candidate

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
            except (ValueError, FileNotFoundError, PathValidationError) as e:
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
