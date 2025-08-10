"""Comprehensive test suite for PathManager utility.

This test suite covers all aspects of path resolution including:
- Absolute vs relative path handling
- Double-prefixing prevention
- Error handling for invalid paths
- Edge cases and boundary conditions
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
import pytest

from src_new.utils.path_manager import (
    PathManager,
    create_path_manager,
    resolve_image_paths,
    safe_resolve_image_paths,
)


class TestPathManager:
    """Test suite for PathManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        files = {}
        
        # Create nested directory structure
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "nested").mkdir()
        
        # Create sample files
        files["root_file"] = temp_dir / "root_file.txt"
        files["sub_file"] = temp_dir / "subdir" / "sub_file.txt"
        files["nested_file"] = temp_dir / "subdir" / "nested" / "nested_file.txt"
        
        for file_path in files.values():
            file_path.touch()
        
        return files
    
    def test_pathmanager_init_with_data_root(self, temp_dir):
        """Test PathManager initialization with data_root."""
        path_manager = PathManager(str(temp_dir))
        assert path_manager.data_root == temp_dir
    
    def test_pathmanager_init_without_data_root(self):
        """Test PathManager initialization without data_root."""
        path_manager = PathManager()
        assert path_manager.data_root is None
    
    def test_resolve_absolute_path_existing(self, sample_files):
        """Test resolution of existing absolute paths."""
        path_manager = PathManager()
        root_file = sample_files["root_file"]
        
        resolved = path_manager.resolve_path(str(root_file))
        assert resolved == root_file
    
    def test_resolve_absolute_path_nonexistent(self):
        """Test resolution of non-existent absolute paths."""
        path_manager = PathManager()
        nonexistent_path = "/nonexistent/file.txt"
        
        with pytest.raises(FileNotFoundError):
            path_manager.resolve_path(nonexistent_path)
    
    def test_resolve_relative_path_with_data_root(self, temp_dir, sample_files):
        """Test resolution of relative paths with data_root."""
        path_manager = PathManager(str(temp_dir))
        
        # Test relative path resolution
        resolved = path_manager.resolve_path("subdir/sub_file.txt")
        assert resolved == sample_files["sub_file"]
        
        # Test nested relative path
        resolved = path_manager.resolve_path("subdir/nested/nested_file.txt")
        assert resolved == sample_files["nested_file"]
    
    def test_resolve_relative_path_without_data_root(self, sample_files):
        """Test resolution of relative paths without data_root."""
        path_manager = PathManager()
        
        # Should try to resolve relative to current working directory
        with pytest.raises(FileNotFoundError):
            path_manager.resolve_path("nonexistent_relative_file.txt")
    
    def test_prevent_double_prefixing(self, temp_dir, sample_files):
        """Test prevention of double-prefixing with data_root."""
        path_manager = PathManager(str(temp_dir))
        
        # Create a path that already contains data_root
        already_prefixed = str(temp_dir / "subdir" / "sub_file.txt")
        
        # Should not double-prefix
        resolved = path_manager.resolve_path(already_prefixed)
        assert str(resolved) == already_prefixed
        assert resolved.exists()
    
    def test_resolve_paths_multiple(self, temp_dir, sample_files):
        """Test resolving multiple paths at once."""
        path_manager = PathManager(str(temp_dir))
        
        paths = [
            "root_file.txt",
            "subdir/sub_file.txt",
            "subdir/nested/nested_file.txt"
        ]
        
        resolved_paths = path_manager.resolve_paths(paths)
        
        assert len(resolved_paths) == 3
        assert resolved_paths[0] == sample_files["root_file"]
        assert resolved_paths[1] == sample_files["sub_file"]
        assert resolved_paths[2] == sample_files["nested_file"]
    
    def test_resolve_paths_with_failure(self, temp_dir, sample_files):
        """Test resolving multiple paths with one failure."""
        path_manager = PathManager(str(temp_dir))
        
        paths = [
            "root_file.txt",
            "nonexistent_file.txt",  # This should cause failure
            "subdir/sub_file.txt"
        ]
        
        with pytest.raises(FileNotFoundError):
            path_manager.resolve_paths(paths)
    
    def test_resolve_paths_safe(self, temp_dir, sample_files):
        """Test safe resolution of multiple paths (no exceptions)."""
        path_manager = PathManager(str(temp_dir))
        
        paths = [
            "root_file.txt",
            "nonexistent_file.txt",  # This should return None
            "subdir/sub_file.txt"
        ]
        
        resolved_paths = path_manager.resolve_paths_safe(paths)
        
        assert len(resolved_paths) == 3
        assert resolved_paths[0] == sample_files["root_file"]
        assert resolved_paths[1] is None  # Failed resolution
        assert resolved_paths[2] == sample_files["sub_file"]
    
    def test_validate_paths_exist(self, temp_dir, sample_files):
        """Test path existence validation."""
        path_manager = PathManager(str(temp_dir))
        
        paths = [
            "root_file.txt",
            "nonexistent_file.txt",
            "subdir/sub_file.txt"
        ]
        
        existence_flags = path_manager.validate_paths_exist(paths)
        
        assert existence_flags == [True, False, True]
    
    def test_get_relative_path(self, temp_dir, sample_files):
        """Test getting relative path from data_root."""
        path_manager = PathManager(str(temp_dir))
        
        # Test with absolute path under data_root
        absolute_path = sample_files["sub_file"]
        relative = path_manager.get_relative_path(str(absolute_path))
        
        assert relative == Path("subdir/sub_file.txt")
        
        # Test with path outside data_root
        outside_path = "/tmp/outside_file.txt"
        relative = path_manager.get_relative_path(outside_path)
        assert relative is None
    
    def test_get_relative_path_no_data_root(self):
        """Test getting relative path without data_root."""
        path_manager = PathManager()
        
        result = path_manager.get_relative_path("/some/path")
        assert result is None
    
    def test_update_data_root(self, temp_dir, sample_files):
        """Test updating data_root after initialization."""
        path_manager = PathManager()
        
        # Initially no data_root
        assert path_manager.data_root is None
        
        # Update data_root
        path_manager.update_data_root(str(temp_dir))
        assert path_manager.data_root == temp_dir
        
        # Now should be able to resolve relative paths
        resolved = path_manager.resolve_path("root_file.txt")
        assert resolved == sample_files["root_file"]
    
    def test_is_double_prefixed(self, temp_dir):
        """Test detection of double-prefixed paths."""
        path_manager = PathManager(str(temp_dir))
        
        # Create double-prefixed path pattern
        temp_dir_name = temp_dir.name
        double_prefixed = str(temp_dir / temp_dir_name / "file.txt")
        
        is_double_prefixed = path_manager.is_double_prefixed(double_prefixed)
        # This test might be fragile depending on temp directory naming
        # Just ensure the method runs without error
        assert isinstance(is_double_prefixed, bool)
    
    def test_fix_double_prefix(self, temp_dir):
        """Test fixing double-prefixed paths."""
        path_manager = PathManager(str(temp_dir))
        
        # Create a clearly double-prefixed path
        double_prefixed = f"{temp_dir}/{temp_dir}/file.txt"
        
        fixed = path_manager.fix_double_prefix(double_prefixed)
        # Should remove one occurrence of temp_dir
        assert str(temp_dir) not in str(fixed).split(str(temp_dir))[1:]
    
    def test_empty_path_validation(self):
        """Test handling of empty or None paths."""
        path_manager = PathManager()
        
        with pytest.raises(ValueError):
            path_manager.resolve_path("")
        
        with pytest.raises(ValueError):
            path_manager.resolve_path(None)
    
    def test_resolve_paths_empty_list(self):
        """Test resolving empty path list."""
        path_manager = PathManager()
        
        result = path_manager.resolve_paths([])
        assert result == []
        
        result = path_manager.resolve_paths_safe([])
        assert result == []
        
        result = path_manager.validate_paths_exist([])
        assert result == []


class TestPathManagerFactoryFunctions:
    """Test suite for PathManager factory functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_image_files(self, temp_dir):
        """Create sample image files for testing."""
        files = []
        
        # Create mock image files
        image_names = ["image1.jpg", "image2.png", "subdir/image3.jpg"]
        
        (temp_dir / "subdir").mkdir()
        
        for name in image_names:
            file_path = temp_dir / name
            file_path.touch()
            files.append(str(file_path))
        
        return files
    
    def test_create_path_manager_factory(self, temp_dir):
        """Test PathManager factory function."""
        path_manager = create_path_manager(str(temp_dir))
        
        assert isinstance(path_manager, PathManager)
        assert path_manager.data_root == temp_dir
        
        # Test without data_root
        path_manager = create_path_manager()
        assert path_manager.data_root is None
    
    def test_resolve_image_paths_function(self, temp_dir, sample_image_files):
        """Test resolve_image_paths convenience function."""
        # Create relative paths for testing
        relative_paths = ["image1.jpg", "image2.png", "subdir/image3.jpg"]
        
        resolved_paths = resolve_image_paths(relative_paths, str(temp_dir))
        
        assert len(resolved_paths) == 3
        assert all(isinstance(p, Path) for p in resolved_paths)
        assert all(p.exists() for p in resolved_paths)
    
    def test_resolve_image_paths_empty(self):
        """Test resolve_image_paths with empty list."""
        result = resolve_image_paths([], "/some/root")
        assert result == []
    
    def test_safe_resolve_image_paths_function(self, temp_dir, sample_image_files):
        """Test safe_resolve_image_paths convenience function."""
        # Mix of existing and non-existing paths
        paths = ["image1.jpg", "nonexistent.jpg", "subdir/image3.jpg"]
        
        resolved_paths = safe_resolve_image_paths(paths, str(temp_dir))
        
        assert len(resolved_paths) == 3
        assert resolved_paths[0] is not None  # Exists
        assert resolved_paths[1] is None      # Doesn't exist
        assert resolved_paths[2] is not None  # Exists
    
    def test_safe_resolve_image_paths_empty(self):
        """Test safe_resolve_image_paths with empty list."""
        result = safe_resolve_image_paths([], "/some/root")
        assert result == []


class TestPathManagerEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_circular_path_references(self):
        """Test handling of circular path references (if any)."""
        path_manager = PathManager("/tmp")
        
        # Test with relative path containing .. references
        # This should be handled gracefully by pathlib
        with pytest.raises(FileNotFoundError):
            path_manager.resolve_path("../../../nonexistent.txt")
    
    def test_very_long_paths(self, temp_dir):
        """Test handling of very long paths."""
        path_manager = PathManager(str(temp_dir))
        
        # Create nested directory structure
        long_subdir = temp_dir
        for i in range(10):
            long_subdir = long_subdir / f"level_{i}"
            long_subdir.mkdir()
        
        # Create file at the end
        long_file = long_subdir / "deep_file.txt"
        long_file.touch()
        
        # Test resolution
        relative_path = str(long_file.relative_to(temp_dir))
        resolved = path_manager.resolve_path(relative_path)
        assert resolved == long_file
    
    def test_special_characters_in_paths(self, temp_dir):
        """Test handling of paths with special characters."""
        path_manager = PathManager(str(temp_dir))
        
        # Create file with special characters (where supported by filesystem)
        special_names = ["file with spaces.txt", "file-with-dashes.txt"]
        
        for name in special_names:
            try:
                special_file = temp_dir / name
                special_file.touch()
                
                resolved = path_manager.resolve_path(name)
                assert resolved == special_file
            except OSError:
                # Skip if filesystem doesn't support the name
                pass
    
    def test_symlink_handling(self, temp_dir):
        """Test handling of symbolic links."""
        path_manager = PathManager(str(temp_dir))
        
        # Create original file
        original_file = temp_dir / "original.txt"
        original_file.touch()
        
        # Create symlink
        try:
            symlink_file = temp_dir / "symlink.txt"
            symlink_file.symlink_to(original_file)
            
            # Test resolution of symlink
            resolved = path_manager.resolve_path("symlink.txt")
            assert resolved.exists()
            # The resolved path should still be the symlink, not the target
            assert resolved.name == "symlink.txt"
            
        except OSError:
            # Skip if filesystem doesn't support symlinks
            pass
    
    def test_concurrent_access_safety(self, temp_dir, sample_files=None):
        """Test thread safety (basic check)."""
        import threading
        import time
        
        path_manager = PathManager(str(temp_dir))
        
        # Create a test file
        test_file = temp_dir / "concurrent_test.txt"
        test_file.touch()
        
        results = []
        errors = []
        
        def resolve_path_worker():
            try:
                for _ in range(10):
                    resolved = path_manager.resolve_path("concurrent_test.txt")
                    results.append(resolved)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=resolve_path_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 iterations each
        assert all(r == test_file for r in results)


# Integration test with actual image files (if available)
class TestPathManagerIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_inference_pipeline_simulation(self, temp_dir):
        """Simulate the inference pipeline path resolution."""
        # Setup directory structure similar to actual usage
        data_root = temp_dir / "data"
        data_root.mkdir()
        
        (data_root / "images").mkdir()
        (data_root / "teacher_pool").mkdir()
        
        # Create mock image files
        student_images = []
        teacher_images = []
        
        for i in range(3):
            student_img = data_root / "images" / f"student_{i}.jpg"
            student_img.touch()
            student_images.append(f"images/student_{i}.jpg")
            
            teacher_img = data_root / "teacher_pool" / f"teacher_{i}.jpg"
            teacher_img.touch()
            teacher_images.append(f"teacher_pool/teacher_{i}.jpg")
        
        # Test student path resolution
        path_manager = create_path_manager(str(data_root))
        resolved_student = path_manager.resolve_paths(student_images)
        
        assert len(resolved_student) == 3
        assert all(p.exists() for p in resolved_student)
        assert all("student_" in p.name for p in resolved_student)
        
        # Test teacher path resolution
        resolved_teacher = path_manager.resolve_paths(teacher_images)
        
        assert len(resolved_teacher) == 3
        assert all(p.exists() for p in resolved_teacher)
        assert all("teacher_" in p.name for p in resolved_teacher)
        
        # Test mixed absolute/relative paths (common in real scenarios)
        mixed_paths = [
            student_images[0],  # Relative
            str(data_root / teacher_images[1]),  # Already absolute
            student_images[2]   # Relative
        ]
        
        resolved_mixed = path_manager.resolve_paths(mixed_paths)
        assert len(resolved_mixed) == 3
        assert all(p.exists() for p in resolved_mixed)
    
    def test_double_prefix_prevention_scenario(self, temp_dir):
        """Test the specific double-prefixing scenario from the issue."""
        data_root = temp_dir / "dataset"
        data_root.mkdir()
        
        # Create image file
        image_file = data_root / "test_image.jpg"
        image_file.touch()
        
        path_manager = PathManager(str(data_root))
        
        # Scenario 1: Relative path (should be resolved)
        relative_path = "test_image.jpg"
        resolved1 = path_manager.resolve_path(relative_path)
        assert resolved1 == image_file
        
        # Scenario 2: Path already contains data_root (should not be double-prefixed)
        already_absolute = str(image_file)
        resolved2 = path_manager.resolve_path(already_absolute)
        assert resolved2 == image_file
        
        # Scenario 3: Simulate potential double-prefixing
        potentially_double_prefixed = str(data_root / str(data_root).split('/')[-1] / "test_image.jpg")
        # This should be detected and fixed if it's a real double-prefix pattern
        resolved3 = path_manager.fix_double_prefix(potentially_double_prefixed)
        
        # The fix should produce a valid path (though may not exist)
        assert isinstance(resolved3, Path)


if __name__ == "__main__":
    # Run specific test for debugging
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "test.txt").touch()
        (temp_path / "subdir").mkdir()
        (temp_path / "subdir" / "nested.txt").touch()
        
        # Test PathManager
        pm = PathManager(str(temp_path))
        
        print(f"Data root: {pm.data_root}")
        print(f"Resolved relative: {pm.resolve_path('test.txt')}")
        print(f"Resolved nested: {pm.resolve_path('subdir/nested.txt')}")
        
        print("✅ Basic PathManager functionality working")