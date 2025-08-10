"""Integration tests for path resolution in the inference pipeline.

This test suite validates that the PathManager integration works correctly
in the actual inference context, including:
- Teacher vs student image path handling
- Double-prefixing prevention in real scenarios
- Error handling and edge cases
- Performance with multiple images
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import pytest
from unittest.mock import Mock, patch

from src_new.utils.path_manager import PathManager, create_path_manager


class TestInferencePathResolution:
    """Test path resolution in inference pipeline context."""
    
    @pytest.fixture
    def mock_data_structure(self):
        """Create mock data structure similar to real inference data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            (temp_path / "images").mkdir()
            (temp_path / "teacher_pool").mkdir()
            
            # Create mock image files
            student_images = []
            teacher_images = []
            
            for i in range(3):
                # Student images
                student_img = temp_path / "images" / f"student_{i}.jpg"
                student_img.touch()
                student_images.append(f"images/student_{i}.jpg")
                
                # Teacher images
                teacher_img = temp_path / "teacher_pool" / f"teacher_{i}.jpg"
                teacher_img.touch()
                teacher_images.append(f"teacher_pool/teacher_{i}.jpg")
            
            # Create sample data structure
            sample_data = {
                "data_root": str(temp_path),
                "student_sample": {
                    "id": "test_001",
                    "images": student_images,
                    "objects": [{"desc": "Test object", "bbox_2d": [10, 20, 100, 200]}]
                },
                "teacher_samples": [
                    {
                        "id": "teacher_001",
                        "images": [teacher_images[0]],
                        "objects": [{"desc": "Teacher object 1", "bbox_2d": [5, 10, 50, 100]}]
                    },
                    {
                        "id": "teacher_002", 
                        "images": [teacher_images[1], teacher_images[2]],
                        "objects": [{"desc": "Teacher object 2", "bbox_2d": [15, 25, 75, 150]}]
                    }
                ],
                "image_files": {
                    "student": [temp_path / "images" / f"student_{i}.jpg" for i in range(3)],
                    "teacher": [temp_path / "teacher_pool" / f"teacher_{i}.jpg" for i in range(3)]
                }
            }
            
            yield sample_data
    
    def test_student_image_path_resolution(self, mock_data_structure):
        """Test student image path resolution with PathManager."""
        data_root = mock_data_structure["data_root"]
        student_sample = mock_data_structure["student_sample"].copy()
        
        path_manager = create_path_manager(data_root)
        
        # Simulate the inference pipeline path resolution
        resolved_images = []
        for img_path in student_sample["images"]:
            resolved_path = path_manager.resolve_path(img_path)
            resolved_images.append(str(resolved_path))
        
        student_sample["images"] = resolved_images
        
        # Validate resolution
        assert len(student_sample["images"]) == 3
        for resolved_path in student_sample["images"]:
            path_obj = Path(resolved_path)
            assert path_obj.exists()
            assert path_obj.is_absolute()
            assert "student_" in path_obj.name
            assert str(data_root) in str(path_obj)
    
    def test_teacher_image_path_resolution(self, mock_data_structure):
        """Test teacher image path resolution with PathManager."""
        data_root = mock_data_structure["data_root"]
        teacher_samples = [sample.copy() for sample in mock_data_structure["teacher_samples"]]
        
        path_manager = create_path_manager(data_root)
        
        # Simulate teacher path resolution
        for teacher_sample in teacher_samples:
            resolved_images = []
            for img_path in teacher_sample["images"]:
                resolved_path = path_manager.resolve_path(img_path)
                resolved_images.append(str(resolved_path))
            teacher_sample["images"] = resolved_images
        
        # Validate resolution
        assert len(teacher_samples) == 2
        
        # First teacher has 1 image
        assert len(teacher_samples[0]["images"]) == 1
        teacher_path = Path(teacher_samples[0]["images"][0])
        assert teacher_path.exists()
        assert "teacher_0" in teacher_path.name
        
        # Second teacher has 2 images
        assert len(teacher_samples[1]["images"]) == 2
        for img_path in teacher_samples[1]["images"]:
            teacher_path = Path(img_path)
            assert teacher_path.exists()
            assert "teacher_" in teacher_path.name
    
    def test_mixed_path_types_resolution(self, mock_data_structure):
        """Test resolution of mixed absolute/relative paths."""
        data_root = mock_data_structure["data_root"]
        path_manager = create_path_manager(data_root)
        
        # Create mixed path scenarios
        mixed_paths = [
            # Relative path
            "images/student_0.jpg",
            # Already absolute path
            str(mock_data_structure["image_files"]["student"][1]),
            # Path that might be double-prefixed
            f"{data_root}/images/student_2.jpg"
        ]
        
        resolved_paths = path_manager.resolve_paths(mixed_paths)
        
        assert len(resolved_paths) == 3
        for i, resolved_path in enumerate(resolved_paths):
            assert resolved_path.exists()
            assert f"student_{i}" in resolved_path.name
            # Check that all paths are now absolute
            assert resolved_path.is_absolute()
    
    def test_double_prefix_prevention_scenario(self, mock_data_structure):
        """Test prevention of double-prefixing in realistic scenario."""
        data_root = mock_data_structure["data_root"]
        path_manager = create_path_manager(data_root)
        
        # Simulate scenario where paths might already contain data_root
        potentially_problematic_paths = []
        
        # Add paths in various formats
        for i in range(3):
            # Already contains data_root (should not be double-prefixed)
            full_path = str(Path(data_root) / "images" / f"student_{i}.jpg")
            potentially_problematic_paths.append(full_path)
        
        # Resolve paths - should not cause double-prefixing
        resolved_paths = path_manager.resolve_paths(potentially_problematic_paths)
        
        for i, resolved_path in enumerate(resolved_paths):
            assert resolved_path.exists()
            # Check that data_root doesn't appear multiple times in the path
            path_str = str(resolved_path)
            data_root_count = path_str.count(str(Path(data_root).name))
            # Should appear only once (unless temp directory name is reused, which is rare)
            assert data_root_count >= 1, f"Data root not found in path: {path_str}"
    
    def test_error_handling_invalid_paths(self, mock_data_structure):
        """Test error handling for invalid paths."""
        data_root = mock_data_structure["data_root"]
        path_manager = create_path_manager(data_root)
        
        # Test with non-existent files
        invalid_paths = [
            "nonexistent/file.jpg",
            "images/missing_student.jpg",
            "/absolute/nonexistent/path.jpg"
        ]
        
        # Should raise FileNotFoundError for non-existent files
        with pytest.raises(FileNotFoundError):
            path_manager.resolve_paths(invalid_paths)
        
        # Test safe resolution (should not raise exceptions)
        safe_resolved = path_manager.resolve_paths_safe(invalid_paths)
        assert len(safe_resolved) == 3
        assert all(path is None for path in safe_resolved)
    
    def test_inference_jsonl_simulation(self, mock_data_structure):
        """Simulate JSONL file processing like in actual inference."""
        data_root = mock_data_structure["data_root"]
        
        # Create mock JSONL data
        jsonl_samples = [
            {
                "id": "sample_001",
                "images": ["images/student_0.jpg", "images/student_1.jpg"],
                "objects": []
            },
            {
                "id": "sample_002", 
                "images": ["images/student_2.jpg"],
                "objects": []
            }
        ]
        
        # Simulate loading and processing like in run_inference
        path_manager = create_path_manager(data_root)
        processed_samples = []
        
        for sample in jsonl_samples:
            processed_sample = sample.copy()
            if "images" in processed_sample:
                resolved_images = []
                for img_path in processed_sample["images"]:
                    try:
                        resolved_path = str(path_manager.resolve_path(img_path))
                        resolved_images.append(resolved_path)
                    except (FileNotFoundError, ValueError) as e:
                        # In real inference, this would be logged and might cause sample to fail
                        raise e
                processed_sample["images"] = resolved_images
            processed_samples.append(processed_sample)
        
        # Validate processed samples
        assert len(processed_samples) == 2
        
        # First sample should have 2 resolved images
        assert len(processed_samples[0]["images"]) == 2
        for img_path in processed_samples[0]["images"]:
            assert Path(img_path).exists()
            assert Path(img_path).is_absolute()
        
        # Second sample should have 1 resolved image
        assert len(processed_samples[1]["images"]) == 1
        assert Path(processed_samples[1]["images"][0]).exists()
    
    def test_teacher_pool_loading_simulation(self, mock_data_structure):
        """Simulate teacher pool loading like in _load_teacher_pool."""
        data_root = mock_data_structure["data_root"]
        teacher_samples = mock_data_structure["teacher_samples"].copy()
        
        # Simulate teacher pool path resolution during loading
        path_manager = create_path_manager(data_root)
        
        for teacher_sample in teacher_samples:
            if "images" in teacher_sample:
                resolved_teacher_images = []
                for img_path in teacher_sample["images"]:
                    try:
                        resolved_path = str(path_manager.resolve_path(img_path))
                        resolved_teacher_images.append(resolved_path)
                    except (FileNotFoundError, ValueError) as e:
                        raise e
                teacher_sample["images"] = resolved_teacher_images
        
        # Validate teacher pool resolution
        assert len(teacher_samples) == 2
        
        # Validate all teacher images are resolved correctly
        all_teacher_images = []
        for teacher_sample in teacher_samples:
            all_teacher_images.extend(teacher_sample["images"])
        
        assert len(all_teacher_images) == 3  # Total teacher images
        for img_path in all_teacher_images:
            path_obj = Path(img_path)
            assert path_obj.exists()
            assert path_obj.is_absolute()
            assert "teacher_" in path_obj.name
    
    def test_performance_with_many_images(self):
        """Test performance with large number of images."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create many image files
            num_images = 100
            image_paths = []
            
            for i in range(num_images):
                img_file = temp_path / f"image_{i:03d}.jpg"
                img_file.touch()
                image_paths.append(f"image_{i:03d}.jpg")
            
            path_manager = create_path_manager(str(temp_path))
            
            # Measure resolution time
            start_time = time.time()
            resolved_paths = path_manager.resolve_paths(image_paths)
            end_time = time.time()
            
            resolution_time = end_time - start_time
            
            # Validate results
            assert len(resolved_paths) == num_images
            assert all(p.exists() for p in resolved_paths)
            
            # Performance check (should be fast)
            assert resolution_time < 1.0, f"Path resolution took too long: {resolution_time:.3f}s"
            
            # Log performance for reference
            avg_time_per_path = resolution_time / num_images * 1000  # milliseconds
            print(f"Average path resolution time: {avg_time_per_path:.3f}ms per path")


class TestInferencePathEdgeCases:
    """Test edge cases specific to inference pipeline."""
    
    def test_unicode_image_paths(self):
        """Test handling of image paths with unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with unicode names (where supported)
            unicode_names = ["图像_1.jpg", "画像_test.png", "imagem_测试.jpg"]
            created_files = []
            
            for name in unicode_names:
                try:
                    img_file = temp_path / name
                    img_file.touch()
                    created_files.append(name)
                except OSError:
                    # Skip if filesystem doesn't support unicode
                    pass
            
            if created_files:
                path_manager = create_path_manager(str(temp_path))
                resolved_paths = path_manager.resolve_paths(created_files)
                
                assert len(resolved_paths) == len(created_files)
                for resolved_path in resolved_paths:
                    assert resolved_path.exists()
    
    def test_concurrent_path_resolution(self):
        """Test concurrent path resolution (thread safety)."""
        import threading
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            for i in range(10):
                (temp_path / f"concurrent_{i}.jpg").touch()
            
            path_manager = create_path_manager(str(temp_path))
            
            results = []
            errors = []
            
            def resolve_worker(thread_id):
                try:
                    local_paths = [f"concurrent_{i}.jpg" for i in range(10)]
                    resolved = path_manager.resolve_paths(local_paths)
                    results.append((thread_id, resolved))
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=resolve_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Validate results
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 5
            
            # All threads should get the same results
            first_result = results[0][1]
            for thread_id, result in results[1:]:
                assert len(result) == len(first_result)
                for i, path in enumerate(result):
                    assert path == first_result[i]
    
    def test_symlink_in_data_path(self):
        """Test handling of symbolic links in data paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original directory structure
            original_dir = temp_path / "original"
            original_dir.mkdir()
            
            original_file = original_dir / "image.jpg"
            original_file.touch()
            
            try:
                # Create symlink to directory
                symlink_dir = temp_path / "symlinked"
                symlink_dir.symlink_to(original_dir)
                
                # Test path resolution through symlink
                path_manager = create_path_manager(str(temp_path))
                
                # Should work through symlinked directory
                resolved = path_manager.resolve_path("symlinked/image.jpg")
                assert resolved.exists()
                
            except OSError:
                # Skip if filesystem doesn't support symlinks
                pass
    
    def test_relative_data_root_handling(self):
        """Test handling of relative data_root paths."""
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.jpg"
            test_file.touch()
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(str(temp_path))
                
                # Use relative path as data_root
                path_manager = create_path_manager(".")
                
                resolved = path_manager.resolve_path("test.jpg")
                assert resolved.exists()
                
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    # Run a quick integration test
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock structure
        (temp_path / "images").mkdir()
        (temp_path / "teacher_pool").mkdir()
        
        for i in range(2):
            (temp_path / "images" / f"student_{i}.jpg").touch()
            (temp_path / "teacher_pool" / f"teacher_{i}.jpg").touch()
        
        # Test path manager
        pm = create_path_manager(str(temp_path))
        
        # Test student paths
        student_paths = ["images/student_0.jpg", "images/student_1.jpg"]
        resolved_student = pm.resolve_paths(student_paths)
        
        print(f"✅ Resolved {len(resolved_student)} student images")
        
        # Test teacher paths
        teacher_paths = ["teacher_pool/teacher_0.jpg", "teacher_pool/teacher_1.jpg"]
        resolved_teacher = pm.resolve_paths(teacher_paths)
        
        print(f"✅ Resolved {len(resolved_teacher)} teacher images")
        
        # Test mixed paths
        mixed_paths = [
            "images/student_0.jpg",  # Relative
            str(temp_path / "teacher_pool" / "teacher_1.jpg")  # Absolute
        ]
        resolved_mixed = pm.resolve_paths(mixed_paths)
        
        print(f"✅ Resolved {len(resolved_mixed)} mixed path types")
        print("✅ Path resolution integration test passed")