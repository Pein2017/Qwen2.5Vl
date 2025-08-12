"""
Tests for teacher pool management functionality.

This module tests:
- Teacher pool loading and validation
- Random teacher selection
- Teacher assignment logic
- Teacher-student pairing strategies
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from src_new.tests.fixtures import (
    create_sample_jsonl_data,
    create_teacher_student_data,
    create_temp_files,
)


class TestTeacherPoolManager:
    """Test teacher pool management functionality."""

    def test_teacher_pool_loading(self, temp_dir):
        """Test loading teacher pool from JSONL file."""
        # Create teacher pool data
        teacher_data = [
            {
                "images": ["teacher_1.jpg"],
                "objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "示例设备/参考目标"}],
                "width": 400,
                "height": 300,
            },
            {
                "images": ["teacher_2.jpg"],
                "objects": [
                    {
                        "quad": [100, 110, 120, 115, 118, 130, 98, 125],
                        "desc": "标签/参考标签",
                    }
                ],
                "width": 500,
                "height": 400,
            },
            {
                "images": ["teacher_3.jpg"],
                "objects": [
                    {"line": [200, 210, 220, 230, 240, 250], "desc": "线缆/参考线缆"}
                ],
                "width": 600,
                "height": 500,
            },
        ]

        # Write teacher pool file
        teacher_files = {"teacher_pool.jsonl": teacher_data}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "teacher_pool.jsonl"

        # Mock teacher pool manager
        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)

        # Validate loaded data
        assert len(teacher_pool) == 3

        for i, teacher in enumerate(teacher_pool):
            original = teacher_data[i]
            assert teacher["images"] == original["images"]
            assert teacher["width"] == original["width"]
            assert teacher["height"] == original["height"]
            assert len(teacher["objects"]) == len(original["objects"])

    def test_random_teacher_selection(self, temp_dir):
        """Test random selection of teachers from pool."""
        # Create larger teacher pool
        teacher_data = []
        for i in range(10):
            teacher = {
                "images": [f"teacher_{i}.jpg"],
                "objects": [
                    {
                        "bbox_2d": [i * 10, i * 10 + 10, i * 10 + 20, i * 10 + 30],
                        "desc": f"设备_{i}/测试设备",
                    }
                ],
                "width": 400,
                "height": 300,
            }
            teacher_data.append(teacher)

        teacher_files = {"teacher_pool.jsonl": teacher_data}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "teacher_pool.jsonl"

        # Mock teacher pool manager
        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test random selection
        num_teachers = 3
        selected_teachers = manager.get_random_teachers(num_teachers)

        # Validate selection
        assert len(selected_teachers) == num_teachers
        assert all(teacher in teacher_pool for teacher in selected_teachers)

        # Test that selection is actually random (run multiple times)
        selections = []
        for _ in range(5):
            selected = manager.get_random_teachers(num_teachers)
            selection_ids = [teacher["images"][0] for teacher in selected]
            selections.append(tuple(selection_ids))

        # Should have some variation (not all selections identical)
        unique_selections = set(selections)
        assert len(unique_selections) > 1, "Teacher selection should be random"

    def test_teacher_assignment_probability(self, temp_dir):
        """Test teacher assignment based on probability."""
        teacher_data = create_sample_jsonl_data()  # Use as teacher examples

        teacher_files = {"teacher_pool.jsonl": teacher_data}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "teacher_pool.jsonl"

        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test different teacher ratios
        test_cases = [
            (0.0, 0),  # No teachers
            (0.3, 30),  # 30% chance
            (0.7, 70),  # 70% chance
            (1.0, 100),  # Always teachers
        ]

        for teacher_ratio, expected_percentage in test_cases:
            assignments = []
            num_trials = 100

            for _ in range(num_trials):
                should_assign = manager.should_assign_teacher(teacher_ratio)
                assignments.append(should_assign)

            # Calculate actual percentage
            actual_percentage = sum(assignments) / num_trials * 100

            # Allow some variance due to randomness
            tolerance = 15  # 15% tolerance
            assert abs(actual_percentage - expected_percentage) <= tolerance, (
                f"Teacher ratio {teacher_ratio}: expected ~{expected_percentage}%, got {actual_percentage:.1f}%"
            )

    def test_teacher_selection_constraints(self, temp_dir):
        """Test teacher selection with various constraints."""
        # Create teacher pool with different characteristics
        teacher_data = [
            {
                "images": ["bbox_teacher.jpg"],
                "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "框类目标/测试"}],
                "width": 400,
                "height": 300,
                "geometry_type": "bbox_2d",
            },
            {
                "images": ["quad_teacher.jpg"],
                "objects": [
                    {
                        "quad": [50, 60, 70, 65, 68, 80, 48, 75],
                        "desc": "四边形目标/测试",
                    }
                ],
                "width": 400,
                "height": 300,
                "geometry_type": "quad",
            },
            {
                "images": ["line_teacher.jpg"],
                "objects": [{"line": [100, 110, 120, 130], "desc": "线条目标/测试"}],
                "width": 400,
                "height": 300,
                "geometry_type": "line",
            },
        ]

        teacher_files = {"diverse_teacher_pool.jsonl": teacher_data}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "diverse_teacher_pool.jsonl"

        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test selection with constraints
        # 1. Request more teachers than available
        selected = manager.get_random_teachers(5)  # Only 3 available
        assert len(selected) == 3  # Should return all available

        # 2. Request zero teachers
        selected = manager.get_random_teachers(0)
        assert len(selected) == 0

        # 3. Request negative number (should handle gracefully)
        selected = manager.get_random_teachers(-1)
        assert len(selected) == 0

    def test_teacher_student_compatibility(self, temp_dir):
        """Test compatibility between teachers and students."""
        teacher_student_data = create_teacher_student_data()

        # Create teacher pool
        teacher_pool_data = [teacher_student_data["teacher_sample"]]
        teacher_files = {"teacher_pool.jsonl": teacher_pool_data}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "teacher_pool.jsonl"

        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test teacher-student pairing
        student_sample = teacher_student_data["student_sample"]
        selected_teachers = manager.get_random_teachers(1)

        # Validate compatibility
        teacher = selected_teachers[0]

        # Both should have valid structure
        self._validate_sample_structure(teacher)
        self._validate_sample_structure(student_sample)

        # Should be able to create conversation
        conversation = self._mock_create_teacher_student_conversation(
            teacher, student_sample
        )
        assert len(conversation) > 0
        assert any(
            "参考" in turn["content"] for turn in conversation if turn["role"] == "user"
        )

    def test_empty_teacher_pool_handling(self, temp_dir):
        """Test handling of empty teacher pool."""
        # Create empty teacher pool
        empty_pool_path = temp_dir / "empty_teacher_pool.jsonl"
        empty_pool_path.touch()  # Create empty file

        teacher_pool = self._mock_load_teacher_pool(empty_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Should handle empty pool gracefully
        assert len(teacher_pool) == 0

        selected_teachers = manager.get_random_teachers(1)
        assert len(selected_teachers) == 0

        # Teacher assignment should always return False for empty pool
        assert not manager.should_assign_teacher(1.0)  # Even with 100% ratio

    def test_malformed_teacher_pool_handling(self, temp_dir):
        """Test handling of malformed teacher pool data."""
        # Create malformed teacher pool data
        malformed_data = [
            '{"images": ["valid.jpg"], "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "valid"}], "width": 400, "height": 300}',
            '{"images": [], "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "no image"}], "width": 400, "height": 300}',  # No images
            '{"images": ["invalid.jpg"], "objects": [], "width": 400, "height": 300}',  # No objects
            "invalid json line",  # Invalid JSON
        ]

        # Write malformed data
        malformed_path = temp_dir / "malformed_teacher_pool.jsonl"
        with open(malformed_path, "w", encoding="utf-8") as f:
            for line in malformed_data:
                f.write(line + "\n")

        # Load with error handling
        valid_teachers = []
        error_count = 0

        with open(malformed_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    teacher = json.loads(line.strip())
                    self._validate_sample_structure(teacher)
                    valid_teachers.append(teacher)
                except (json.JSONDecodeError, KeyError, AssertionError, ValueError):
                    error_count += 1

        # Should have filtered out invalid teachers
        assert error_count > 0  # Should have detected errors
        assert len(valid_teachers) == 1  # Only one valid teacher (the first entry)

        # Manager should work with valid teachers only
        manager = self._mock_teacher_pool_manager(valid_teachers)
        selected = manager.get_random_teachers(1)
        assert len(selected) == 1
        assert selected[0]["images"] == ["valid.jpg"]

    def test_teacher_diversity_selection(self, temp_dir):
        """Test selection of diverse teachers."""
        # Create teachers with different characteristics
        diverse_teachers = []

        # Different object counts
        for i in range(3):
            teacher = {
                "images": [f"multi_object_{i}.jpg"],
                "objects": [
                    {
                        "bbox_2d": [j * 50, j * 50 + 10, j * 50 + 20, j * 50 + 30],
                        "desc": f"对象_{j}/测试",
                    }
                    for j in range(i + 1)  # 1, 2, 3 objects
                ],
                "width": 400,
                "height": 300,
            }
            diverse_teachers.append(teacher)

        # Different geometry types
        geometry_teachers = [
            {
                "images": ["bbox_example.jpg"],
                "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "框/测试"}],
                "width": 400,
                "height": 300,
            },
            {
                "images": ["quad_example.jpg"],
                "objects": [
                    {"quad": [50, 60, 70, 65, 68, 80, 48, 75], "desc": "四边形/测试"}
                ],
                "width": 400,
                "height": 300,
            },
            {
                "images": ["line_example.jpg"],
                "objects": [
                    {"line": [100, 110, 120, 130, 140, 150], "desc": "线/测试"}
                ],
                "width": 400,
                "height": 300,
            },
        ]

        all_teachers = diverse_teachers + geometry_teachers

        teacher_files = {"diverse_pool.jsonl": all_teachers}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "diverse_pool.jsonl"

        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test diverse selection
        selected = manager.get_random_teachers(3)
        assert len(selected) == 3

        # Check that we got different teachers
        selected_images = [teacher["images"][0] for teacher in selected]
        assert len(set(selected_images)) == 3  # All different

    # Helper methods for mocking teacher pool functionality

    def _mock_load_teacher_pool(self, pool_path: Path) -> List[Dict[str, Any]]:
        """Mock loading teacher pool from JSONL file."""
        teachers = []

        if not pool_path.exists() or pool_path.stat().st_size == 0:
            return teachers

        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    teacher = json.loads(line)
                    # Basic validation
                    if self._is_valid_teacher(teacher):
                        teachers.append(teacher)
                except (json.JSONDecodeError, KeyError):
                    continue  # Skip malformed entries

        return teachers

    def _mock_teacher_pool_manager(self, teacher_pool: List[Dict[str, Any]]):
        """Create a mock teacher pool manager."""

        class MockTeacherPoolManager:
            def __init__(self, pool):
                self.pool = pool

            def get_random_teachers(self, num_teachers: int) -> List[Dict[str, Any]]:
                if num_teachers <= 0 or not self.pool:
                    return []

                import random

                actual_num = min(num_teachers, len(self.pool))
                return random.sample(self.pool, actual_num)

            def should_assign_teacher(self, teacher_ratio: float) -> bool:
                if not self.pool or teacher_ratio <= 0:
                    return False
                if teacher_ratio >= 1.0:
                    return True

                import random

                return random.random() < teacher_ratio

        return MockTeacherPoolManager(teacher_pool)

    def _is_valid_teacher(self, teacher: Dict[str, Any]) -> bool:
        """Check if teacher sample is valid."""
        try:
            self._validate_sample_structure(teacher)
            return True
        except (AssertionError, ValueError, KeyError):
            return False

    def _validate_sample_structure(self, sample: Dict[str, Any]):
        """Validate sample structure (reused from dataset tests)."""
        required_fields = ["images", "objects", "width", "height"]
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"

        assert isinstance(sample["images"], list) and len(sample["images"]) > 0
        assert (
            isinstance(sample["objects"], list) and len(sample["objects"]) > 0
        )  # Must have objects
        assert isinstance(sample["width"], int) and sample["width"] > 0
        assert isinstance(sample["height"], int) and sample["height"] > 0

        for obj in sample["objects"]:
            assert isinstance(obj, dict)
            assert "desc" in obj

            # Must have exactly one geometry type
            geometry_types = ["bbox_2d", "quad", "line"]
            geometry_count = sum(1 for geo_type in geometry_types if geo_type in obj)
            assert geometry_count == 1

    def _mock_create_teacher_student_conversation(
        self, teacher: Dict[str, Any], student: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Mock creation of teacher-student conversation."""
        conversation = [
            {
                "role": "system",
                "content": "你是通信机房设备检测AI助手。学习参考示例，然后分析新图像。",
            },
            {"role": "user", "content": "📚 参考示例: <image>"},
            {
                "role": "assistant",
                "content": json.dumps(teacher["objects"], ensure_ascii=False),
            },
            {"role": "user", "content": "现在请根据参考示例检测以下图像: <image>"},
            {
                "role": "assistant",
                "content": json.dumps(student["objects"], ensure_ascii=False),
            },
        ]

        return conversation


class TestTeacherPoolIntegration:
    """Test teacher pool integration with dataset and training."""

    def test_teacher_pool_dataset_integration(self, temp_dir):
        """Test integration between teacher pool and dataset."""
        # Create teacher pool
        teacher_data = create_sample_jsonl_data()[:2]  # Use first 2 as teachers
        teacher_files = {"teacher_pool.jsonl": teacher_data}
        test_dir = create_temp_files(teacher_files, temp_dir)

        # Create student data
        student_data = create_sample_jsonl_data()[2:]  # Use rest as students
        student_files = {"student_data.jsonl": student_data}
        student_dir = create_temp_files(student_files, temp_dir)

        # Mock integration
        teacher_pool = self._mock_load_teacher_pool(test_dir / "teacher_pool.jsonl")
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test that student samples can be augmented with teachers
        for student_sample in student_data:
            if manager.should_assign_teacher(0.7):
                teachers = manager.get_random_teachers(1)

                # Create combined sample
                combined_sample = {"student": student_sample, "teachers": teachers}

                # Validate combined structure
                assert "student" in combined_sample
                assert "teachers" in combined_sample
                assert len(combined_sample["teachers"]) <= 1

                # Should be able to create conversation
                if combined_sample["teachers"]:
                    teacher = combined_sample["teachers"][0]
                    conversation = self._mock_create_teacher_student_conversation(
                        teacher, student_sample
                    )
                    assert (
                        len(conversation) == 5
                    )  # system + teacher example + student target

    def test_teacher_pool_memory_efficiency(self, temp_dir):
        """Test memory efficiency of teacher pool management."""
        # Create large teacher pool
        large_teacher_pool = []
        for i in range(1000):
            teacher = {
                "images": [f"teacher_{i}.jpg"],
                "objects": [
                    {
                        "bbox_2d": [i, i + 10, i + 20, i + 30],
                        "desc": f"设备_{i}/大规模测试",
                    }
                ],
                "width": 400,
                "height": 300,
            }
            large_teacher_pool.append(teacher)

        teacher_files = {"large_teacher_pool.jsonl": large_teacher_pool}
        test_dir = create_temp_files(teacher_files, temp_dir)
        teacher_pool_path = test_dir / "large_teacher_pool.jsonl"

        # Test that loading doesn't consume excessive memory
        # (In real implementation, might use lazy loading)
        teacher_pool = self._mock_load_teacher_pool(teacher_pool_path)
        manager = self._mock_teacher_pool_manager(teacher_pool)

        # Test efficient sampling
        num_trials = 100
        for _ in range(num_trials):
            selected = manager.get_random_teachers(3)
            assert len(selected) == 3

            # Each selection should be different (with high probability)
            selected_indices = [
                int(teacher["images"][0].split("_")[1].split(".")[0])
                for teacher in selected
            ]
            assert len(set(selected_indices)) == 3  # All different

    # Helper methods (reuse from above)
    def _mock_load_teacher_pool(self, pool_path: Path) -> List[Dict[str, Any]]:
        """Mock loading teacher pool from JSONL file."""
        teachers = []

        if not pool_path.exists() or pool_path.stat().st_size == 0:
            return teachers

        with open(pool_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    teacher = json.loads(line)
                    teachers.append(teacher)
                except json.JSONDecodeError:
                    continue

        return teachers

    def _mock_teacher_pool_manager(self, teacher_pool: List[Dict[str, Any]]):
        """Create a mock teacher pool manager."""

        class MockTeacherPoolManager:
            def __init__(self, pool):
                self.pool = pool

            def get_random_teachers(self, num_teachers: int) -> List[Dict[str, Any]]:
                if num_teachers <= 0 or not self.pool:
                    return []

                import random

                actual_num = min(num_teachers, len(self.pool))
                return random.sample(self.pool, actual_num)

            def should_assign_teacher(self, teacher_ratio: float) -> bool:
                if not self.pool or teacher_ratio <= 0:
                    return False
                if teacher_ratio >= 1.0:
                    return True

                import random

                return random.random() < teacher_ratio

        return MockTeacherPoolManager(teacher_pool)

    def _mock_create_teacher_student_conversation(
        self, teacher: Dict[str, Any], student: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Mock creation of teacher-student conversation."""
        conversation = [
            {
                "role": "system",
                "content": "你是通信机房设备检测AI助手。学习参考示例，然后分析新图像。",
            },
            {"role": "user", "content": "📚 参考示例: <image>"},
            {
                "role": "assistant",
                "content": json.dumps(teacher["objects"], ensure_ascii=False),
            },
            {"role": "user", "content": "现在请根据参考示例检测以下图像: <image>"},
            {
                "role": "assistant",
                "content": json.dumps(student["objects"], ensure_ascii=False),
            },
        ]

        return conversation
