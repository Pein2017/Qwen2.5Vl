"""
Tests for dataset loading and processing functionality.

This module tests:
- JSONL data loading and validation
- Sample structure validation
- Image and annotation processing
- Error handling for malformed data
"""

import json
from typing import Any, Dict, List

import pytest

from src_new.tests.fixtures import (
    MockImageProcessor,
    MockTokenizer,
    create_sample_jsonl_data,
    create_temp_files,
)


class TestJSONLDataLoading:
    """Test JSONL data loading functionality."""

    def test_load_valid_jsonl(self, temp_dir):
        """Test loading valid JSONL data."""
        # Create test JSONL data
        jsonl_data = create_sample_jsonl_data()
        jsonl_files = {"test_data.jsonl": jsonl_data}
        test_dir = create_temp_files(jsonl_files, temp_dir)
        jsonl_path = test_dir / "test_data.jsonl"

        # Load JSONL data
        loaded_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                loaded_samples.append(json.loads(line.strip()))

        # Validate loaded data
        assert len(loaded_samples) == len(jsonl_data)

        for original, loaded in zip(jsonl_data, loaded_samples):
            assert loaded["images"] == original["images"]
            assert loaded["width"] == original["width"]
            assert loaded["height"] == original["height"]
            assert len(loaded["objects"]) == len(original["objects"])

    def test_sample_structure_validation(self, temp_dir):
        """Test validation of sample structure."""
        # Create sample with all geometry types
        valid_sample = {
            "images": ["test_image.jpg"],
            "objects": [
                {"bbox_2d": [100, 150, 200, 250], "desc": "测试设备/基础检测目标"},
                {
                    "square": [300, 400, 350, 410, 348, 425, 302, 415],
                    "desc": "标签贴纸/测试标识",
                },
                {
                    "line": [50, 100, 150, 120, 250, 140, 350, 160],
                    "desc": "线缆/测试连接线",
                },
            ],
            "width": 800,
            "height": 600,
        }

        # Validate structure
        self._validate_sample_structure(valid_sample)

    def test_bbox_2d_validation(self, temp_dir):
        """Test bbox_2d coordinate validation."""
        # Valid bbox_2d
        valid_bbox = [100, 150, 200, 250]  # [x1, y1, x2, y2]
        assert len(valid_bbox) == 4
        assert valid_bbox[0] < valid_bbox[2]  # x1 < x2
        assert valid_bbox[1] < valid_bbox[3]  # y1 < y2

        # Test various invalid bbox formats
        invalid_bboxes = [
            [100, 150, 200],  # Too few coordinates
            [100, 150, 200, 250, 300],  # Too many coordinates
            [200, 150, 100, 250],  # x1 > x2
            [100, 250, 200, 150],  # y1 > y2
        ]

        for invalid_bbox in invalid_bboxes:
            with pytest.raises((AssertionError, ValueError)):
                self._validate_bbox_2d(invalid_bbox)

    def test_quad_validation(self, temp_dir):
        """Test quad coordinate validation."""
        # Valid quad (8 coordinates: 4 points, each with x,y)
        valid_quad = [300, 400, 350, 410, 348, 425, 302, 415]
        assert len(valid_quad) == 8
        assert len(valid_quad) % 2 == 0  # Even number for x,y pairs

        # Test invalid quad formats
        invalid_quads = [
            [300, 400, 350, 410, 348, 425],  # Too few coordinates
            [300, 400, 350, 410, 348, 425, 302, 415, 100],  # Odd number
        ]

        for invalid_quad in invalid_quads:
            with pytest.raises((AssertionError, ValueError)):
                self._validate_quad(invalid_quad)

    def test_line_validation(self, temp_dir):
        """Test line coordinate validation."""
        # Valid lines (variable length, but must be even)
        valid_lines = [
            [50, 100, 150, 120],  # 2 points
            [50, 100, 150, 120, 250, 140],  # 3 points
            [50, 100, 150, 120, 250, 140, 350, 160],  # 4 points
            [50, 100, 150, 120, 250, 140, 350, 160, 450, 180],  # 5 points
        ]

        for valid_line in valid_lines:
            assert len(valid_line) >= 4  # At least 2 points
            assert len(valid_line) % 2 == 0  # Even number for x,y pairs

        # Test invalid line formats
        invalid_lines = [
            [50, 100],  # Too few coordinates (need at least 2 points)
            [50, 100, 150],  # Odd number of coordinates
        ]

        for invalid_line in invalid_lines:
            with pytest.raises((AssertionError, ValueError)):
                self._validate_line(invalid_line)

    def test_description_validation(self, temp_dir):
        """Test object description validation."""
        # Valid descriptions
        valid_descriptions = [
            "测试设备/基础检测目标",
            "标签贴纸/GPS信号线标识",
            "线缆/有遮挡,捆扎整齐",
            "螺丝连接点/光纤插头连接点,显示完整,符合要求",
        ]

        for desc in valid_descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
            assert "/" in desc  # Should contain category/description format

    def test_malformed_data_handling(self, temp_dir):
        """Test handling of malformed JSONL data."""
        # Create malformed JSONL data
        malformed_data = [
            '{"images": ["test.jpg"], "objects": [], "width": 800}',  # Missing height
            '{"images": [], "objects": [{"bbox_2d": [100, 200, 300, 400], "desc": "test"}], "width": 800, "height": 600}',  # Empty images
            '{"images": ["test.jpg"], "objects": [{"bbox_2d": [100, 200], "desc": "test"}], "width": 800, "height": 600}',  # Invalid bbox
            "invalid json line",  # Invalid JSON
        ]

        # Write malformed data to file
        malformed_path = temp_dir / "malformed.jsonl"
        with open(malformed_path, "w", encoding="utf-8") as f:
            for line in malformed_data:
                f.write(line + "\n")

        # Test loading with error handling
        valid_samples = []
        error_count = 0

        with open(malformed_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    self._validate_sample_structure(sample)
                    valid_samples.append(sample)
                except (json.JSONDecodeError, KeyError, AssertionError, ValueError):
                    error_count += 1
                    print(f"Error in line {line_num}: skipping malformed sample")

        # Should have errors but continue processing
        assert error_count > 0
        assert len(valid_samples) == 0  # All samples in this test are malformed

    def test_empty_jsonl_handling(self, temp_dir):
        """Test handling of empty JSONL files."""
        # Create empty JSONL file
        empty_path = temp_dir / "empty.jsonl"
        empty_path.touch()

        # Load empty file
        samples = []
        with open(empty_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        assert len(samples) == 0

    def test_large_jsonl_loading(self, temp_dir):
        """Test loading of larger JSONL files."""
        # Create larger dataset
        large_data = []
        for i in range(100):
            sample = {
                "images": [f"test_image_{i}.jpg"],
                "objects": [
                    {
                        "bbox_2d": [i * 10, i * 10 + 50, i * 10 + 100, i * 10 + 150],
                        "desc": f"测试设备_{i}/自动生成目标",
                    }
                ],
                "width": 800,
                "height": 600,
            }
            large_data.append(sample)

        # Write to file
        jsonl_files = {"large_data.jsonl": large_data}
        test_dir = create_temp_files(jsonl_files, temp_dir)
        jsonl_path = test_dir / "large_data.jsonl"

        # Load and validate
        loaded_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                loaded_samples.append(json.loads(line.strip()))

        assert len(loaded_samples) == 100

        # Validate random samples
        for i in [0, 25, 50, 75, 99]:
            sample = loaded_samples[i]
            assert sample["images"] == [f"test_image_{i}.jpg"]
            assert sample["objects"][0]["bbox_2d"] == [
                i * 10,
                i * 10 + 50,
                i * 10 + 100,
                i * 10 + 150,
            ]

    def _validate_sample_structure(self, sample: Dict[str, Any]):
        """Validate the structure of a single sample."""
        # Required fields
        required_fields = ["images", "objects", "width", "height"]
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"

        # Validate types
        assert isinstance(sample["images"], list), "images must be a list"
        assert len(sample["images"]) > 0, "images list cannot be empty"
        assert isinstance(sample["objects"], list), "objects must be a list"
        assert isinstance(sample["width"], int), "width must be an integer"
        assert isinstance(sample["height"], int), "height must be an integer"
        assert sample["width"] > 0, "width must be positive"
        assert sample["height"] > 0, "height must be positive"

        # Validate objects
        for obj in sample["objects"]:
            assert isinstance(obj, dict), "each object must be a dictionary"
            assert "desc" in obj, "each object must have a description"
            assert isinstance(obj["desc"], str), "description must be a string"
            assert len(obj["desc"]) > 0, "description cannot be empty"

            # Must have exactly one geometry type
            geometry_types = ["bbox_2d", "square", "line"]
            geometry_count = sum(1 for geo_type in geometry_types if geo_type in obj)
            assert geometry_count == 1, (
                f"each object must have exactly one geometry type, got {geometry_count}"
            )

            # Validate specific geometry types
            if "bbox_2d" in obj:
                self._validate_bbox_2d(obj["bbox_2d"])
            elif "square" in obj:
                self._validate_square(obj["square"])
            elif "line" in obj:
                self._validate_line(obj["line"])

    def _validate_bbox_2d(self, bbox: List[int]):
        """Validate bbox_2d coordinates."""
        assert isinstance(bbox, list), "bbox_2d must be a list"
        assert len(bbox) == 4, (
            f"bbox_2d must have exactly 4 coordinates, got {len(bbox)}"
        )

        x1, y1, x2, y2 = bbox
        assert isinstance(x1, int), "bbox coordinates must be integers"
        assert isinstance(y1, int), "bbox coordinates must be integers"
        assert isinstance(x2, int), "bbox coordinates must be integers"
        assert isinstance(y2, int), "bbox coordinates must be integers"

        assert x1 < x2, f"x1 ({x1}) must be less than x2 ({x2})"
        assert y1 < y2, f"y1 ({y1}) must be less than y2 ({y2})"
        assert x1 >= 0 and y1 >= 0, "coordinates must be non-negative"

    def _validate_square(self, square: List[int]):
        """Validate square coordinates."""
        assert isinstance(square, list), "square must be a list"
        assert len(square) == 8, (
            f"square must have exactly 8 coordinates, got {len(square)}"
        )
        assert all(isinstance(coord, int) for coord in square), (
            "square coordinates must be integers"
        )
        assert all(coord >= 0 for coord in square), "coordinates must be non-negative"

    def _validate_line(self, line: List[int]):
        """Validate line coordinates."""
        assert isinstance(line, list), "line must be a list"
        assert len(line) >= 4, f"line must have at least 4 coordinates, got {len(line)}"
        assert len(line) % 2 == 0, (
            f"line must have even number of coordinates, got {len(line)}"
        )
        assert all(isinstance(coord, int) for coord in line), (
            "line coordinates must be integers"
        )
        assert all(coord >= 0 for coord in line), "coordinates must be non-negative"


class TestDatasetIntegration:
    """Test dataset integration with tokenizer and image processor."""

    def test_dataset_with_tokenizer(self, temp_dir):
        """Test dataset integration with tokenizer."""
        # Create test data
        jsonl_data = create_sample_jsonl_data()
        jsonl_files = {"test_data.jsonl": jsonl_data}
        test_dir = create_temp_files(jsonl_files, temp_dir)

        # Create mock components
        tokenizer = MockTokenizer()
        image_processor = MockImageProcessor()

        # Mock dataset processing
        for sample in jsonl_data:
            # Test that sample can be processed
            assert "images" in sample
            assert "objects" in sample

            # Mock conversation building
            conversation = [
                {"role": "system", "content": "你是通信机房设备检测AI助手。"},
                {"role": "user", "content": "请检测图像中的设备和部件: <image>"},
                {
                    "role": "assistant",
                    "content": json.dumps(sample["objects"], ensure_ascii=False),
                },
            ]

            # Mock tokenization
            chat_text = tokenizer.apply_chat_template(conversation)
            token_ids = tokenizer.encode(chat_text)

            # Mock image processing
            processed_images = image_processor.preprocess(sample["images"])

            # Validate processing results
            assert isinstance(token_ids, list)
            assert len(token_ids) > 0
            assert "pixel_values" in processed_images
            assert "image_grid_thw" in processed_images

    def test_dataset_geometry_processing(self, temp_dir):
        """Test processing of different geometry types."""
        # Create sample with all geometry types
        mixed_sample = {
            "images": ["mixed_test.jpg"],
            "objects": [
                {"bbox_2d": [100, 150, 200, 250], "desc": "矩形目标/基础框"},
                {
                    "square": [300, 400, 350, 410, 348, 425, 302, 415],
                    "desc": "四边形目标/倾斜框",
                },
                {"line": [50, 100, 150, 120, 250, 140], "desc": "线条目标/连接线"},
            ],
            "width": 800,
            "height": 600,
        }

        # Test geometry processing
        for obj in mixed_sample["objects"]:
            if "bbox_2d" in obj:
                # Mock bbox processing
                bbox = obj["bbox_2d"]
                assert len(bbox) == 4
                bbox_text = f"[{','.join(map(str, bbox))}]"
                assert "100,150,200,250" in bbox_text or similar_format(bbox_text, bbox)

            elif "square" in obj:
                # Mock square processing
                square = obj["square"]
                assert len(square) == 8
                square_text = f"[{','.join(map(str, square))}]"
                assert len(square_text) > 10  # Should have reasonable length

            elif "line" in obj:
                # Mock line processing
                line = obj["line"]
                assert len(line) >= 4
                assert len(line) % 2 == 0
                line_text = f"[{','.join(map(str, line))}]"
                assert len(line_text) > 5  # Should have reasonable length


def similar_format(text: str, coords: List[int]) -> bool:
    """Helper to check if text contains coordinates in similar format."""
    coord_strs = [str(c) for c in coords]
    return all(coord_str in text for coord_str in coord_strs)
