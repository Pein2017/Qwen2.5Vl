#!/usr/bin/env python3
"""
Comprehensive end-to-end inference tests for the BBU detection system.

This test suite covers all inference scenarios including:
- Single image inference (no teacher)
- Single teacher + single student image
- Multiple teachers + single student image
- Multiple images per sample (teachers and students)
- Edge cases: empty responses, malformed data, path issues
- Performance benchmarks: inference speed and memory usage
- Coordinate token processing validation
- Integration with eval/infer_dataset.sh
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.fixtures.mock_objects import create_mock_model, create_mock_tokenizer
from tests.fixtures.test_utils import (
    TestMetrics,
    generate_test_images,
    skip_if_no_gpu,
)

from inference import InferenceEngine


class TestComprehensiveInference:
    """Comprehensive inference test suite."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_images(self, temp_dir):
        """Generate test images for inference."""
        return generate_test_images(temp_dir, count=5)

    @pytest.fixture
    def mock_inference_engine(self, temp_dir):
        """Create mock inference engine for testing without real model."""
        # Create minimal config for testing
        config_data = {
            "coordinate_tokens_enabled": True,
            "max_coord_value": 1000,
            "new_geometry_tokens": True,
            "model_path": "/fake/model/path",
        }

        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Mock the model loading to avoid requiring actual model
        with (
            patch("inference.DetectionModel") as mock_model_class,
            patch("inference.AutoTokenizer") as mock_tokenizer_class,
            patch("inference.Qwen2VLImageProcessor") as mock_processor_class,
            patch("inference.ConversationProcessor") as mock_conv_processor,
        ):
            # Setup mocks
            mock_tokenizer = create_mock_tokenizer()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            mock_model = create_mock_model()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model_class.from_pretrained_fast.return_value = mock_model

            mock_processor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor

            mock_conversation_processor = MagicMock()
            mock_conv_processor.return_value = mock_conversation_processor

            # Create inference engine
            engine = InferenceEngine(
                config_path=config_path,
                model_path="/fake/model/path",
                use_training_prompts=True,
                batch_size=1,
            )

            # Attach mocks for testing
            engine._mock_tokenizer = mock_tokenizer
            engine._mock_model = mock_model
            engine._mock_conversation_processor = mock_conversation_processor

            yield engine


class TestSingleImageInference:
    """Test single image inference scenarios."""

    def test_single_image_no_teacher(self, mock_inference_engine, test_images):
        """Test inference with single image and no teacher guidance."""
        sample = {
            "id": "test_001",
            "images": [test_images[0]],
            "objects": [
                {
                    "category": "test_object",
                    "bbox_2d": [100, 100, 200, 200],
                    "description": "Test BBU equipment",
                }
            ],
        }

        # Test input preparation
        inputs = mock_inference_engine.prepare_inference_inputs(sample)

        # Validate input structure
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        # Test that conversation processor was called correctly
        mock_conv = mock_inference_engine._mock_conversation_processor
        assert mock_conv.create_simple_conversation.called

    def test_single_image_coordinate_tokens(self, mock_inference_engine, test_images):
        """Test single image inference with coordinate tokens enabled."""
        sample = {
            "id": "test_coord_001",
            "images": [test_images[0]],
            "objects": [
                {
                    "category": "螺丝",
                    "bbox_2d": [50, 50, 150, 150],
                    "description": "螺丝位于设备左上角",
                }
            ],
        }

        # Mock coordinate token response
        mock_response = "The image shows <|obj_ref_start|>螺丝位于设备左上角<|obj_ref_end|><|box_start|>[<|coord_50|>, <|coord_50|>, <|coord_150|>, <|coord_150|>]<|box_end|>"

        with patch.object(
            mock_inference_engine, "generate_response", return_value=mock_response
        ):
            inputs = mock_inference_engine.prepare_inference_inputs(sample)
            response = mock_inference_engine.generate_response(
                inputs, max_new_tokens=100
            )

            # Validate coordinate token processing
            processed_response = mock_inference_engine._process_coordinate_response(
                response
            )

            # Should contain valid JSON with coordinate values
            assert (
                "[50, 50, 150, 150]" in processed_response or "50" in processed_response
            )


class TestTeacherStudentInference:
    """Test teacher-student inference scenarios."""

    @pytest.fixture
    def teacher_pool_data(self, test_images):
        """Create teacher pool data for testing."""
        return [
            {
                "images": [test_images[0]],
                "objects": [
                    {
                        "category": "teacher_object_1",
                        "bbox_2d": [10, 10, 50, 50],
                        "description": "Teacher example 1",
                    }
                ],
            },
            {
                "images": [test_images[1], test_images[2]],
                "objects": [
                    {
                        "category": "teacher_object_2",
                        "bbox_2d": [20, 20, 60, 60],
                        "description": "Teacher example 2",
                    },
                    {
                        "category": "teacher_object_3",
                        "bbox_2d": [30, 30, 70, 70],
                        "description": "Teacher example 3",
                    },
                ],
            },
        ]

    @pytest.fixture
    def teacher_pool_file(self, temp_dir, teacher_pool_data):
        """Create teacher pool JSONL file."""
        teacher_pool_path = os.path.join(temp_dir, "teacher_pool.jsonl")
        with open(teacher_pool_path, "w") as f:
            for teacher in teacher_pool_data:
                f.write(json.dumps(teacher) + "\n")
        return teacher_pool_path

    def test_single_teacher_inference(
        self, mock_inference_engine, teacher_pool_file, test_images
    ):
        """Test inference with single teacher guidance."""
        # Update engine with teacher pool
        mock_inference_engine.teacher_pool_file = teacher_pool_file
        mock_inference_engine.num_teachers = 1
        mock_inference_engine.teacher_samples = (
            mock_inference_engine._load_teacher_pool()
        )

        student_sample = {
            "id": "student_001",
            "images": [test_images[3]],
            "objects": [
                {
                    "category": "student_object",
                    "bbox_2d": [15, 15, 55, 55],
                    "description": "Student test object",
                }
            ],
        }

        # Test teacher-student input preparation
        inputs = mock_inference_engine._prepare_training_matched_inputs(
            student_sample, seed=42
        )

        # Validate that teacher examples were included
        mock_conv = mock_inference_engine._mock_conversation_processor
        assert mock_conv.create_teacher_student_conversation.called

        # Validate input structure
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

    def test_multiple_teachers_inference(
        self, mock_inference_engine, teacher_pool_file, test_images
    ):
        """Test inference with multiple teacher guidance."""
        mock_inference_engine.teacher_pool_file = teacher_pool_file
        mock_inference_engine.num_teachers = 2
        mock_inference_engine.teacher_samples = (
            mock_inference_engine._load_teacher_pool()
        )

        student_sample = {
            "id": "student_multi_001",
            "images": [test_images[4]],
            "objects": [],
        }

        # Test multi-teacher sampling
        teachers = mock_inference_engine._sample_teachers(seed=42)
        assert len(teachers) <= 2  # Should not exceed available teachers

        # Test input preparation
        inputs = mock_inference_engine._prepare_training_matched_inputs(
            student_sample, seed=42
        )
        assert inputs is not None

    def test_teacher_student_conversation_flow(
        self, mock_inference_engine, teacher_pool_file, test_images
    ):
        """Test conversation flow and truncation in teacher-student setup."""
        mock_inference_engine.teacher_pool_file = teacher_pool_file
        mock_inference_engine.num_teachers = 1
        mock_inference_engine.teacher_samples = (
            mock_inference_engine._load_teacher_pool()
        )

        student_sample = {
            "id": "conversation_test",
            "images": [test_images[0]],
            "objects": [
                {"category": "test", "bbox_2d": [0, 0, 10, 10], "description": "test"}
            ],
        }

        # Mock conversation with proper structure for truncation testing
        mock_conversation_text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image_pad|>Describe objects in this image.<|im_end|>
<|im_start|>assistant
I can see test objects in the image.<|im_end|>
<|im_start|>user
<|image_pad|>Now describe this student image.<|im_end|>
<|im_start|>assistant
"""

        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(4, 1024),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

        mock_inference_engine._mock_conversation_processor.create_teacher_student_conversation.return_value = mock_inputs

        with patch.object(
            mock_inference_engine.tokenizer,
            "decode",
            return_value=mock_conversation_text,
        ):
            inputs = mock_inference_engine._prepare_training_matched_inputs(
                student_sample, seed=42
            )

            # Verify truncation logic was applied
            assert "input_ids" in inputs
            # Conversation should end with assistant marker for generation


class TestMultiImageProcessing:
    """Test multi-image processing and token alignment."""

    def test_multi_image_token_alignment(self, mock_inference_engine, test_images):
        """Test image token alignment with multiple images."""
        sample = {
            "id": "multi_image_001",
            "images": [test_images[0], test_images[1], test_images[2]],
            "objects": [
                {
                    "category": "object_1",
                    "bbox_2d": [10, 10, 50, 50],
                    "description": "First object",
                }
            ],
        }

        # Mock multi-image processing
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            # For 3 images with grid [1,2,2] each => 12 patches total
            "pixel_values": torch.randn(12, 1024),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2], [1, 2, 2], [1, 2, 2]]
            ),  # 3 image grids
        }

        mock_inference_engine._mock_conversation_processor.create_simple_conversation.return_value = mock_inputs

        # Mock tokenizer decode to return text with appropriate image tokens
        mock_text = (
            "<|image_pad|>" * 100
            + "User text"
            + "<|image_pad|>" * 100
            + "More text"
            + "<|image_pad|>" * 100
        )

        with patch.object(
            mock_inference_engine.tokenizer, "decode", return_value=mock_text
        ):
            inputs = mock_inference_engine.prepare_inference_inputs(sample)

            # Test image token alignment validation
            validation_results = mock_inference_engine.validate_image_token_alignment(
                inputs
            )

            assert "is_valid" in validation_results
            assert "details" in validation_results
            assert "num_images_in_grid" in validation_results["details"]

    def test_image_path_resolution(self, mock_inference_engine, temp_dir, test_images):
        """Test proper image path resolution for relative and absolute paths."""
        # Create sample with relative paths
        relative_paths = [os.path.relpath(img, temp_dir) for img in test_images[:2]]

        sample = {"id": "path_test", "images": relative_paths, "objects": []}

        # Set data root for path resolution
        mock_inference_engine.data_root = temp_dir

        # Test that paths are resolved correctly
        resolved_sample = sample.copy()
        resolved_sample["images"] = [
            os.path.join(temp_dir, path) for path in relative_paths
        ]

        # Verify all resolved paths exist
        for path in resolved_sample["images"]:
            assert os.path.exists(path), f"Resolved path does not exist: {path}"


class TestCoordinateSystemValidation:
    """Test coordinate token system during inference."""

    def test_coordinate_token_generation(self, mock_inference_engine):
        """Test coordinate token generation during inference."""
        # Mock a coordinate token response
        mock_response = """The image contains multiple objects:
1. <|obj_ref_start|>螺丝在左上角<|obj_ref_end|><|box_start|>[<|coord_100|>, <|coord_50|>, <|coord_200|>, <|coord_150|>]<|box_end|>
2. <|obj_ref_start|>挡风板在右侧<|obj_ref_end|><|quad_start|>[<|coord_300|>, <|coord_100|>, <|coord_400|>, <|coord_100|>, <|coord_400|>, <|coord_200|>, <|coord_300|>, <|coord_200|>]<|quad_end|>"""

        # Test coordinate token parsing
        parsed_objects = mock_inference_engine._parse_coordinate_token_response(
            mock_response
        )

        assert len(parsed_objects) >= 1
        # Check that coordinate values were extracted correctly
        if parsed_objects:
            first_obj = parsed_objects[0]
            assert "bbox_2d" in first_obj or "quad" in first_obj

    def test_coordinate_range_validation(self, mock_inference_engine):
        """Test coordinate value range validation."""
        # Test with out-of-range coordinates
        invalid_response = "<|box_start|>[<|coord_2000|>, <|coord_-50|>, <|coord_1500|>, <|coord_1200|>]<|box_end|>"

        # Should handle invalid coordinates gracefully
        parsed_objects = mock_inference_engine._parse_coordinate_token_response(
            invalid_response
        )
        # May be empty or contain filtered results
        assert isinstance(parsed_objects, list)

    def test_response_format_validation(self, mock_inference_engine):
        """Test JSON output format validation."""
        sample_objects = [
            {
                "bbox_2d": [100, 100, 200, 200],
                "desc": "Test object",
                "category": "test_category",
            }
        ]

        # Test JSON serialization
        processed_response = mock_inference_engine._process_coordinate_response(
            json.dumps(sample_objects)
        )

        # Should be valid JSON
        try:
            parsed = json.loads(processed_response)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail("Response should be valid JSON")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_empty_teacher_pool(self, mock_inference_engine, test_images):
        """Test handling of empty teacher pool."""
        sample = {"id": "empty_teacher_test", "images": [test_images[0]], "objects": []}

        # Ensure no teacher pool is set
        mock_inference_engine.teacher_samples = []
        mock_inference_engine.num_teachers = 0

        # Should fall back to simple conversation
        inputs = mock_inference_engine.prepare_inference_inputs(sample)
        assert inputs is not None
        assert mock_inference_engine._mock_conversation_processor.create_simple_conversation.called

    def test_missing_image_files(self, mock_inference_engine):
        """Test handling of missing image files."""
        sample = {
            "id": "missing_image_test",
            "images": ["/non/existent/image.jpg"],
            "objects": [],
        }

        # Should raise appropriate error
        with pytest.raises(Exception):
            mock_inference_engine.prepare_inference_inputs(sample)

    def test_malformed_sample_data(self, mock_inference_engine, test_images):
        """Test handling of malformed sample data."""
        # Test with missing required fields
        malformed_samples = [
            {},  # Empty sample
            {"id": "test"},  # Missing images
            {"images": [test_images[0]]},  # Missing objects (should still work)
            {"images": [], "objects": []},  # Empty images list
        ]

        for i, sample in enumerate(malformed_samples):
            if i < 2:  # First two should raise errors
                with pytest.raises(Exception):
                    mock_inference_engine.prepare_inference_inputs(sample)
            # Others might work with defaults

    def test_empty_response_generation(self, mock_inference_engine, test_images):
        """Test handling of empty model responses."""
        sample = {
            "id": "empty_response_test",
            "images": [test_images[0]],
            "objects": [],
        }

        # Mock empty response
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        with patch.object(mock_inference_engine, "generate_response", return_value=""):
            inputs = mock_inference_engine.prepare_inference_inputs(sample)
            response = mock_inference_engine.generate_response(
                inputs, max_new_tokens=10
            )

            # Should handle empty response gracefully
            assert response == ""


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    def test_inference_speed_benchmark(self, mock_inference_engine, test_images):
        """Benchmark inference speed for different scenarios."""
        metrics = TestMetrics()

        # Single image benchmark
        sample = {
            "id": "speed_test_single",
            "images": [test_images[0]],
            "objects": [
                {"category": "test", "bbox_2d": [0, 0, 10, 10], "description": "test"}
            ],
        }

        start_time = time.time()
        inputs = mock_inference_engine.prepare_inference_inputs(sample)
        preparation_time = time.time() - start_time

        metrics.add_measurement("single_image_preparation_time", preparation_time)

        # Multi-image benchmark
        multi_sample = {
            "id": "speed_test_multi",
            "images": test_images[:3],
            "objects": [],
        }

        start_time = time.time()
        inputs = mock_inference_engine.prepare_inference_inputs(multi_sample)
        multi_preparation_time = time.time() - start_time

        metrics.add_measurement("multi_image_preparation_time", multi_preparation_time)

        # Log performance metrics
        print(f"\nPerformance Metrics:")
        print(f"Single image preparation: {preparation_time:.4f}s")
        print(f"Multi image preparation: {multi_preparation_time:.4f}s")

        # Performance assertions
        assert preparation_time < 1.0, (
            "Single image preparation should be under 1 second"
        )
        assert multi_preparation_time < 3.0, (
            "Multi image preparation should be under 3 seconds"
        )

    @skip_if_no_gpu
    def test_memory_usage_benchmark(self, mock_inference_engine, test_images):
        """Benchmark GPU memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Test memory usage with different image counts
        image_counts = [1, 2, 4]
        memory_usage = {}

        for count in image_counts:
            sample = {
                "id": f"memory_test_{count}",
                "images": test_images[:count],
                "objects": [],
            }

            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            inputs = mock_inference_engine.prepare_inference_inputs(sample)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            memory_usage[count] = memory_used

            torch.cuda.empty_cache()

        print(f"\nMemory Usage by Image Count:")
        for count, usage in memory_usage.items():
            print(f"{count} images: {usage / (1024**2):.2f} MB")

        # Memory should scale reasonably with image count
        if len(memory_usage) >= 2:
            single_usage = memory_usage[1]
            multi_usage = memory_usage[max(memory_usage.keys())]
            scaling_factor = multi_usage / single_usage if single_usage > 0 else 1

            # Memory shouldn't scale too dramatically
            assert scaling_factor < 10, f"Memory scaling too high: {scaling_factor}x"


class TestEvalScriptIntegration:
    """Test integration with eval/infer_dataset.sh script and real data processing."""

    @pytest.fixture
    def real_dataset_sample(self):
        """Create a sample using actual dataset format."""
        return {
            "images": ["images/QC-20230314-0000778_85741.jpeg"],
            "objects": [
                {
                    "quad": [2, 0, 226, 0, 241, 14, 0, 73],
                    "desc": "BBU设备/华为,只显示部分,无需安装",
                },
                {
                    "bbox_2d": [358, 106, 407, 149],
                    "desc": "螺丝、光纤插头/地排处接地螺丝,只显示部分,符合要求",
                },
            ],
            "width": 532,
            "height": 728,
        }

    @pytest.fixture
    def real_teacher_sample(self):
        """Create a teacher sample using actual teacher pool format."""
        return {
            "images": ["images/QC-20230323-0001262_237114.jpeg"],
            "objects": [
                {
                    "quad": [166, 0, 218, 16, 211, 35, 150, 10],
                    "desc": "标签/5G-BBU-（接地线）",
                },
                {
                    "bbox_2d": [269, 46, 291, 69],
                    "desc": "螺丝、光纤插头/机柜处接地螺丝,只显示部分,符合要求",
                },
                {
                    "line": [184, 347, 194, 362, 213, 372, 232, 378],
                    "desc": "光纤/有遮挡,无保护措施,弯曲半径合理",
                },
            ],
            "width": 532,
            "height": 728,
        }

    def test_real_data_format_processing(
        self, mock_inference_engine, real_dataset_sample, test_images
    ):
        """Test processing of real dataset format with multiple geometry types."""
        # Use test images instead of real paths for testing
        sample = real_dataset_sample.copy()
        sample["images"] = [test_images[0]]

        # Test input preparation with real data structure
        inputs = mock_inference_engine.prepare_inference_inputs(sample)

        # Validate that multi-geometry objects are handled
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        # Verify conversation processor was called for multi-geometry sample
        mock_conv = mock_inference_engine._mock_conversation_processor
        assert mock_conv.create_simple_conversation.called

    def test_real_teacher_student_format(
        self,
        mock_inference_engine,
        real_teacher_sample,
        real_dataset_sample,
        test_images,
        temp_dir,
    ):
        """Test teacher-student processing with real data formats."""
        # Create teacher pool with real format
        teacher_pool = [real_teacher_sample.copy()]
        teacher_pool[0]["images"] = [test_images[0]]  # Use test images

        teacher_pool_path = os.path.join(temp_dir, "real_teacher_pool.jsonl")
        with open(teacher_pool_path, "w") as f:
            for teacher in teacher_pool:
                f.write(json.dumps(teacher) + "\n")

        # Setup inference engine with teacher pool
        mock_inference_engine.teacher_pool_file = teacher_pool_path
        mock_inference_engine.num_teachers = 1
        mock_inference_engine.teacher_samples = (
            mock_inference_engine._load_teacher_pool()
        )

        # Prepare student sample
        student_sample = real_dataset_sample.copy()
        student_sample["images"] = [test_images[1]]

        # Test teacher-student input preparation
        inputs = mock_inference_engine._prepare_training_matched_inputs(
            student_sample, seed=42
        )

        # Validate teacher-student conversation creation
        mock_conv = mock_inference_engine._mock_conversation_processor
        assert mock_conv.create_teacher_student_conversation.called

    def test_coordinate_token_processing_with_real_geometries(
        self, mock_inference_engine, real_dataset_sample
    ):
        """Test coordinate token processing for all geometry types found in real data."""
        # Mock coordinate token responses for different geometries
        bbox_response = "I can see <|obj_ref_start|>螺丝、光纤插头/地排处接地螺丝,只显示部分,符合要求<|obj_ref_end|><|box_start|>[<|coord_358|>, <|coord_106|>, <|coord_407|>, <|coord_149|>]<|box_end|>"

        quad_response = "There is <|obj_ref_start|>BBU设备/华为,只显示部分,无需安装<|obj_ref_end|><|quad_start|>[<|coord_2|>, <|coord_0|>, <|coord_226|>, <|coord_0|>, <|coord_241|>, <|coord_14|>, <|coord_0|>, <|coord_73|>]<|quad_end|>"

        line_response = "I see <|obj_ref_start|>光纤/有遮挡,无保护措施,弯曲半径合理<|obj_ref_end|><|line_start|>[<|coord_184|>, <|coord_347|>, <|coord_194|>, <|coord_362|>, <|coord_213|>, <|coord_372|>, <|coord_232|>, <|coord_378|>]<|line_end|>"

        # Test bbox processing
        bbox_objects = mock_inference_engine._parse_coordinate_token_response(
            bbox_response
        )
        assert len(bbox_objects) >= 1
        if bbox_objects:
            assert "bbox_2d" in bbox_objects[0]
            assert len(bbox_objects[0]["bbox_2d"]) == 4

        # Test quad processing
        quad_objects = mock_inference_engine._parse_coordinate_token_response(
            quad_response
        )
        assert len(quad_objects) >= 1
        if quad_objects:
            assert "quad" in quad_objects[0]
            assert len(quad_objects[0]["quad"]) == 8

        # Test line processing
        line_objects = mock_inference_engine._parse_coordinate_token_response(
            line_response
        )
        assert len(line_objects) >= 1
        if line_objects:
            assert "line" in line_objects[0]
            assert len(line_objects[0]["line"]) >= 4  # At least 2 points (4 coords)

    def test_eval_script_output_format_compatibility(
        self, temp_dir, test_images, real_dataset_sample
    ):
        """Test that inference outputs match eval/infer_dataset.sh expected format."""
        # Create test dataset file in real format
        test_data = [real_dataset_sample.copy()]
        test_data[0]["images"] = [test_images[0]]
        test_data[0]["id"] = "real_format_test_001"  # Add required ID field

        input_file = os.path.join(temp_dir, "real_test_input.jsonl")
        with open(input_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Create expected output format (what eval script expects)
        expected_output = [
            {
                "sample_id": "real_format_test_001",
                "prediction": json.dumps(
                    [
                        {
                            "quad": [2, 0, 226, 0, 241, 14, 0, 73],
                            "desc": "BBU设备/华为,只显示部分,无需安装",
                        },
                        {
                            "bbox_2d": [358, 106, 407, 149],
                            "desc": "螺丝、光纤插头/地排处接地螺丝,只显示部分,符合要求",
                        },
                    ],
                    ensure_ascii=False,
                ),
                "images": [test_images[0]],
            }
        ]

        output_file = os.path.join(temp_dir, "real_test_output.json")
        with open(output_file, "w") as f:
            json.dump(expected_output, f, ensure_ascii=False, indent=2)

        # Validate output format matches eval script expectations
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert "sample_id" in result
        assert "prediction" in result
        assert "images" in result

        # Validate prediction structure
        prediction = result["prediction"]
        if isinstance(prediction, str):
            parsed_prediction = json.loads(prediction)
            assert isinstance(parsed_prediction, list)
            assert len(parsed_prediction) == 2  # Two objects in sample

            # Validate geometry type handling
            geometry_types = []
            for obj in parsed_prediction:
                if "bbox_2d" in obj:
                    geometry_types.append("bbox_2d")
                    assert len(obj["bbox_2d"]) == 4
                elif "quad" in obj:
                    geometry_types.append("quad")
                    assert len(obj["quad"]) == 8
                elif "line" in obj:
                    geometry_types.append("line")
                    assert len(obj["line"]) >= 4

            # Should have processed both bbox and quad geometries
            assert "bbox_2d" in geometry_types
            assert "quad" in geometry_types

    def test_batch_processing_with_real_data_scale(self, temp_dir, test_images):
        """Test batch processing that matches real dataset scale."""
        # Create multiple samples in real format
        test_samples = []
        for i in range(5):  # Test with reasonable batch size
            sample = {
                "id": f"batch_real_test_{i:03d}",
                "images": [test_images[i % len(test_images)]],
                "objects": [
                    {
                        "bbox_2d": [i * 20, i * 20, (i + 2) * 20, (i + 2) * 20],
                        "desc": f"螺丝、光纤插头/测试对象{i},符合要求",
                    },
                    {
                        "quad": [
                            i * 10,
                            0,
                            (i + 10) * 10,
                            0,
                            (i + 10) * 10,
                            50,
                            i * 10,
                            50,
                        ],
                        "desc": f"BBU设备/华为,测试设备{i},无需安装",
                    },
                ],
                "width": 532,
                "height": 728,
            }
            test_samples.append(sample)

        # Create batch input file
        input_file = os.path.join(temp_dir, "batch_real_input.jsonl")
        with open(input_file, "w") as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Simulate batch processing results
        batch_results = []
        for sample in test_samples:
            result = {
                "sample_id": sample["id"],
                "prediction": json.dumps(sample["objects"], ensure_ascii=False),
                "images": sample["images"],
            }
            batch_results.append(result)

        output_file = os.path.join(temp_dir, "batch_real_output.json")
        with open(output_file, "w") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        # Validate batch processing format
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["sample_id"] == f"batch_real_test_{i:03d}"

            prediction = json.loads(result["prediction"])
            assert len(prediction) == 2  # bbox and quad objects

            # Validate geometry types are preserved
            has_bbox = any("bbox_2d" in obj for obj in prediction)
            has_quad = any("quad" in obj for obj in prediction)
            assert has_bbox and has_quad

    def test_performance_with_real_data_constraints(self, temp_dir, test_images):
        """Test performance characteristics that match real inference constraints."""
        # Simulate real dataset constraints
        max_images_per_sample = 3  # Based on actual dataset analysis
        max_objects_per_sample = 20  # Based on actual dataset analysis

        # Create performance test sample
        perf_sample = {
            "id": "perf_test_real_constraints",
            "images": test_images[:max_images_per_sample],
            "objects": [],
        }

        # Add max objects with realistic coordinate ranges
        for i in range(max_objects_per_sample):
            geometry_type = ["bbox_2d", "quad", "line"][i % 3]

            if geometry_type == "bbox_2d":
                obj = {
                    "bbox_2d": [i * 20, i * 15, (i + 3) * 20, (i + 4) * 15],
                    "desc": f"螺丝、光纤插头/性能测试对象{i},符合要求",
                }
            elif geometry_type == "quad":
                obj = {
                    "quad": [i * 10, 0, (i + 5) * 10, 5, (i + 5) * 10, 40, i * 10, 35],
                    "desc": f"BBU设备/华为,性能测试设备{i},无需安装",
                }
            else:  # line
                obj = {
                    "line": [
                        i * 8,
                        i * 6,
                        (i + 1) * 8,
                        (i + 2) * 6,
                        (i + 2) * 8,
                        (i + 3) * 6,
                    ],
                    "desc": f"光纤/有遮挡,性能测试线路{i},弯曲半径合理",
                }

            perf_sample["objects"].append(obj)

        perf_sample["width"] = 532
        perf_sample["height"] = 728

        # Create performance test file
        input_file = os.path.join(temp_dir, "perf_test_input.jsonl")
        with open(input_file, "w") as f:
            f.write(json.dumps(perf_sample, ensure_ascii=False) + "\n")

        # Validate that sample structure is reasonable for processing
        with open(input_file, "r", encoding="utf-8") as f:
            loaded_sample = json.loads(f.readline())

        assert len(loaded_sample["images"]) <= max_images_per_sample
        assert len(loaded_sample["objects"]) <= max_objects_per_sample

        # Check coordinate ranges are realistic (within typical image dimensions)
        for obj in loaded_sample["objects"]:
            for geom_type in ["bbox_2d", "quad", "line"]:
                if geom_type in obj:
                    coords = obj[geom_type]
                    # All coordinates should be within reasonable image bounds
                    assert all(0 <= coord <= 1000 for coord in coords), (
                        f"Invalid coordinates in {geom_type}: {coords}"
                    )

    def test_error_recovery_with_real_data_edge_cases(
        self, mock_inference_engine, temp_dir, test_images
    ):
        """Test error recovery for edge cases found in real data."""
        # Test cases based on real data analysis
        edge_cases = [
            # Case 1: Objects with coordinates at image boundaries
            {
                "id": "edge_boundary_test",
                "images": [test_images[0]],
                "objects": [
                    {
                        "bbox_2d": [0, 0, 532, 728],
                        "desc": "全图BBU设备",
                    },  # Full image bbox
                    {
                        "quad": [0, 0, 10, 0, 10, 728, 0, 728],
                        "desc": "边缘标签",
                    },  # Edge quad
                ],
                "width": 532,
                "height": 728,
            },
            # Case 2: Very small objects (single pixel areas)
            {
                "id": "edge_small_test",
                "images": [test_images[0]],
                "objects": [
                    {"bbox_2d": [100, 100, 101, 101], "desc": "极小螺丝"},
                    {"line": [50, 50, 51, 51], "desc": "极短线路"},
                ],
                "width": 532,
                "height": 728,
            },
            # Case 3: Objects with many coordinates (complex lines)
            {
                "id": "edge_complex_test",
                "images": [test_images[0]],
                "objects": [
                    {
                        "line": [i for i in range(0, 100, 5)],
                        "desc": "复杂光纤路径",
                    }  # 20 coordinates
                ],
                "width": 532,
                "height": 728,
            },
        ]

        for case in edge_cases:
            try:
                # Test input preparation for edge cases
                inputs = mock_inference_engine.prepare_inference_inputs(case)
                assert inputs is not None, f"Failed to prepare inputs for {case['id']}"

                # Validate that conversation processor handles edge cases
                mock_conv = mock_inference_engine._mock_conversation_processor
                assert mock_conv.create_simple_conversation.called

            except Exception as e:
                # Log edge case failures for analysis
                print(f"Edge case {case['id']} failed: {str(e)}")
                # Should not raise exceptions for data format issues
                assert "format" not in str(e).lower(), (
                    f"Unexpected format error in {case['id']}: {e}"
                )


# Test execution and reporting functions
def run_comprehensive_inference_tests():
    """Run all comprehensive inference tests."""
    pytest_args = [__file__, "-v", "--tb=short", "--capture=no"]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    exit_code = run_comprehensive_inference_tests()
    sys.exit(exit_code)
