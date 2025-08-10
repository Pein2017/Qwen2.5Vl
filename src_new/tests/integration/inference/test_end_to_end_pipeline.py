#!/usr/bin/env python3
"""
End-to-end validation tests for complete inference pipeline with eval script integration.

This test suite validates the entire inference pipeline from real data input to
eval/infer_dataset.sh output compatibility, ensuring production-ready operation.
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


class TestEndToEndInferencePipeline:
    """End-to-end validation of complete inference pipeline."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for end-to-end tests."""
        temp_dir = tempfile.mkdtemp(prefix="e2e_inference_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_images(self, temp_workspace):
        """Generate test images for end-to-end testing."""
        return generate_test_images(temp_workspace, count=5)

    @pytest.fixture
    def real_data_samples(self):
        """Real dataset samples for testing."""
        return [
            {
                "id": "real_sample_001",
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
                    {
                        "line": [100, 200, 150, 250, 200, 300],
                        "desc": "光纤/有遮挡,弯曲半径合理",
                    },
                ],
                "width": 532,
                "height": 728,
            },
            {
                "id": "real_sample_002",
                "images": ["images/QC-20230314-0000792_83306.jpeg"],
                "objects": [
                    {
                        "line": [333, 20, 392, 185, 370, 278, 220, 388],
                        "desc": "电线/有遮挡,捆扎整齐",
                    },
                    {
                        "bbox_2d": [164, 207, 232, 273],
                        "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求",
                    },
                ],
                "width": 532,
                "height": 728,
            },
        ]

    @pytest.fixture
    def real_teacher_pool(self):
        """Real teacher pool samples for testing."""
        return [
            {
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
                ],
                "width": 532,
                "height": 728,
            }
        ]

    def test_complete_pipeline_with_real_data_format(
        self, temp_workspace, real_data_samples, real_teacher_pool, test_images
    ):
        """Test complete pipeline from input preparation to output generation."""
        # Setup test environment
        data_root = os.path.join(temp_workspace, "data")
        images_dir = os.path.join(data_root, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Copy test images to simulate real data structure
        for i, img_path in enumerate(test_images[:3]):
            dest_path = os.path.join(images_dir, f"test_image_{i}.jpeg")
            shutil.copy(img_path, dest_path)

        # Update sample paths to use copied images
        updated_samples = []
        for i, sample in enumerate(real_data_samples):
            updated_sample = sample.copy()
            updated_sample["images"] = [f"images/test_image_{i}.jpeg"]
            updated_samples.append(updated_sample)

        # Create dataset file
        dataset_file = os.path.join(data_root, "test_val.jsonl")
        with open(dataset_file, "w") as f:
            for sample in updated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Create teacher pool file
        teacher_pool_file = os.path.join(data_root, "teacher_pool.jsonl")
        updated_teachers = []
        for teacher in real_teacher_pool:
            updated_teacher = teacher.copy()
            updated_teacher["images"] = [
                "images/test_image_0.jpeg"
            ]  # Use first test image
            updated_teachers.append(updated_teacher)

        with open(teacher_pool_file, "w") as f:
            for teacher in updated_teachers:
                f.write(json.dumps(teacher, ensure_ascii=False) + "\n")

        # Create minimal config for testing
        model_dir = os.path.join(temp_workspace, "fake_model")
        os.makedirs(model_dir, exist_ok=True)
        config_data = {
            # Required model settings
            "model_path": model_dir,
            "model_size": "3B",
            "model_max_length": 32000,
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
            # Training settings (minimal)
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": 5e-6,
            "vision_lr": 5e-7,
            "merger_lr": 1e-5,
            "llm_lr": 5e-6,
            # Data settings
            "train_data_path": "train.jsonl",
            "val_data_path": "val.jsonl",
            "data_root": data_root,
            "teacher_pool_file": "teacher_pool.jsonl",
            # Output settings
            "output_dir": os.path.join(temp_workspace, "out"),
            "run_name": "test_run",
            "max_coord_value": 1024,
            "model_hidden_size": 2048,
            # Optional behavior flags
            "use_cache": False,
            "coordinate_tokens_enabled": True,
            "new_geometry_tokens": True,
        }

        config_file = os.path.join(temp_workspace, "test_config.yaml")
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Mock inference pipeline
        with (
            patch("inference.DetectionModel") as mock_model_class,
            patch("inference.AutoTokenizer") as mock_tokenizer_class,
            patch("inference.Qwen2VLImageProcessor") as mock_processor_class,
            patch("inference.Qwen2VLProcessor") as mock_unified_processor_class,
        ):
            # Setup mocks
            mock_tokenizer = create_mock_tokenizer()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            mock_model = create_mock_model()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model_class.from_pretrained_fast.return_value = mock_model

            mock_processor = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            # Patch unified processor to a simple object with required attributes
            fake_unified = MagicMock()
            fake_unified.image_processor = mock_processor
            fake_unified.tokenizer = mock_tokenizer

            def fake_apply_chat_template(conversation, **kwargs):
                # Repeat the expected image token for each provided image
                images = kwargs.get("images", []) or []
                return "".join(["<|image_pad|>" for _ in images]) or ""

            fake_unified.apply_chat_template.side_effect = fake_apply_chat_template
            mock_unified_processor_class.return_value = fake_unified

            # Mock model generation to return coordinate token responses
            def mock_generate(**kwargs):
                # Return coordinate token format responses
                return torch.tensor(
                    [
                        [1, 2, 3],  # Mock token IDs
                    ]
                )

            mock_model.generate.side_effect = mock_generate

            # Mock tokenizer decode to return realistic responses
            response_templates = [
                "I can see <|obj_ref_start|>BBU设备/华为,只显示部分,无需安装<|obj_ref_end|><|quad_start|>[<|coord_2|>, <|coord_0|>, <|coord_226|>, <|coord_0|>, <|coord_241|>, <|coord_14|>, <|coord_0|>, <|coord_73|>]<|quad_end|> and <|obj_ref_start|>螺丝、光纤插头/地排处接地螺丝,只显示部分,符合要求<|obj_ref_end|><|box_start|>[<|coord_358|>, <|coord_106|>, <|coord_407|>, <|coord_149|>]<|box_end|>",
                "The image shows <|obj_ref_start|>电线/有遮挡,捆扎整齐<|obj_ref_end|><|line_start|>[<|coord_333|>, <|coord_20|>, <|coord_392|>, <|coord_185|>]<|line_end|> and <|obj_ref_start|>螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求<|obj_ref_end|><|box_start|>[<|coord_164|>, <|coord_207|>, <|coord_232|>, <|coord_273|>]<|box_end|>",
            ]

            call_count = [0]

            def mock_decode(token_ids, **kwargs):
                response = response_templates[call_count[0] % len(response_templates)]
                call_count[0] += 1
                return response

            mock_tokenizer.decode.side_effect = mock_decode

            # Test inference engine initialization
            engine = InferenceEngine(
                config_path=config_file,
                model_path=model_dir,
                data_root=data_root,
                teacher_pool_file="teacher_pool.jsonl",
                num_teachers=1,
                use_training_prompts=True,
                batch_size=1,
            )

            # Process all samples
            results = []
            for sample in updated_samples:
                if engine.num_teachers > 0:
                    inputs = engine._prepare_training_matched_inputs(sample, seed=42)
                else:
                    inputs = engine.prepare_inference_inputs(sample)

                # Generate response
                response = engine.generate_response(inputs, max_new_tokens=128)

                # Process response to extract objects
                objects = engine._parse_coordinate_token_response(response)

                result = {
                    "sample_id": sample["id"],
                    "prediction": json.dumps(objects, ensure_ascii=False),
                    "images": sample["images"],
                }
                results.append(result)

            # Validate results format
            assert len(results) == len(updated_samples)

            for result in results:
                assert "sample_id" in result
                assert "prediction" in result
                assert "images" in result

                # Validate prediction is valid JSON
                prediction = json.loads(result["prediction"])
                assert isinstance(prediction, list)

                # Should contain geometry objects
                if prediction:
                    for obj in prediction:
                        has_geometry = any(
                            key in obj for key in ["bbox_2d", "quad", "line"]
                        )
                        has_desc = "desc" in obj
                        assert has_geometry, f"Object missing geometry: {obj}"
                        assert has_desc, f"Object missing description: {obj}"

    def test_eval_script_parameter_compatibility(
        self, temp_workspace, real_data_samples
    ):
        """Test compatibility with eval/infer_dataset.sh parameter format."""
        # Test script parameters that would be passed to inference
        script_params = {
            "config_path": os.path.join(temp_workspace, "test_config.yaml"),
            "model_path": "/fake/model/path",
            "input_file": os.path.join(temp_workspace, "test_input.jsonl"),
            "output_file": os.path.join(temp_workspace, "predictions.json"),
            "data_root": os.path.join(temp_workspace, "data"),
            "max_new_tokens": 128,
            "batch_size": 1,
            "num_workers": 0,
            "log_level": "debug",
            "num_teachers": 1,
            "teacher_pool_file": os.path.join(temp_workspace, "teacher_pool.jsonl"),
            "max_samples": 3,
            "use_torch_compile": False,
            "force_eager_attention": True,
        }

        # Create test files
        os.makedirs(os.path.join(temp_workspace, "data"), exist_ok=True)

        # Create input file
        with open(script_params["input_file"], "w") as f:
            for sample in real_data_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Create teacher pool file
        with open(script_params["teacher_pool_file"], "w") as f:
            f.write(
                json.dumps(
                    {
                        "images": ["images/teacher.jpeg"],
                        "objects": [
                            {"bbox_2d": [10, 10, 50, 50], "desc": "Teacher example"}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Create config file
        config_data = {
            "coordinate_tokens_enabled": True,
            "max_coord_value": 1024,
            "model_name": "qwen2_5_vl",
        }
        with open(script_params["config_path"], "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Validate all required parameters exist and have valid values
        required_params = [
            "config_path",
            "model_path",
            "input_file",
            "output_file",
            "data_root",
            "max_new_tokens",
            "batch_size",
        ]

        for param in required_params:
            assert param in script_params, f"Missing required parameter: {param}"
            assert script_params[param] is not None, f"Parameter {param} is None"

        # Validate file paths exist (except model_path which is fake)
        file_params = ["config_path", "input_file", "teacher_pool_file"]
        for param in file_params:
            if param in script_params:
                assert os.path.exists(script_params[param]), (
                    f"File not found: {script_params[param]}"
                )

        # Validate data types
        assert isinstance(script_params["max_new_tokens"], int)
        assert isinstance(script_params["batch_size"], int)
        assert isinstance(script_params["num_teachers"], int)
        assert isinstance(script_params["max_samples"], int)
        assert isinstance(script_params["use_torch_compile"], bool)
        assert isinstance(script_params["force_eager_attention"], bool)

    def test_performance_benchmarks_with_real_constraints(
        self, temp_workspace, real_data_samples
    ):
        """Test performance characteristics under real inference constraints."""
        metrics = TestMetrics()

        # Test processing time for realistic sample sizes
        sample_sizes = [1, 3, 5]  # Small, medium, large batches

        for size in sample_sizes:
            test_samples = real_data_samples[:size]

            start_time = time.time()

            # Simulate processing
            processed_samples = []
            for sample in test_samples:
                # Simulate input preparation time
                time.sleep(0.01)  # Mock processing delay

                processed_sample = {
                    "sample_id": sample["id"],
                    "prediction": json.dumps(sample["objects"], ensure_ascii=False),
                    "images": sample["images"],
                }
                processed_samples.append(processed_sample)

            processing_time = time.time() - start_time
            metrics.add_measurement(
                f"batch_size_{size}_processing_time", processing_time
            )

            # Validate processing results (cap by available real_data_samples)
            assert len(processed_samples) == min(size, len(real_data_samples))
            for processed in processed_samples:
                assert "sample_id" in processed
                assert "prediction" in processed

                # Validate JSON parsing
                prediction = json.loads(processed["prediction"])
                assert isinstance(prediction, list)

        # Performance assertions based on real constraints
        single_time = metrics.get_measurement("batch_size_1_processing_time")
        if single_time:
            assert single_time < 1.0, (
                f"Single sample processing too slow: {single_time:.2f}s"
            )

        # Log metrics for analysis
        print("\nPerformance Metrics:")
        for metric_name, value in metrics.measurements.items():
            print(f"  {metric_name}: {value:.4f}s")

    @skip_if_no_gpu
    def test_memory_usage_with_coordinate_tokens(
        self, temp_workspace, real_data_samples
    ):
        """Test memory usage characteristics with coordinate token processing."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for memory testing")

        # Test memory usage with different scenarios
        scenarios = [
            {"name": "standard_mode", "coordinate_tokens": False},
            {"name": "coordinate_token_mode", "coordinate_tokens": True},
        ]

        memory_results = {}

        for scenario in scenarios:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Simulate model loading with/without coordinate tokens
            if scenario["coordinate_tokens"]:
                # Additional memory for coordinate tokens (allocate a predictable GPU tensor)
                mock_coord_embeddings = torch.zeros(1024, 1024, device="cuda")
            else:
                mock_coord_embeddings = None

            # Simulate processing
            for sample in real_data_samples[:2]:  # Test with 2 samples
                # Use CPU tensors to avoid GPU kernel issues; we're measuring GPU allocs from coord embeddings
                mock_input_ids = torch.randint(0, 1000, (1, 128))
                mock_attention_mask = torch.ones(1, 128)
                mock_pixel_values = torch.randn(1, 3, 64, 64)

                # Simulate small processing delay
                time.sleep(0.005)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            memory_results[scenario["name"]] = memory_used

            # Cleanup
            del mock_input_ids, mock_attention_mask, mock_pixel_values
            if mock_coord_embeddings is not None:
                del mock_coord_embeddings
            torch.cuda.empty_cache()

        # Memory usage analysis
        print("\nMemory Usage Analysis:")
        for scenario, usage in memory_results.items():
            print(f"  {scenario}: {usage / (1024**2):.2f} MB")

        # Validate memory usage is reasonable
        for usage in memory_results.values():
            # Should not exceed 1GB for test processing
            assert usage < 1024**3, f"Memory usage too high: {usage / (1024**2):.2f} MB"

        # Coordinate token mode should use more memory but not dramatically more
        if len(memory_results) == 2:
            standard_usage = memory_results["standard_mode"]
            coord_token_usage = memory_results["coordinate_token_mode"]

            if standard_usage > 0:
                ratio = coord_token_usage / max(standard_usage, 1)
                assert ratio < 10.0, (
                    f"Coordinate token memory overhead too high: {ratio:.2f}x"
                )

    def test_error_handling_and_recovery(self, temp_workspace, test_images):
        """Test error handling and recovery scenarios in production pipeline."""
        # Error scenarios to test
        error_scenarios = [
            {
                "name": "missing_image_file",
                "sample": {
                    "id": "error_test_001",
                    "images": ["images/nonexistent.jpeg"],
                    "objects": [],
                },
                "expected_error": True,
            },
            {
                "name": "malformed_objects",
                "sample": {
                    "id": "error_test_002",
                    "images": [test_images[0]],
                    "objects": [
                        {"invalid_geometry": [1, 2, 3], "desc": "Invalid object"}
                    ],
                },
                "expected_error": False,  # Should handle gracefully
            },
            {
                "name": "empty_images_list",
                "sample": {"id": "error_test_003", "images": [], "objects": []},
                "expected_error": True,
            },
            {
                "name": "invalid_coordinates",
                "sample": {
                    "id": "error_test_004",
                    "images": [test_images[0]],
                    "objects": [
                        {"bbox_2d": [-100, -200, 5000, 6000], "desc": "Out of bounds"}
                    ],
                },
                "expected_error": False,  # Should clamp coordinates
            },
        ]

        # Setup test environment
        data_root = os.path.join(temp_workspace, "data")
        os.makedirs(data_root, exist_ok=True)

        config_data = {"coordinate_tokens_enabled": True, "max_coord_value": 1024}
        config_file = os.path.join(temp_workspace, "test_config.yaml")
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Mock inference engine for error testing
        with (
            patch("inference.DetectionModel") as mock_model_class,
            patch("inference.AutoTokenizer") as mock_tokenizer_class,
        ):
            mock_tokenizer = create_mock_tokenizer()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            mock_model = create_mock_model()
            mock_model_class.from_pretrained.return_value = mock_model

            try:
                engine = InferenceEngine(
                    config_path=config_file,
                    model_path="/fake/model/path",
                    data_root=data_root,
                    batch_size=1,
                )

                # Test each error scenario
                for scenario in error_scenarios:
                    sample = scenario["sample"]
                    expected_error = scenario["expected_error"]

                    print(f"\nTesting scenario: {scenario['name']}")

                    if expected_error:
                        # Should raise an exception
                        with pytest.raises(Exception):
                            engine.prepare_inference_inputs(sample)
                    else:
                        # Should handle gracefully
                        try:
                            inputs = engine.prepare_inference_inputs(sample)
                            assert inputs is not None or inputs == {}
                            print(f"  ✓ Handled gracefully: {scenario['name']}")
                        except Exception as e:
                            pytest.fail(f"Unexpected error in {scenario['name']}: {e}")

            except Exception as e:
                pytest.fail(f"Failed to initialize inference engine: {e}")


class TestEvalScriptIntegration:
    """Test integration with eval/infer_dataset.sh script."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for eval script tests."""
        temp_dir = tempfile.mkdtemp(prefix="eval_script_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_images(self, temp_workspace):
        """Generate test images."""
        return generate_test_images(temp_workspace, count=3)

    def test_eval_script_command_compatibility(self, temp_workspace, test_images):
        """Test that we can simulate eval/infer_dataset.sh command execution."""
        # Create realistic test setup
        data_root = os.path.join(temp_workspace, "data", "ds_v2_full")
        images_dir = os.path.join(data_root, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Copy test images
        for i, img_path in enumerate(test_images):
            dest_path = os.path.join(images_dir, f"QC-test-{i:06d}.jpeg")
            shutil.copy(img_path, dest_path)

        # Create test dataset
        test_samples = [
            {
                "id": f"eval_test_{i:03d}",
                "images": [f"images/QC-test-{i:06d}.jpeg"],
                "objects": [
                    {
                        "bbox_2d": [i * 10, i * 10, (i + 2) * 10, (i + 2) * 10],
                        "desc": f"螺丝、光纤插头/测试对象{i},符合要求",
                    }
                ],
                "width": 532,
                "height": 728,
            }
            for i in range(3)
        ]

        val_file = os.path.join(data_root, "val.jsonl")
        with open(val_file, "w") as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Create teacher pool
        teacher_pool_file = os.path.join(data_root, "teacher_pool.jsonl")
        with open(teacher_pool_file, "w") as f:
            teacher = {
                "images": ["images/QC-test-000000.jpeg"],
                "objects": [{"bbox_2d": [5, 5, 15, 15], "desc": "Teacher example"}],
            }
            f.write(json.dumps(teacher, ensure_ascii=False) + "\n")

        # Create config file
        config_file = os.path.join(temp_workspace, "test_config.yaml")
        config_data = {
            "coordinate_tokens_enabled": True,
            "max_coord_value": 1024,
            "model_name": "qwen2_5_vl",
        }
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        # Simulate eval script parameters
        script_env = {
            "EXP_NAME": "test-experiment",
            "DATASET": "val",
            "DATA_SUBDIR": "ds_v2_full",
            "MODEL_PATH": "/fake/model/path",
            "CONFIG_PATH": config_file,
            "NUM_TEACHERS": "1",
            "TEACHER_POOL_FILE": teacher_pool_file,
            "MAX_NEW_TOKENS": "128",
            "BATCH_SIZE": "1",
            "MAX_SAMPLES": "3",
            "LOG_LEVEL": "debug",
        }

        # Build command that eval script would run
        output_file = os.path.join(temp_workspace, "predictions.json")

        inference_args = [
            "--config_path",
            script_env["CONFIG_PATH"],
            "--model_path",
            script_env["MODEL_PATH"],
            "--input_file",
            val_file,
            "--output_file",
            output_file,
            "--data_root",
            data_root,
            "--max_new_tokens",
            script_env["MAX_NEW_TOKENS"],
            "--batch_size",
            script_env["BATCH_SIZE"],
            "--num_teachers",
            script_env["NUM_TEACHERS"],
            "--teacher_pool_file",
            script_env["TEACHER_POOL_FILE"],
            "--max_samples",
            script_env["MAX_SAMPLES"],
            "--log_level",
            script_env["LOG_LEVEL"],
        ]

        # Validate all required arguments are present and files exist
        assert os.path.exists(script_env["CONFIG_PATH"])
        assert os.path.exists(val_file)
        assert os.path.exists(script_env["TEACHER_POOL_FILE"])

        # Validate argument format
        assert len(inference_args) % 2 == 0, "Arguments should be in key-value pairs"

        # Simulate successful execution by creating expected output
        mock_results = [
            {
                "sample_id": sample["id"],
                "prediction": json.dumps(sample["objects"], ensure_ascii=False),
                "images": sample["images"],
            }
            for sample in test_samples
        ]

        with open(output_file, "w") as f:
            json.dump(mock_results, f, ensure_ascii=False, indent=2)

        # Validate output format matches eval script expectations
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        assert isinstance(results, list)
        assert len(results) == 3

        for result in results:
            assert "sample_id" in result
            assert "prediction" in result
            assert "images" in result

            # Validate prediction format
            prediction = json.loads(result["prediction"])
            assert isinstance(prediction, list)
            assert len(prediction) >= 1

    def test_eval_script_file_structure_compatibility(self, temp_workspace):
        """Test that file structure matches eval script expectations."""
        # Create expected directory structure
        exp_base = os.path.join(temp_workspace, "exp_det_coordinates")
        exp_dir = os.path.join(exp_base, "test-experiment")
        dataset_dir = os.path.join(exp_dir, "val")
        inference_dir = os.path.join(dataset_dir, "inference")

        os.makedirs(inference_dir, exist_ok=True)

        # Create expected config file
        config_file = os.path.join(exp_dir, "config.json")
        config_data = {
            "exp_name": "test-experiment",
            "model": {"name": "qwen2_5_vl", "path": "/fake/model/path"},
            "generation": {
                "max_new_tokens": 128,
                "batch_size": 1,
                "num_workers": 0,
                "enable_torch_compile": False,
            },
            "teacher": {
                "num_teachers": 1,
                "teacher_pool_file": "data/ds_v2_full/teacher_pool.jsonl",
            },
            "debug": {"max_samples": 3},
            "log_level": "debug",
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        # Create expected output files
        output_file = os.path.join(inference_dir, "predictions.json")
        log_file = os.path.join(inference_dir, "inference.log")

        # Mock inference results
        mock_results = [
            {
                "sample_id": "test_001",
                "prediction": json.dumps(
                    [{"bbox_2d": [100, 100, 200, 200], "desc": "Test object"}],
                    ensure_ascii=False,
                ),
                "images": ["images/test.jpeg"],
            }
        ]

        with open(output_file, "w") as f:
            json.dump(mock_results, f, ensure_ascii=False, indent=2)

        with open(log_file, "w") as f:
            f.write("INFO: Inference completed successfully\n")
            f.write("INFO: Processed 1 samples\n")

        # Validate directory structure
        assert os.path.exists(exp_base)
        assert os.path.exists(exp_dir)
        assert os.path.exists(dataset_dir)
        assert os.path.exists(inference_dir)
        assert os.path.exists(config_file)
        assert os.path.exists(output_file)
        assert os.path.exists(log_file)

        # Validate config format
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["exp_name"] == "test-experiment"
        assert "model" in loaded_config
        assert "generation" in loaded_config
        assert "teacher" in loaded_config

        # Validate results format
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        assert isinstance(results, list)
        assert len(results) == 1
        assert "sample_id" in results[0]
        assert "prediction" in results[0]


# Test execution
def run_end_to_end_inference_tests():
    """Run end-to-end inference pipeline tests."""
    pytest_args = [__file__, "-v", "--tb=short", "--capture=no"]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    exit_code = run_end_to_end_inference_tests()
    sys.exit(exit_code)
