"""
Performance benchmarking tests for src_new implementation.

This module provides benchmarks for:
- Data loading and processing speed
- Memory usage optimization
- Training throughput measurement
- Model inference speed
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
import pytest
import torch

from src_new.tests.fixtures import (
    MockImageProcessor,
    MockModel,
    MockTokenizer,
    create_sample_jsonl_data,
    create_temp_files,
)


class TestPerformanceBenchmarks:
    """Performance benchmarking test suite."""

    def test_data_loading_speed_benchmark(self, temp_dir):
        """Benchmark data loading speed for different dataset sizes."""
        # Create datasets of different sizes
        dataset_sizes = [10, 50, 100, 500]
        results = {}

        for size in dataset_sizes:
            # Generate dataset
            large_data = []
            for i in range(size):
                sample = {
                    "images": [f"image_{i}.jpg"],
                    "objects": [
                        {
                            "bbox_2d": [i * 2, i * 2 + 10, i * 2 + 20, i * 2 + 30],
                            "desc": f"设备_{i}/性能测试",
                        }
                    ],
                    "width": 800,
                    "height": 600,
                }
                large_data.append(sample)

            # Write to file
            test_files = {f"dataset_{size}.jsonl": large_data}
            test_dir = create_temp_files(test_files, temp_dir)
            dataset_path = test_dir / f"dataset_{size}.jsonl"

            # Benchmark loading
            start_time = time.time()
            loaded_samples = self._mock_load_jsonl(dataset_path)
            end_time = time.time()

            loading_time = end_time - start_time
            samples_per_second = (
                size / loading_time if loading_time > 0 else float("inf")
            )

            results[size] = {
                "loading_time": loading_time,
                "samples_per_second": samples_per_second,
                "samples_loaded": len(loaded_samples),
            }

            # Validate results
            assert len(loaded_samples) == size
            assert samples_per_second > 10, (
                f"Loading speed too slow: {samples_per_second} samples/sec"
            )

        # Analyze scaling
        print("\nData Loading Performance:")
        for size, metrics in results.items():
            print(
                f"  {size} samples: {metrics['loading_time']:.3f}s, {metrics['samples_per_second']:.1f} samples/sec"
            )

        # Check that loading scales reasonably
        small_throughput = results[10]["samples_per_second"]
        large_throughput = results[100]["samples_per_second"]

        # Should not degrade by more than 50%
        assert large_throughput > small_throughput * 0.5, (
            "Loading speed degrades too much with size"
        )

    def test_tokenization_speed_benchmark(self, temp_dir):
        """Benchmark tokenization speed for different text lengths."""
        tokenizer = MockTokenizer()

        # Test different conversation lengths
        text_lengths = [100, 500, 1000, 2000]  # Character counts
        results = {}

        for length in text_lengths:
            # Create conversation of specified length
            base_conversation = "你是通信机房设备检测AI助手。" * (length // 20)
            conversation = base_conversation[:length]

            # Benchmark tokenization
            start_time = time.time()

            # Simulate multiple tokenization calls
            num_calls = 100
            for _ in range(num_calls):
                token_ids = tokenizer.encode(conversation)

            end_time = time.time()

            total_time = end_time - start_time
            time_per_call = total_time / num_calls
            chars_per_second = (
                length / time_per_call if time_per_call > 0 else float("inf")
            )

            results[length] = {
                "time_per_call": time_per_call,
                "chars_per_second": chars_per_second,
                "tokens_generated": len(token_ids),
            }

            # Validate performance
            assert time_per_call < 0.01, (
                f"Tokenization too slow: {time_per_call:.4f}s per call"
            )
            assert chars_per_second > 10000, (
                f"Character processing too slow: {chars_per_second:.1f} chars/sec"
            )

        print("\nTokenization Performance:")
        for length, metrics in results.items():
            print(
                f"  {length} chars: {metrics['time_per_call']:.4f}s/call, {metrics['chars_per_second']:.0f} chars/sec"
            )

    def test_batch_processing_speed_benchmark(self, temp_dir):
        """Benchmark batch processing speed for different batch sizes."""
        tokenizer = MockTokenizer()
        image_processor = MockImageProcessor()

        batch_sizes = [1, 2, 4, 8]
        results = {}

        # Create sample data
        sample_data = create_sample_jsonl_data()

        for batch_size in batch_sizes:
            # Create batch
            batch_samples = (sample_data * ((batch_size // len(sample_data)) + 1))[
                :batch_size
            ]

            # Benchmark processing
            start_time = time.time()

            processed_batch = self._mock_process_batch(
                batch_samples, tokenizer, image_processor
            )

            end_time = time.time()

            processing_time = end_time - start_time
            samples_per_second = (
                batch_size / processing_time if processing_time > 0 else float("inf")
            )

            results[batch_size] = {
                "processing_time": processing_time,
                "samples_per_second": samples_per_second,
                "batch_size": batch_size,
            }

            # Validate results
            assert "input_ids" in processed_batch
            assert processed_batch["input_ids"].size(0) == batch_size
            assert samples_per_second > 1, (
                f"Batch processing too slow: {samples_per_second:.2f} samples/sec"
            )

        print("\nBatch Processing Performance:")
        for batch_size, metrics in results.items():
            print(
                f"  Batch size {batch_size}: {metrics['processing_time']:.3f}s, {metrics['samples_per_second']:.1f} samples/sec"
            )

        # Check batch processing efficiency
        single_throughput = results[1]["samples_per_second"]
        batch_throughput = results[4]["samples_per_second"]

        # Batching should provide some efficiency gain
        assert batch_throughput >= single_throughput, (
            "Batching should not decrease throughput"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_usage_benchmark(self, temp_dir):
        """Benchmark GPU memory usage for different model configurations."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for memory benchmarking")

        # Test different model sizes
        model_configs = [
            {"vocab_size": 151665, "hidden_size": 1024, "name": "small"},
            {"vocab_size": 151665, "hidden_size": 2048, "name": "medium"},
            {
                "vocab_size": 151665 + 2049,
                "hidden_size": 2048,
                "name": "extended",
            },  # With coordinate tokens
        ]

        results = {}

        for config in model_configs:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model
            model = MockModel(
                vocab_size=config["vocab_size"], hidden_size=config["hidden_size"]
            )
            model = model.to("cuda")

            # Create test batch
            batch_size = 2
            seq_len = 1000
            batch = {
                "input_ids": torch.randint(
                    0, config["vocab_size"], (batch_size, seq_len)
                ).cuda(),
                "attention_mask": torch.ones(batch_size, seq_len).cuda(),
                "labels": torch.randint(
                    0, config["vocab_size"], (batch_size, seq_len)
                ).cuda(),
            }

            # Measure memory usage
            start_memory = torch.cuda.memory_allocated()
            peak_memory_before = torch.cuda.max_memory_allocated()

            # Forward pass
            with torch.no_grad():
                output = model.forward(**batch)

            peak_memory_after = torch.cuda.max_memory_allocated()
            end_memory = torch.cuda.memory_allocated()

            memory_used = end_memory - start_memory
            peak_memory_increase = peak_memory_after - peak_memory_before

            results[config["name"]] = {
                "memory_used_mb": memory_used / 1024 / 1024,
                "peak_memory_mb": peak_memory_increase / 1024 / 1024,
                "vocab_size": config["vocab_size"],
                "hidden_size": config["hidden_size"],
            }

            # Clean up
            del model, batch, output
            torch.cuda.empty_cache()

        print("\nGPU Memory Usage:")
        for name, metrics in results.items():
            print(
                f"  {name}: {metrics['memory_used_mb']:.1f}MB used, {metrics['peak_memory_mb']:.1f}MB peak"
            )

        # Validate memory usage is reasonable
        for name, metrics in results.items():
            assert metrics["memory_used_mb"] < 8000, (
                f"{name} model uses too much memory: {metrics['memory_used_mb']:.1f}MB"
            )

    def test_coordinate_token_processing_benchmark(self, temp_dir):
        """Benchmark coordinate token processing overhead."""
        # Compare standard vs coordinate token processing
        test_coordinates = [
            [100, 150, 200, 250],  # bbox_2d
            [300, 400, 350, 410, 348, 425, 302, 415],  # square
            [50, 100, 150, 120, 250, 140, 350, 160],  # line
        ]

        num_iterations = 1000

        # Benchmark standard processing (coordinates as integers)
        start_time = time.time()
        for _ in range(num_iterations):
            for coords in test_coordinates:
                # Mock standard processing
                coord_str = str(coords)
                tokens = coord_str.split()  # Simple tokenization
        end_time = time.time()

        standard_time = end_time - start_time

        # Benchmark coordinate token processing
        start_time = time.time()
        for _ in range(num_iterations):
            for coords in test_coordinates:
                # Mock coordinate token processing
                coord_tokens = [f"<|coord_{coord}|>" for coord in coords]
                token_mapping = {token: i for i, token in enumerate(coord_tokens)}
        end_time = time.time()

        coordinate_token_time = end_time - start_time

        # Calculate overhead
        overhead_ratio = (
            coordinate_token_time / standard_time if standard_time > 0 else 1
        )

        print(f"\nCoordinate Token Processing:")
        print(f"  Standard processing: {standard_time:.4f}s")
        print(f"  Coordinate token processing: {coordinate_token_time:.4f}s")
        print(f"  Overhead ratio: {overhead_ratio:.2f}x")

        # Validate overhead is acceptable
        assert overhead_ratio < 5.0, (
            f"Coordinate token overhead too high: {overhead_ratio:.2f}x"
        )
        assert coordinate_token_time < 1.0, (
            f"Coordinate token processing too slow: {coordinate_token_time:.4f}s"
        )

    def test_teacher_student_processing_benchmark(self, temp_dir):
        """Benchmark teacher-student processing overhead."""
        # Create test data
        teacher_sample = {
            "images": ["teacher.jpg"],
            "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "教师示例/设备"}],
            "width": 400,
            "height": 300,
        }

        student_sample = {
            "images": ["student.jpg"],
            "objects": [{"bbox_2d": [100, 150, 200, 250], "desc": "学生目标/设备"}],
            "width": 800,
            "height": 600,
        }

        tokenizer = MockTokenizer()
        num_iterations = 100

        # Benchmark standalone processing
        start_time = time.time()
        for _ in range(num_iterations):
            conversation = self._mock_create_standalone_conversation(student_sample)
            tokens = tokenizer.encode(conversation)
        end_time = time.time()

        standalone_time = end_time - start_time

        # Benchmark teacher-student processing
        start_time = time.time()
        for _ in range(num_iterations):
            conversation = self._mock_create_teacher_student_conversation(
                teacher_sample, student_sample
            )
            tokens = tokenizer.encode(conversation)
        end_time = time.time()

        teacher_student_time = end_time - start_time

        # Calculate overhead
        overhead_ratio = (
            teacher_student_time / standalone_time if standalone_time > 0 else 1
        )

        print(f"\nTeacher-Student Processing:")
        print(f"  Standalone processing: {standalone_time:.4f}s")
        print(f"  Teacher-student processing: {teacher_student_time:.4f}s")
        print(f"  Overhead ratio: {overhead_ratio:.2f}x")

        # Validate overhead is reasonable (teacher-student should be ~2x due to extra content)
        assert overhead_ratio < 5.0, (
            f"Teacher-student overhead too high: {overhead_ratio:.2f}x"
        )
        assert overhead_ratio > 1.0, (
            "Teacher-student should have some overhead due to extra content"
        )

    def test_memory_usage_monitoring(self, temp_dir):
        """Monitor memory usage during processing."""
        process = psutil.Process()

        # Baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_data = []
        for i in range(1000):
            sample = {
                "images": [f"image_{i}.jpg"],
                "objects": [
                    {
                        "bbox_2d": [i, i + 10, i + 20, i + 30],
                        "desc": f"设备_{i}/内存测试",
                    }
                ],
                "width": 800,
                "height": 600,
            }
            large_data.append(sample)

        # Monitor memory during processing
        memory_checkpoints = []

        # Checkpoint 1: After data creation
        memory_checkpoints.append(process.memory_info().rss / 1024 / 1024)

        # Process data
        tokenizer = MockTokenizer()
        processed_samples = []

        for i, sample in enumerate(large_data):
            conversation = self._mock_create_standalone_conversation(sample)
            tokens = tokenizer.encode(conversation)
            processed_samples.append(tokens)

            # Memory checkpoint every 100 samples
            if i % 100 == 0:
                memory_checkpoints.append(process.memory_info().rss / 1024 / 1024)

        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_checkpoints.append(final_memory)

        # Analyze memory usage
        max_memory = max(memory_checkpoints)
        memory_increase = final_memory - initial_memory
        peak_increase = max_memory - initial_memory

        print(f"\nMemory Usage Analysis:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Peak increase: {peak_increase:.1f}MB")

        # Validate memory usage is reasonable
        assert memory_increase < 1000, (
            f"Memory increase too large: {memory_increase:.1f}MB"
        )
        assert peak_increase < 1500, (
            f"Peak memory increase too large: {peak_increase:.1f}MB"
        )

        # Check for memory leaks (final should not be much larger than expected)
        expected_increase = len(large_data) * 0.01  # ~0.01MB per sample
        assert memory_increase < expected_increase * 10, "Possible memory leak detected"

    # Helper methods for benchmarking

    def _mock_load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Mock JSONL loading for benchmarking."""
        import json

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)

        return samples

    def _mock_process_batch(
        self, samples: List[Dict[str, Any]], tokenizer, image_processor
    ) -> Dict[str, torch.Tensor]:
        """Mock batch processing for benchmarking."""
        processed_samples = []

        for sample in samples:
            # Mock processing
            conversation = self._mock_create_standalone_conversation(sample)
            tokens = tokenizer.encode(conversation)

            processed_sample = {
                "input_ids": torch.tensor(tokens),
                "attention_mask": torch.ones(len(tokens)),
                "labels": torch.tensor(tokens),
            }

            # Mock image processing
            if image_processor:
                image_data = image_processor.preprocess(sample["images"])
                processed_sample.update(image_data)

            processed_samples.append(processed_sample)

        # Create batch
        max_length = max(len(sample["input_ids"]) for sample in processed_samples)

        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            padded_sequences = []
            for sample in processed_samples:
                seq = sample[key]
                pad_length = max_length - len(seq)
                padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=seq.dtype)])
                padded_sequences.append(padded_seq)
            batch[key] = torch.stack(padded_sequences)

        # Add image data
        if "pixel_values" in processed_samples[0]:
            batch["pixel_values"] = torch.stack(
                [sample["pixel_values"] for sample in processed_samples]
            )

        return batch

    def _mock_create_standalone_conversation(self, sample: Dict[str, Any]) -> str:
        """Mock standalone conversation creation."""
        import json

        conversation_parts = [
            "<im_start>system\n你是通信机房设备检测AI助手。<im_end>",
            "<im_start>user\n请检测图像中的设备和部件: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(sample['objects'], ensure_ascii=False)}<im_end>",
        ]

        return "\n".join(conversation_parts)

    def _mock_create_teacher_student_conversation(
        self, teacher: Dict[str, Any], student: Dict[str, Any]
    ) -> str:
        """Mock teacher-student conversation creation."""
        import json

        conversation_parts = [
            "<im_start>system\n你是通信机房设备检测AI助手。学习参考示例。<im_end>",
            "<im_start>user\n参考示例: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(teacher['objects'], ensure_ascii=False)}<im_end>",
            "<im_start>user\n检测目标图像: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(student['objects'], ensure_ascii=False)}<im_end>",
        ]

        return "\n".join(conversation_parts)


class TestPerformanceComparison:
    """Performance comparison tests between different configurations."""

    def test_coordinate_tokens_vs_standard_performance(self, temp_dir):
        """Compare performance between coordinate token and standard modes."""
        # Test data
        test_samples = create_sample_jsonl_data()

        test_files = {"test_data.jsonl": test_samples}
        test_dir = create_temp_files(test_files, temp_dir)

        tokenizer = MockTokenizer()

        # Benchmark standard mode
        start_time = time.time()
        standard_results = []
        for sample in test_samples:
            conversation = self._mock_create_conversation(
                sample, coordinate_tokens=False
            )
            tokens = tokenizer.encode(conversation)
            standard_results.append(len(tokens))
        standard_time = time.time() - start_time

        # Benchmark coordinate token mode
        start_time = time.time()
        coordinate_results = []
        for sample in test_samples:
            conversation = self._mock_create_conversation(
                sample, coordinate_tokens=True
            )
            tokens = tokenizer.encode(conversation)
            coordinate_results.append(len(tokens))
        coordinate_time = time.time() - start_time

        # Compare results
        avg_standard_tokens = sum(standard_results) / len(standard_results)
        avg_coordinate_tokens = sum(coordinate_results) / len(coordinate_results)

        token_overhead = avg_coordinate_tokens / avg_standard_tokens
        time_overhead = coordinate_time / standard_time

        print(f"\nCoordinate Tokens vs Standard Comparison:")
        print(
            f"  Standard mode: {standard_time:.4f}s, {avg_standard_tokens:.1f} avg tokens"
        )
        print(
            f"  Coordinate mode: {coordinate_time:.4f}s, {avg_coordinate_tokens:.1f} avg tokens"
        )
        print(f"  Token overhead: {token_overhead:.2f}x")
        print(f"  Time overhead: {time_overhead:.2f}x")

        # Validate overheads are acceptable
        assert token_overhead < 2.0, f"Token overhead too high: {token_overhead:.2f}x"
        assert time_overhead < 3.0, f"Time overhead too high: {time_overhead:.2f}x"

    def test_teacher_student_vs_standalone_performance(self, temp_dir):
        """Compare performance between teacher-student and standalone modes."""
        test_samples = create_sample_jsonl_data()
        teacher_samples = test_samples[:1]  # Use first as teacher
        student_samples = test_samples[1:]

        tokenizer = MockTokenizer()

        # Benchmark standalone mode
        start_time = time.time()
        standalone_tokens = []
        for sample in student_samples:
            conversation = self._mock_create_standalone_conversation(sample)
            tokens = tokenizer.encode(conversation)
            standalone_tokens.append(len(tokens))
        standalone_time = time.time() - start_time

        # Benchmark teacher-student mode
        start_time = time.time()
        teacher_student_tokens = []
        for sample in student_samples:
            # Use first teacher sample
            conversation = self._mock_create_teacher_student_conversation(
                teacher_samples[0], sample
            )
            tokens = tokenizer.encode(conversation)
            teacher_student_tokens.append(len(tokens))
        teacher_student_time = time.time() - start_time

        # Compare results
        avg_standalone = sum(standalone_tokens) / len(standalone_tokens)
        avg_teacher_student = sum(teacher_student_tokens) / len(teacher_student_tokens)

        token_ratio = avg_teacher_student / avg_standalone
        time_ratio = teacher_student_time / standalone_time

        print(f"\nTeacher-Student vs Standalone Comparison:")
        print(f"  Standalone: {standalone_time:.4f}s, {avg_standalone:.1f} avg tokens")
        print(
            f"  Teacher-student: {teacher_student_time:.4f}s, {avg_teacher_student:.1f} avg tokens"
        )
        print(f"  Token ratio: {token_ratio:.2f}x")
        print(f"  Time ratio: {time_ratio:.2f}x")

        # Teacher-student should have more tokens but reasonable overhead
        assert token_ratio > 1.5, (
            "Teacher-student should have significantly more tokens"
        )
        assert token_ratio < 4.0, f"Token increase too large: {token_ratio:.2f}x"
        assert time_ratio < 5.0, f"Time overhead too large: {time_ratio:.2f}x"

    def _mock_create_conversation(
        self, sample: Dict[str, Any], coordinate_tokens: bool = False
    ) -> str:
        """Mock conversation creation with optional coordinate tokens."""
        import json

        objects = sample["objects"]

        if coordinate_tokens:
            # Mock coordinate token conversion
            processed_objects = []
            for obj in objects:
                processed_obj = obj.copy()
                for geo_type in ["bbox_2d", "square", "line"]:
                    if geo_type in obj:
                        coords = obj[geo_type]
                        # Convert to coordinate tokens
                        coord_tokens = [f"<|coord_{coord}|>" for coord in coords]
                        processed_obj[geo_type] = coord_tokens
                processed_objects.append(processed_obj)
            objects = processed_objects

        conversation_parts = [
            "<im_start>system\n你是通信机房设备检测AI助手。<im_end>",
            "<im_start>user\n请检测图像: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(objects, ensure_ascii=False)}<im_end>",
        ]

        return "\n".join(conversation_parts)

    def _mock_create_standalone_conversation(self, sample: Dict[str, Any]) -> str:
        """Mock standalone conversation creation."""
        import json

        conversation_parts = [
            "<im_start>system\n你是通信机房设备检测AI助手。<im_end>",
            "<im_start>user\n检测图像: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(sample['objects'], ensure_ascii=False)}<im_end>",
        ]

        return "\n".join(conversation_parts)

    def _mock_create_teacher_student_conversation(
        self, teacher: Dict[str, Any], student: Dict[str, Any]
    ) -> str:
        """Mock teacher-student conversation creation."""
        import json

        conversation_parts = [
            "<im_start>system\n你是通信机房设备检测AI助手。学习参考示例，然后分析新图像。<im_end>",
            "<im_start>user\n参考示例: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(teacher['objects'], ensure_ascii=False)}<im_end>",
            "<im_start>user\n现在检测目标图像: <image><im_end>",
            f"<im_start>assistant\n{json.dumps(student['objects'], ensure_ascii=False)}<im_end>",
        ]

        return "\n".join(conversation_parts)
