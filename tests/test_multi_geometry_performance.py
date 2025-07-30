"""
Performance Tests for Multi-Geometry Token Parser

Tests performance characteristics, memory usage, and scalability
of the multi-geometry token parsing functionality.
"""

import pytest
import sys
import time
import gc
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.response_parser import ResponseParser
from src.logger_utils import configure_global_logging, get_logger


logger = get_logger("test_multi_geometry_performance")


class TestMultiGeometryPerformance:
    """Performance tests for multi-geometry token parsing."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        configure_global_logging(rank=0, world_size=1)
        cls.parser = ResponseParser()

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_large_input_processing_100_objects(self):
        """Test processing 100 geometry objects in a single response."""
        # Generate 100 bbox objects with coordinates
        objects = []
        for i in range(100):
            obj = (
                f"<obj_ref_start><bbox_2d_start>设备_{i}"
                f"<|coord_{i*4}|><|coord_{i*4+1}|><|coord_{i*4+2}|><|coord_{i*4+3}|>"
                f"<bbox_2d_end><obj_ref_end>"
            )
            objects.append(obj)
        
        large_input = "".join(objects)
        
        # Measure parsing time
        start_time = time.time()
        parsed_objects = self.parser._parse_multi_geometry_tokens(large_input, coordinate_tokens_enabled=True)
        end_time = time.time()
        
        parsing_time = end_time - start_time
        
        # Assertions
        assert len(parsed_objects) == 100, f"Expected 100 objects, got {len(parsed_objects)}"
        assert parsing_time < 1.0, f"Parsing took too long: {parsing_time:.3f}s (should be < 1.0s)"
        
        # Verify correctness of first and last objects
        assert parsed_objects[0]["geometry_type"] == "bbox_2d"
        assert parsed_objects[0]["caption"] == "设备_0"
        assert parsed_objects[0]["coordinates"] == [0, 1, 2, 3]
        
        assert parsed_objects[99]["geometry_type"] == "bbox_2d"
        assert parsed_objects[99]["caption"] == "设备_99"
        assert parsed_objects[99]["coordinates"] == [396, 397, 398, 399]
        
        logger.info(f"✅ Parsed 100 objects in {parsing_time:.3f}s ({100/parsing_time:.1f} objects/s)")

    def test_large_input_processing_500_objects(self):
        """Test processing 500 geometry objects (stress test)."""
        # Generate 500 mixed geometry objects
        objects = []
        geometry_types = ["bbox_2d", "line", "square"]
        
        for i in range(500):
            geom_type = geometry_types[i % 3]
            
            if geom_type == "bbox_2d":
                coords = f"<|coord_{i*4}|><|coord_{i*4+1}|><|coord_{i*4+2}|><|coord_{i*4+3}|>"
            elif geom_type == "line":
                coords = f"<|coord_{i*6}|><|coord_{i*6+1}|><|coord_{i*6+2}|><|coord_{i*6+3}|><|coord_{i*6+4}|><|coord_{i*6+5}|>"
            else:  # square
                coords = "".join(f"<|coord_{i*8+j}|>" for j in range(8))
            
            obj = f"<obj_ref_start><{geom_type}_start>对象_{i}{coords}<{geom_type}_end><obj_ref_end>"
            objects.append(obj)
        
        large_input = "".join(objects)
        
        # Measure memory before parsing
        gc.collect()
        memory_before = self.get_memory_usage()
        
        # Measure parsing time
        start_time = time.time()
        parsed_objects = self.parser._parse_multi_geometry_tokens(large_input, coordinate_tokens_enabled=True)
        end_time = time.time()
        
        # Measure memory after parsing
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        parsing_time = end_time - start_time
        
        # Assertions
        assert len(parsed_objects) == 500, f"Expected 500 objects, got {len(parsed_objects)}"
        assert parsing_time < 5.0, f"Parsing took too long: {parsing_time:.3f}s (should be < 5.0s)"
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.1f}MB (should be < 100MB)"
        
        logger.info(f"✅ Parsed 500 objects in {parsing_time:.3f}s ({500/parsing_time:.1f} objects/s)")
        logger.info(f"📊 Memory increase: {memory_increase:.1f}MB")

    def test_parsing_speed_benchmark(self):
        """Benchmark parsing speed for different input sizes."""
        sizes = [10, 50, 100, 200]
        results = {}
        
        for size in sizes:
            # Generate objects
            objects = []
            for i in range(size):
                obj = (
                    f"<obj_ref_start><bbox_2d_start>设备_{i}"
                    f"<|coord_{i*4}|><|coord_{i*4+1}|><|coord_{i*4+2}|><|coord_{i*4+3}|>"
                    f"<bbox_2d_end><obj_ref_end>"
                )
                objects.append(obj)
            
            input_text = "".join(objects)
            
            # Measure parsing time (average of 3 runs)
            times = []
            for _ in range(3):
                start_time = time.time()
                parsed_objects = self.parser._parse_multi_geometry_tokens(input_text, coordinate_tokens_enabled=True)
                end_time = time.time()
                times.append(end_time - start_time)
                
                assert len(parsed_objects) == size
            
            avg_time = sum(times) / len(times)
            objects_per_second = size / avg_time
            results[size] = {
                "time": avg_time,
                "objects_per_second": objects_per_second
            }
            
            logger.info(f"📊 {size} objects: {avg_time:.3f}s ({objects_per_second:.1f} objects/s)")
        
        # Verify performance scales reasonably (not exponential)
        # Performance should not degrade dramatically with size
        small_rate = results[10]["objects_per_second"]
        large_rate = results[200]["objects_per_second"]
        
        # Allow up to 50% performance degradation for 20x more objects
        assert large_rate > small_rate * 0.5, f"Performance degraded too much: {small_rate:.1f} -> {large_rate:.1f} objects/s"

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with input size."""
        sizes = [50, 100, 200]
        memory_usage = {}
        
        for size in sizes:
            # Generate objects
            objects = []
            for i in range(size):
                obj = (
                    f"<obj_ref_start><bbox_2d_start>设备_{i}"
                    f"<|coord_{i*4}|><|coord_{i*4+1}|><|coord_{i*4+2}|><|coord_{i*4+3}|>"
                    f"<bbox_2d_end><obj_ref_end>"
                )
                objects.append(obj)
            
            input_text = "".join(objects)
            
            # Measure memory usage
            gc.collect()
            memory_before = self.get_memory_usage()
            
            parsed_objects = self.parser._parse_multi_geometry_tokens(input_text, coordinate_tokens_enabled=True)
            
            memory_after = self.get_memory_usage()
            memory_increase = memory_after - memory_before
            memory_usage[size] = memory_increase
            
            assert len(parsed_objects) == size
            
            logger.info(f"📊 {size} objects: {memory_increase:.1f}MB memory increase")
            
            # Clean up
            del parsed_objects
            del input_text
            gc.collect()
        
        # Memory usage should scale roughly linearly (not exponentially)
        # Allow some overhead but not excessive
        ratio_100_50 = memory_usage[100] / memory_usage[50] if memory_usage[50] > 0 else 1
        ratio_200_100 = memory_usage[200] / memory_usage[100] if memory_usage[100] > 0 else 1
        
        # Ratios should be reasonable (not more than 3x for 2x the data)
        assert ratio_100_50 < 3.0, f"Memory scaling 50->100 too high: {ratio_100_50:.2f}x"
        assert ratio_200_100 < 3.0, f"Memory scaling 100->200 too high: {ratio_200_100:.2f}x"

    def test_coordinate_token_parsing_performance(self):
        """Test performance of coordinate token parsing specifically."""
        # Create object with many coordinate tokens
        coords = "".join(f"<|coord_{i}|>" for i in range(100))
        content = f"设备{coords}"
        
        # Measure parsing time for coordinate tokens
        times = []
        for _ in range(10):  # Average of 10 runs
            start_time = time.time()
            result = self.parser._parse_geometry_content(content, coordinate_tokens_enabled=True)
            end_time = time.time()
            times.append(end_time - start_time)
            
            assert result is not None
            assert len(result["coordinates"]) == 100
        
        avg_time = sum(times) / len(times)
        coords_per_second = 100 / avg_time
        
        # Should be able to parse at least 1000 coordinate tokens per second
        assert coords_per_second > 1000, f"Coordinate parsing too slow: {coords_per_second:.1f} coords/s"
        
        logger.info(f"✅ Coordinate token parsing: {coords_per_second:.1f} tokens/s")

    def test_regex_performance_large_text(self):
        """Test regex performance on large text with no matches."""
        # Create large text with no multi-geometry tokens
        large_text = "这是一段很长的文本，包含很多中文字符和标点符号。" * 1000
        
        # Measure parsing time
        start_time = time.time()
        objects = self.parser._parse_multi_geometry_tokens(large_text, coordinate_tokens_enabled=True)
        end_time = time.time()
        
        parsing_time = end_time - start_time
        
        # Should quickly determine no matches and return empty list
        assert len(objects) == 0
        assert parsing_time < 0.1, f"Regex on large text took too long: {parsing_time:.3f}s"
        
        logger.info(f"✅ Large text with no matches processed in {parsing_time:.3f}s")

    def test_alternative_pattern_performance(self):
        """Test performance of alternative pattern matching."""
        # Create input that requires alternative pattern matching
        text = "< obj_ref_start >< bbox_2d_start >BBU设备< bbox_2d_end >< obj_ref_end >"
        
        # Measure parsing time for alternative patterns
        times = []
        for _ in range(100):  # Average of 100 runs
            start_time = time.time()
            objects = self.parser._parse_multi_geometry_tokens(text, coordinate_tokens_enabled=False)
            end_time = time.time()
            times.append(end_time - start_time)
            
            assert len(objects) == 1
        
        avg_time = sum(times) / len(times)
        
        # Alternative pattern matching should still be fast
        assert avg_time < 0.01, f"Alternative pattern matching too slow: {avg_time:.4f}s"
        
        logger.info(f"✅ Alternative pattern matching: {avg_time:.4f}s average")

    @pytest.mark.slow
    def test_stress_test_1000_objects(self):
        """Stress test with 1000 objects (marked as slow test)."""
        # Generate 1000 objects
        objects = []
        for i in range(1000):
            obj = (
                f"<obj_ref_start><bbox_2d_start>设备_{i}"
                f"<|coord_{i}|><|coord_{i+1000}|><|coord_{i+2000}|><|coord_{i+3000}|>"
                f"<bbox_2d_end><obj_ref_end>"
            )
            objects.append(obj)
        
        large_input = "".join(objects)
        
        # Measure memory and time
        gc.collect()
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        parsed_objects = self.parser._parse_multi_geometry_tokens(large_input, coordinate_tokens_enabled=True)
        
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        parsing_time = end_time - start_time
        memory_increase = memory_after - memory_before
        
        # Assertions for stress test
        assert len(parsed_objects) == 1000
        assert parsing_time < 10.0, f"Stress test took too long: {parsing_time:.3f}s"
        assert memory_increase < 200, f"Memory increase too large: {memory_increase:.1f}MB"
        
        # Verify correctness of sample objects
        assert parsed_objects[0]["coordinates"] == [0, 1000, 2000, 3000]
        assert parsed_objects[999]["coordinates"] == [999, 1999, 2999, 3999]
        
        logger.info(f"✅ Stress test: 1000 objects in {parsing_time:.3f}s ({1000/parsing_time:.1f} objects/s)")
        logger.info(f"📊 Memory increase: {memory_increase:.1f}MB")

    def test_concurrent_parsing_simulation(self):
        """Simulate concurrent parsing by running multiple parsers."""
        import threading
        import queue
        
        # Create multiple parser instances
        parsers = [ResponseParser() for _ in range(5)]
        results_queue = queue.Queue()
        
        def parse_worker(parser_id, parser):
            # Each worker parses 20 objects
            objects = []
            for i in range(20):
                obj = (
                    f"<obj_ref_start><bbox_2d_start>设备_{parser_id}_{i}"
                    f"<|coord_{i*4}|><|coord_{i*4+1}|><|coord_{i*4+2}|><|coord_{i*4+3}|>"
                    f"<bbox_2d_end><obj_ref_end>"
                )
                objects.append(obj)
            
            input_text = "".join(objects)
            
            start_time = time.time()
            parsed_objects = parser._parse_multi_geometry_tokens(input_text, coordinate_tokens_enabled=True)
            end_time = time.time()
            
            results_queue.put({
                "parser_id": parser_id,
                "count": len(parsed_objects),
                "time": end_time - start_time
            })
        
        # Start all workers
        threads = []
        start_time = time.time()
        
        for i, parser in enumerate(parsers):
            thread = threading.Thread(target=parse_worker, args=(i, parser))
            thread.start()
            threads.append(thread)
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all workers completed successfully
        assert len(results) == 5
        for result in results:
            assert result["count"] == 20
            assert result["time"] < 1.0  # Each worker should complete quickly
        
        total_objects = sum(r["count"] for r in results)
        objects_per_second = total_objects / total_time
        
        logger.info(f"✅ Concurrent parsing: {total_objects} objects in {total_time:.3f}s ({objects_per_second:.1f} objects/s)")
        
        # Should handle concurrent parsing efficiently
        assert total_time < 2.0, f"Concurrent parsing took too long: {total_time:.3f}s"
