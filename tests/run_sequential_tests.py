#!/usr/bin/env python3
"""
Sequential Test Runner with Memory Management

Runs BBU tests sequentially with aggressive memory cleanup between tests
to prevent CUDA out of memory errors on 80GB GPU.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

from src.logger_utils import configure_global_logging, get_logger

configure_global_logging(rank=0, world_size=1)
logger = get_logger("sequential_test_runner")


def force_memory_cleanup():
    """Force comprehensive memory cleanup between tests."""
    if torch.cuda.is_available():
        logger.info("🧹 Forcing memory cleanup between tests...")
        
        # Force garbage collection
        import gc
        for _ in range(3):
            gc.collect()
        
        # Synchronize and clear CUDA cache
        torch.cuda.synchronize()
        for _ in range(5):
            torch.cuda.empty_cache()
            time.sleep(0.1)
        
        # Additional cleanup
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.synchronize()
        
        # Log memory state
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"💾 GPU Memory after cleanup: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")


def run_test_file(test_file: str, python_path: str = "/root/miniconda3/envs/ms/bin/python") -> bool:
    """
    Run a single test file with memory management.
    
    Args:
        test_file: Path to test file
        python_path: Path to Python interpreter
        
    Returns:
        True if test passed, False if failed
    """
    logger.info(f"🧪 Running test file: {test_file}")
    
    # Force cleanup before starting test
    force_memory_cleanup()
    
    # Set environment variables for test isolation
    env = os.environ.copy()
    env.update({
        "CUDA_LAUNCH_BLOCKING": "1",  # Synchronous CUDA for better error reporting
        "TORCH_SHOW_CPP_STACKTRACES": "1",  # Better error traces
        "BBU_TEST_MODE": "true",
        "TOKENIZERS_PARALLELISM": "false",
    })
    
    # Run test with timeout
    try:
        result = subprocess.run(
            [python_path, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd="/data3/Qwen2.5-VL-main",
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per test file
        )
        
        # Log output
        if result.stdout:
            logger.info(f"📝 Test output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"⚠️ Test errors:\n{result.stderr}")
        
        success = result.returncode == 0
        if success:
            logger.info(f"✅ Test passed: {test_file}")
        else:
            logger.error(f"❌ Test failed: {test_file} (exit code: {result.returncode})")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Test timed out: {test_file}")
        return False
    except Exception as e:
        logger.error(f"💥 Test execution error: {test_file} - {e}")
        return False
    finally:
        # Force cleanup after test
        force_memory_cleanup()
        # Wait a bit between tests
        time.sleep(2)


def main():
    """Main test runner."""
    logger.info("🚀 Starting Sequential Test Runner")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU: {gpu_name} ({total_memory:.1f} GB)")
    else:
        logger.warning("⚠️ No GPU available")
    
    # Define test files in order of increasing complexity
    test_files = [
        "tests/test_model_loading.py",
        "tests/test_training_components.py", 
        "tests/test_integration.py",
    ]
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run tests sequentially
    for test_file in test_files:
        test_path = Path(test_file)
        if not test_path.exists():
            logger.warning(f"⚠️ Test file not found: {test_file}")
            results[test_file] = False
            continue
        
        start_time = time.time()
        success = run_test_file(test_file)
        duration = time.time() - start_time
        
        results[test_file] = success
        logger.info(f"⏱️ Test duration: {duration:.1f} seconds")
        
        # Early exit on failure (optional)
        if not success:
            logger.warning(f"⚠️ Test failed, continuing with remaining tests...")
    
    # Summary
    total_duration = time.time() - total_start_time
    passed = sum(results.values())
    total = len(results)
    
    logger.info("📊 Test Summary:")
    logger.info(f"   Total time: {total_duration:.1f} seconds")
    logger.info(f"   Tests passed: {passed}/{total}")
    
    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {status}: {test_file}")
    
    # Exit with error code if any tests failed
    if passed < total:
        logger.error(f"❌ {total - passed} test(s) failed")
        sys.exit(1)
    else:
        logger.info("🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()