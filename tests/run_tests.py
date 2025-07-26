#!/usr/bin/env python3
"""
BBU Training Pipeline Test Runner

Professional test runner for the BBU training pipeline test suite.
Provides options for running different test categories and performance monitoring.
"""

import argparse
import gc
import sys
import time
import unittest

import torch

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton


patch_torch_library_wrap_triton()

from src.logger_utils import configure_global_logging, get_logger
from tests.fixtures.test_utils import TestUtils


logger = get_logger("test_runner")


def run_test_suite(test_modules: list, verbosity: int = 2) -> bool:
    """
    Run specified test modules with GPU memory management.

    Args:
        test_modules: List of test module names to run
        verbosity: Test output verbosity level

    Returns:
        True if all tests passed, False otherwise
    """
    configure_global_logging(
        log_dir="tests/logs", log_file="test_run.log", rank=0, world_size=1
    )

    logger.info("🧪 Starting BBU Training Pipeline Test Suite")
    logger.info("=" * 60)
    logger.info("")
    
    # Check GPU availability and initial cleanup
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 GPU Available: {gpu_name} ({total_memory:.1f} GB)")
        TestUtils.force_gpu_memory_cleanup(verbose=True)
    else:
        logger.warning("⚠️ No GPU available - tests will run on CPU")

    total_start_time = time.time()
    all_passed = True
    results = {}

    # Run tests sequentially with GPU memory management
    for module_idx, module_name in enumerate(test_modules):
        logger.info(f"\n🔄 Running {module_name} ({module_idx + 1}/{len(test_modules)})")
        logger.info("-" * 40)
        
        # Force GPU cleanup before each module
        if torch.cuda.is_available():
            TestUtils.force_gpu_memory_cleanup(verbose=True)
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"💾 Starting {module_name} - GPU Memory: {initial_memory:.1f} MB")

        start_time = time.time()

        try:
            # Import test module
            module = __import__(f"tests.{module_name}", fromlist=[module_name])

            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)

            # Run tests with GPU memory monitoring
            with TestUtils.gpu_memory_guard(f"Module {module_name}", cleanup_after=True):
                runner = unittest.TextTestRunner(
                    verbosity=verbosity, buffer=True, stream=sys.stdout
                )
                result = runner.run(suite)

            # Record results
            elapsed_time = time.time() - start_time
            passed = result.wasSuccessful()

            results[module_name] = {
                "passed": passed,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "elapsed_time": elapsed_time,
            }

            if not passed:
                all_passed = False
                logger.error(f"❌ {module_name} FAILED")
            else:
                logger.info(f"✅ {module_name} PASSED ({elapsed_time:.2f}s)")

        except Exception as e:
            logger.error(f"❌ Failed to run {module_name}: {e}")
            all_passed = False
            results[module_name] = {
                "passed": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
        
        # Force cleanup after each module to prevent memory accumulation
        if torch.cuda.is_available():
            TestUtils.force_gpu_memory_cleanup(verbose=True)
            gc.collect()  # Force garbage collection
            time.sleep(0.5)  # Allow GPU memory to stabilize

    # Print summary
    total_elapsed = time.time() - total_start_time

    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Suite Summary")
    logger.info("=" * 60)

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for module_name, result in results.items():
        if result["passed"]:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        tests_run = result.get("tests_run", 0)
        failures = result.get("failures", 0)
        errors = result.get("errors", 0)
        elapsed = result["elapsed_time"]

        logger.info(f"{status} {module_name}")
        logger.info(f"   Tests: {tests_run}, Failures: {failures}, Errors: {errors}")
        logger.info(f"   Time: {elapsed:.2f}s")

        total_tests += tests_run
        total_failures += failures
        total_errors += errors

    logger.info("-" * 60)
    logger.info(f"Total Tests Run: {total_tests}")
    logger.info(f"Total Failures: {total_failures}")
    logger.info(f"Total Errors: {total_errors}")
    logger.info(f"Total Time: {total_elapsed:.2f}s")

    # Final GPU cleanup
    if torch.cuda.is_available():
        TestUtils.force_gpu_memory_cleanup(verbose=True)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("💥 SOME TESTS FAILED!")

    return all_passed


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="BBU Training Pipeline Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/run_tests.py --all
  # Run specific test modules
  python tests/run_tests.py --modules test_data_pipeline test_model_loading
  # Run quick tests (skip slow integration tests)
  python tests/run_tests.py --quick
  # Run with minimal output
  python tests/run_tests.py --all --verbosity 1
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all test modules")

    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests (skip integration tests)"
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        help="Specific test modules to run",
        choices=[
            "test_data_pipeline",
            "test_model_loading",
            "test_training_components",
            "test_integration",
        ],
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)",
    )

    args = parser.parse_args()

    # Determine which test modules to run
    if args.all:
        test_modules = [
            "test_data_pipeline",
            "test_model_loading",
            "test_training_components",
            "test_integration",
        ]
    elif args.quick:
        test_modules = [
            "test_data_pipeline",
            "test_model_loading",
            "test_training_components",
        ]
    elif args.modules:
        test_modules = args.modules
    else:
        # Default to quick tests
        test_modules = ["test_data_pipeline", "test_model_loading"]

    # Run tests
    success = run_test_suite(test_modules, args.verbosity)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
