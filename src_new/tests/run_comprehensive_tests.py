#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Runner for Training Pipeline

This script runs all comprehensive tests for the training pipeline with real data,
providing detailed logging and validation of the complete workflow.

Test Categories:
1. Real Data Integration Tests
2. Training Component Tests (Loss Computation, Token Masking)
3. Validation Tests (Edge Cases, Boundary Conditions)
4. Complete Pipeline Integration Tests

Usage:
    python src_new/tests/run_comprehensive_tests.py [--category CATEGORY] [--verbose]

Categories:
    - integration: Real data integration and complete pipeline tests
    - training: Training-specific tests (loss, masking, spans)
    - validation: Edge cases and boundary condition tests
    - all: Run all comprehensive tests (default)
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from src_new.utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def run_pytest_command(test_path: str, verbose: bool = False) -> bool:
    """
    Run pytest command and return success status.

    Args:
        test_path: Path to test file or directory
        verbose: Whether to use verbose output

    Returns:
        True if tests passed, False otherwise
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v" if verbose else "-q",
        "--tb=short",
        "--disable-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"✅ PASSED: {test_path}")
            if verbose:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ FAILED: {test_path}")
            logger.info(result.stdout)
            logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ TIMEOUT: {test_path} (exceeded 5 minutes)")
        return False
    except Exception as e:
        logger.error(f"💥 ERROR: {test_path} - {e}")
        return False


def run_integration_tests(verbose: bool = False) -> List[bool]:
    """Run integration tests with real data."""
    logger.info("🧪 Running Integration Tests with Real Data...")

    test_files = [
        "src_new/tests/test_integration/test_real_data_pipeline.py",
        "src_new/tests/test_integration/test_complete_pipeline.py",
    ]

    results = []
    for test_file in test_files:
        logger.info(f"🔄 Running {test_file}...")
        start_time = time.time()

        success = run_pytest_command(test_file, verbose)
        results.append(success)

        elapsed = time.time() - start_time
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} {test_file} ({elapsed:.1f}s)")

    return results


def run_training_tests(verbose: bool = False) -> List[bool]:
    """Run training-specific tests."""
    logger.info("🎯 Running Training Component Tests...")

    test_files = [
        "src_new/tests/test_training/test_loss_computation.py",
        "src_new/tests/test_training/test_token_masking.py",
    ]

    results = []
    for test_file in test_files:
        logger.info(f"🔄 Running {test_file}...")
        start_time = time.time()

        success = run_pytest_command(test_file, verbose)
        results.append(success)

        elapsed = time.time() - start_time
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} {test_file} ({elapsed:.1f}s)")

    return results


def run_validation_tests(verbose: bool = False) -> List[bool]:
    """Run validation and edge case tests."""
    logger.info("🔍 Running Validation and Edge Case Tests...")

    test_files = ["src_new/tests/test_validation/test_edge_cases.py"]

    results = []
    for test_file in test_files:
        logger.info(f"🔄 Running {test_file}...")
        start_time = time.time()

        success = run_pytest_command(test_file, verbose)
        results.append(success)

        elapsed = time.time() - start_time
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} {test_file} ({elapsed:.1f}s)")

    return results


def run_all_tests(verbose: bool = False) -> bool:
    """Run all comprehensive tests."""
    logger.info("🚀 Running All Comprehensive Training Pipeline Tests...")

    start_time = time.time()

    # Run each test category
    integration_results = run_integration_tests(verbose)
    training_results = run_training_tests(verbose)
    validation_results = run_validation_tests(verbose)

    # Combine all results
    all_results = integration_results + training_results + validation_results

    # Calculate summary
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests

    elapsed = time.time() - start_time

    # Print summary
    logger.info("=" * 60)
    logger.info("📊 COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests} ✅")
    logger.info(f"Failed: {failed_tests} ❌")
    logger.info(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
    logger.info(f"Total time: {elapsed:.1f}s")

    if failed_tests == 0:
        logger.info("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        logger.info("✅ Training pipeline is ready for production use")
        return True
    else:
        logger.error(f"💥 {failed_tests} TESTS FAILED")
        logger.error("❌ Please fix failing tests before proceeding")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive training pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--category",
        choices=["integration", "training", "validation", "all"],
        default="all",
        help="Test category to run (default: all)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.verbose)

    # Verify we're in the correct directory
    if not Path("src_new/tests").exists():
        logger.error("❌ Must run from project root directory")
        logger.error("Current directory should contain src_new/tests/")
        sys.exit(1)

    # Verify real data files exist
    data_root = Path("data/ds_v2_full")
    if not data_root.exists():
        logger.error(f"❌ Real data directory not found: {data_root}")
        logger.error("Please ensure data/ds_v2_full/ exists with required JSONL files")
        sys.exit(1)

    logger.info(f"🎯 Running {args.category} tests...")

    # Run selected test category
    if args.category == "integration":
        results = run_integration_tests(args.verbose)
        success = all(results)
    elif args.category == "training":
        results = run_training_tests(args.verbose)
        success = all(results)
    elif args.category == "validation":
        results = run_validation_tests(args.verbose)
        success = all(results)
    else:  # all
        success = run_all_tests(args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
