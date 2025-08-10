#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for rank-aware logging functionality.

This script validates that the rank-aware logging system works correctly
in both single-process and distributed training scenarios.

Usage:
    # Single process test
    python src_new/utils/test_rank_aware_logging.py

    # Distributed test (simulate with environment variables)
    RANK=1 WORLD_SIZE=4 python src_new/utils/test_rank_aware_logging.py
"""

import logging
import os

from src_new.utils.rank_aware_logging import (
    all_ranks,
    configure_rank_aware_logging,
    get_rank_aware_logger,
    get_rank_info,
    log_distributed_info,
    rank0_only,
)


def test_basic_logging():
    """Test basic rank-aware logging functionality."""
    print("=" * 60)
    print("Testing Basic Rank-Aware Logging")
    print("=" * 60)

    # Get rank information
    rank, world_size, is_main = get_rank_info()
    print(f"Process Info: rank={rank}, world_size={world_size}, is_main={is_main}")

    # Configure logging
    configure_rank_aware_logging(log_level=logging.DEBUG)

    # Get logger
    logger = get_rank_aware_logger("test_logger")

    # Test different log levels
    logger.debug(f"DEBUG message from rank {rank}")
    logger.info(f"INFO message from rank {rank}")
    logger.warning(f"WARNING message from rank {rank}")
    logger.error(f"ERROR message from rank {rank}")

    print(f"Expected behavior:")
    print(f"  - DEBUG/INFO: Only shown if rank={rank} == 0")
    print(f"  - WARNING/ERROR: Always shown (with rank prefix if world_size > 1)")


@rank0_only
def test_rank0_decorator():
    """Test the rank0_only decorator."""
    print("\n" + "=" * 60)
    print("Testing @rank0_only Decorator")
    print("=" * 60)

    logger = get_rank_aware_logger("decorator_test")
    logger.info("This message should only appear on rank 0")
    return "rank0_only_result"


@all_ranks
def test_all_ranks_decorator():
    """Test the all_ranks decorator (documentation only)."""
    rank, _, _ = get_rank_info()
    logger = get_rank_aware_logger("all_ranks_test")
    logger.warning(f"This WARNING should appear on all ranks (current: {rank})")
    return f"all_ranks_result_from_{rank}"


def test_distributed_info():
    """Test distributed information logging."""
    print("\n" + "=" * 60)
    print("Testing Distributed Info Logging")
    print("=" * 60)

    logger = get_rank_aware_logger("dist_info_test")
    log_distributed_info(logger)


def test_multiple_loggers():
    """Test multiple loggers with rank-aware filtering."""
    print("\n" + "=" * 60)
    print("Testing Multiple Loggers")
    print("=" * 60)

    # Create multiple loggers
    logger1 = get_rank_aware_logger("module1")
    logger2 = get_rank_aware_logger("module2")
    logger3 = get_rank_aware_logger("module3")

    # Test that they all have rank-aware filtering
    logger1.info("Module 1 info message")
    logger2.info("Module 2 info message")
    logger3.info("Module 3 info message")

    # Test error messages (should appear on all ranks)
    logger1.error("Module 1 error message")
    logger2.error("Module 2 error message")


def simulate_training_logging():
    """Simulate typical training logging patterns."""
    print("\n" + "=" * 60)
    print("Simulating Training Logging Patterns")
    print("=" * 60)

    trainer_logger = get_rank_aware_logger("trainer")
    model_logger = get_rank_aware_logger("model")
    loss_logger = get_rank_aware_logger("loss")

    # Simulate training progress (should only show on rank 0)
    trainer_logger.info("Starting training...")
    trainer_logger.info("Epoch 1/10, Step 100/1000")
    trainer_logger.info("Loss: 0.5432, Learning Rate: 1e-4")

    # Simulate model operations (should only show on rank 0)
    model_logger.info("Loading model checkpoint...")
    model_logger.info("Model loaded successfully")

    # Simulate loss computation (should only show on rank 0)
    loss_logger.info("Computing loss components...")
    loss_logger.debug("Teacher loss: 0.3, Student loss: 0.7")

    # Simulate errors (should show on all ranks)
    if get_rank_info()[0] == 1:  # Simulate error on rank 1
        trainer_logger.error("Simulated training error on rank 1")

    # Simulate warnings (should show on all ranks)
    model_logger.warning("Model checkpoint is from different version")


def main():
    """Main test function."""
    print("🧪 Rank-Aware Logging Test Suite")
    print(
        f"Environment: RANK={os.getenv('RANK', 'unset')}, "
        f"WORLD_SIZE={os.getenv('WORLD_SIZE', 'unset')}"
    )

    # Run tests
    test_basic_logging()

    result1 = test_rank0_decorator()
    print(f"@rank0_only result: {result1}")

    result2 = test_all_ranks_decorator()
    print(f"@all_ranks result: {result2}")

    test_distributed_info()
    test_multiple_loggers()
    simulate_training_logging()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    rank, world_size, is_main = get_rank_info()
    print(f"Process: rank={rank}, world_size={world_size}, is_main={is_main}")
    print("Expected behavior:")
    print("  ✅ INFO/DEBUG messages: Only on rank 0")
    print("  ✅ WARNING/ERROR messages: On all ranks")
    print("  ✅ @rank0_only functions: Only execute on rank 0")
    print("  ✅ Rank prefix: Added to ERROR messages when world_size > 1")

    if is_main:
        print("\n🎉 Test completed successfully!")
    else:
        print(f"\n✅ Rank {rank} test completed!")


if __name__ == "__main__":
    main()
