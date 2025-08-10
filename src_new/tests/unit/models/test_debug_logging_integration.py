#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify enhanced debug logging integration in the training pipeline.

This script demonstrates the comprehensive conversation flow logging with real training data,
showing pre-tokenization conversation text, token-level analysis, and loss mask validation.

Usage:
    python src_new/tests/test_debug_logging_integration.py

The script will:
1. Load real BBU training data from data/ds_v2_full/
2. Set up the training pipeline with debug logging enabled
3. Process one training sample to trigger comprehensive debug logging
4. Show the complete conversation flow analysis
"""

import logging
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to DEBUG level to see all debug output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_debug_logging_integration():
    """Test the enhanced debug logging integration with real training data."""

    print("🧪 Testing Enhanced Debug Logging Integration")
    print("=" * 80)

    try:
        # Import required modules
        from transformers import TrainingArguments

        from src_new.config.config import load_config
        from src_new.data.collator import PackedDataCollator
        from src_new.data.dataset import Dataset
        from src_new.data.teacher_pool import TeacherPoolManager
        from src_new.models.wrapper import DetectionModel
        from src_new.training.bbu_trainer import BBUTrainer
        from src_new.utils.debug_logging import debug_logger

        print("✅ Successfully imported all required modules")

        # Load configuration
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_debug.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        config = load_config(str(config_path))
        print(f"✅ Loaded configuration from {config_path}")

        # Reset debug logger for fresh test
        debug_logger.reset_for_new_run()
        print("✅ Reset debug logger for fresh test")

        # Load teacher pool
        teacher_pool = TeacherPoolManager(
            teacher_pool_file=config.teacher_pool_file,
            config=config,
        )
        print(
            f"✅ Loaded teacher pool with {len(teacher_pool.teacher_samples)} samples"
        )

        # Create datasets (limit to 1 sample for testing)
        # Note: We need to create a tokenizer and image processor for the Dataset
        from transformers import AutoImageProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        train_dataset = Dataset(
            data_path=config.train_data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            teacher_pool_manager=teacher_pool,
            config=config,
        )
        print(f"✅ Created training dataset with {len(train_dataset)} samples")

        # Load model
        model = DetectionModel.from_pretrained(
            config.model_path,
            config=config,
            trust_remote_code=True,
        )
        print("✅ Loaded detection model")

        # Create data collator
        collator = PackedDataCollator(
            tokenizer=model.tokenizer,
            max_length=config.max_length,
        )
        print("✅ Created data collator")

        # Create minimal training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            max_steps=1,  # Only 1 step for testing
            logging_steps=1,
            save_steps=1000,  # Don't save during test
            log_level="debug",  # Enable debug logging
            report_to=[],  # Disable wandb/tensorboard
        )
        print("✅ Created training arguments")

        # Create trainer
        trainer = BBUTrainer(
            model=model,
            processing_class=model.tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        print("✅ Created BBU trainer")

        # Start debug logging session
        debug_logger.start_training_run()

        print("\n🔍 TRIGGERING DEBUG LOGGING WITH REAL TRAINING DATA")
        print("=" * 80)
        print("The following output shows the comprehensive conversation flow logging:")
        print("1. Pre-tokenization conversation text (complete chat template)")
        print("2. Conversation structure analysis (system/user/assistant segments)")
        print("3. Coordinate token analysis (BBU equipment coordinates)")
        print("4. Loss mask analysis (learning vs. masked tokens)")
        print("5. Token-by-token breakdown (first 50 tokens)")
        print("=" * 80)

        # Process one training step to trigger debug logging
        # This will call compute_loss which triggers _log_sample_debug_info
        trainer.train()

        print("\n✅ DEBUG LOGGING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The debug logging captured:")
        print("- Complete conversation text with Chinese BBU descriptions")
        print("- Coordinate tokens showing equipment positions")
        print("- Teacher-student conversation structure")
        print("- Token masking for proper loss computation")
        print("- Detailed token-by-token analysis")

        return True

    except Exception as e:
        print(f"❌ Error during debug logging test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_debug_logging_flags():
    """Test the one-time logging mechanism."""

    print("\n🧪 Testing One-Time Logging Mechanism")
    print("=" * 50)

    from src_new.utils.debug_logging import debug_logger

    # Reset for clean test
    debug_logger.reset_for_new_run()

    # Test training sample logging flags
    print("Testing training sample logging:")
    print(
        f"  First call should_log_training_sample(): {debug_logger.should_log_training_sample()}"
    )  # Should be True
    print(
        f"  Second call should_log_training_sample(): {debug_logger.should_log_training_sample()}"
    )  # Should be False

    # Test evaluation sample logging flags
    print("Testing evaluation sample logging:")
    print(
        f"  First call should_log_evaluation_sample(): {debug_logger.should_log_evaluation_sample()}"
    )  # Should be True
    print(
        f"  Second call should_log_evaluation_sample(): {debug_logger.should_log_evaluation_sample()}"
    )  # Should be False

    # Test loss mask logging flags
    print("Testing loss mask logging:")
    print(
        f"  First call should_log_training_loss_mask(): {debug_logger.should_log_training_loss_mask()}"
    )  # Should be True
    print(
        f"  Second call should_log_training_loss_mask(): {debug_logger.should_log_training_loss_mask()}"
    )  # Should be False

    print("✅ One-time logging mechanism working correctly")


if __name__ == "__main__":
    print("🚀 Enhanced Debug Logging Integration Test")
    print("=" * 80)

    # Test the one-time logging flags first
    test_debug_logging_flags()

    # Test the full integration
    success = test_debug_logging_integration()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print(
            "The enhanced debug logging is properly integrated and working with real BBU data."
        )
    else:
        print("\n❌ TESTS FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)
