#!/usr/bin/env python
"""
Debug script to reproduce the exact conditions that cause the trainer compatibility issue.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from transformers.training_args import TrainingArguments
from src.config import init_config, load_config
from src.training.trainer_factory import create_trainer_with_coordinator
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_exact_test_conditions():
    """Debug using exact same conditions as the failing test."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_exact_test")
    
    # Use same parameters as the integration test
    data_generator = SyntheticDataGenerator(num_samples=8)  # Same as test
    train_path, val_path, teacher_path, all_samples_path = data_generator.generate_complete_dataset()
    data_root = str(data_generator.temp_dir)
    
    # Create coordinate-enabled configuration using same method as test
    config_factory = ConfigFactory()
    config_path = config_factory.create_coordinate_enabled_config(data_root, "standard")
    
    # Initialize configuration
    init_config(config_path)
    config = load_config(config_path)
    
    logger.info("🔍 Using EXACT same training arguments as integration test...")
    
    # Use EXACT same training arguments as the integration test
    training_args = TrainingArguments(
        output_dir=f"{data_root}/pipeline_test_coordinate_mode",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Match exactly
        per_device_eval_batch_size=2,   # Match exactly  
        max_steps=2,                   # Match exactly - this might be key!
        logging_steps=1,
        eval_steps=2,
        eval_strategy="steps",         # Match exactly
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
    )
    
    logger.info(f"Training args: max_steps={training_args.max_steps}, per_device_train_batch_size={training_args.per_device_train_batch_size}")
    
    # Wrap the collator to log what it receives
    class LoggingCollatorWrapper:
        def __init__(self, original_collator, logger):
            self.original_collator = original_collator
            self.logger = logger
            self.call_count = 0
            
        def __call__(self, features):
            self.call_count += 1
            self.logger.info(f"🔍 Collator call #{self.call_count}")
            self.logger.info(f"   Received {len(features)} features")
            
            for i, feature in enumerate(features):
                if isinstance(feature, dict):
                    if not feature:
                        self.logger.error(f"   ❌ Feature {i} is EMPTY DICT!")
                        # Let's see where this is coming from
                        import inspect
                        frame = inspect.currentframe()
                        try:
                            stack_info = []
                            current_frame = frame
                            for j in range(10):  # Get 10 levels of stack
                                if current_frame is None:
                                    break
                                stack_info.append(f"Frame {j}: {current_frame.f_code.co_filename}:{current_frame.f_lineno} in {current_frame.f_code.co_name}")
                                current_frame = current_frame.f_back
                            self.logger.error("Call stack leading to empty dict:")
                            for stack_line in stack_info:
                                self.logger.error(f"     {stack_line}")
                        finally:
                            del frame
                    else:
                        self.logger.info(f"   Feature {i}: dict with keys {list(feature.keys())}")
                else:
                    self.logger.info(f"   Feature {i}: {type(feature)}")
            
            # Call original collator
            return self.original_collator(features)
    
    try:
        # Create trainer using exact same method as test
        trainer = create_trainer_with_coordinator(
            training_args=training_args, 
            config=config
        )
        
        # Wrap the data collator for debugging
        original_collator = trainer.data_collator
        trainer.data_collator = LoggingCollatorWrapper(original_collator, logger)
        
        logger.info("🔍 Testing trainer.train() with exact test conditions...")
        trainer.train()
        logger.info("✅ Training completed successfully!")
        
    except ValueError as e:
        if "TRAINER COMPATIBILITY ISSUE" in str(e):
            logger.error(f"❌ Got the trainer compatibility issue: {e}")
        else:
            logger.error(f"❌ Got different error: {e}")
            raise
    except Exception as e:
        logger.error(f"❌ Got unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_exact_test_conditions()