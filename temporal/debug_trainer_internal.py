#!/usr/bin/env python
"""
Debug script to trace the exact point where the trainer clears data.
We'll add logging to understand what happens inside the trainer's data loading pipeline.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from transformers.training_args import TrainingArguments
from src.config import init_config, load_config
from src.training.trainer_factory import create_trainer_with_coordinator
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_trainer_internal():
    """Debug the trainer's internal data handling."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_trainer_internal")
    
    # Create test data
    data_generator = SyntheticDataGenerator(num_samples=4)
    train_path, val_path, teacher_path, all_samples_path = data_generator.generate_complete_dataset()
    data_root = str(data_generator.temp_dir)
    
    # Create coordinate-enabled configuration
    config_factory = ConfigFactory()
    config_path = config_factory.create_coordinate_enabled_config(data_root, "standard")
    
    # Initialize configuration
    init_config(config_path)
    config = load_config(config_path)
    
    logger.info("🔍 Creating trainer with different settings...")
    
    # Test different training arguments that might affect data handling
    test_configs = [
        {
            "name": "remove_unused_columns_false",
            "args": {
                "remove_unused_columns": False,
                "dataloader_drop_last": False,
                "dataloader_pin_memory": False,
            }
        },
        {
            "name": "remove_unused_columns_true", 
            "args": {
                "remove_unused_columns": True,
                "dataloader_drop_last": False,
                "dataloader_pin_memory": False,
            }
        }
    ]
    
    for test_config in test_configs:
        logger.info(f"🧪 Testing configuration: {test_config['name']}")
        
        # Create training arguments with specific settings
        training_args = TrainingArguments(
            output_dir=f"{data_root}/debug_output_{test_config['name']}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            **test_config["args"]
        )
        
        logger.info(f"   remove_unused_columns: {training_args.remove_unused_columns}")
        logger.info(f"   dataloader_drop_last: {training_args.dataloader_drop_last}")
        logger.info(f"   dataloader_pin_memory: {training_args.dataloader_pin_memory}")
        
        try:
            # Create trainer
            trainer = create_trainer_with_coordinator(
                training_args=training_args, 
                config=config
            )
            
            # Try to get a single training step
            logger.info("   🔍 Testing trainer.train()...")
            trainer.train()
            logger.info(f"   ✅ Configuration {test_config['name']} WORKS!")
            
        except ValueError as e:
            if "TRAINER COMPATIBILITY ISSUE" in str(e):
                logger.error(f"   ❌ Configuration {test_config['name']} failed with trainer compatibility issue")
            else:
                logger.error(f"   ❌ Configuration {test_config['name']} failed with other error: {e}")
        except Exception as e:
            logger.error(f"   ❌ Configuration {test_config['name']} failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    # If we get here without finding a working config, let's try a different approach
    logger.info("🔍 Testing custom data collator wrapper...")
    
    # Let's try wrapping the collator to see what data it receives
    class DebuggingCollatorWrapper:
        def __init__(self, original_collator):
            self.original_collator = original_collator
            self.call_count = 0
            
        def __call__(self, features):
            self.call_count += 1
            logger.info(f"🔍 Collator call #{self.call_count}")
            logger.info(f"   Received {len(features)} features")
            for i, feature in enumerate(features):
                if isinstance(feature, dict):
                    logger.info(f"   Feature {i}: dict with keys {list(feature.keys())}")
                    if not feature:
                        logger.error(f"   ❌ Feature {i} is EMPTY DICT!")
                else:
                    logger.info(f"   Feature {i}: {type(feature)}")
            
            # Call original collator
            return self.original_collator(features)
    
    # Test with wrapped collator
    training_args = TrainingArguments(
        output_dir=f"{data_root}/debug_output_wrapped",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    try:
        trainer = create_trainer_with_coordinator(
            training_args=training_args, 
            config=config
        )
        
        # Wrap the data collator
        original_collator = trainer.data_collator
        trainer.data_collator = DebuggingCollatorWrapper(original_collator)
        
        logger.info("🔍 Testing with wrapped collator...")
        trainer.train()
        logger.info("✅ Wrapped collator test WORKED!")
        
    except Exception as e:
        logger.error(f"❌ Wrapped collator test failed: {e}")
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_trainer_internal()