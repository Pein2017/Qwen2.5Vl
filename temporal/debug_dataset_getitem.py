#!/usr/bin/env python
"""
Debug script to understand why dataset.__getitem__ returns empty dicts in trainer context.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from transformers.training_args import TrainingArguments
from src.config import init_config, load_config
from src.training.trainer_factory import create_trainer_with_coordinator
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_dataset_getitem():
    """Debug the dataset __getitem__ method."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_dataset")
    
    # Use same parameters as the integration test
    data_generator = SyntheticDataGenerator(num_samples=8)  
    train_path, val_path, teacher_path, all_samples_path = data_generator.generate_complete_dataset()
    data_root = str(data_generator.temp_dir)
    
    # Create coordinate-enabled configuration
    config_factory = ConfigFactory()
    config_path = config_factory.create_coordinate_enabled_config(data_root, "standard")
    
    # Initialize configuration
    init_config(config_path)
    config = load_config(config_path)
    
    # Create trainer to get access to the dataset
    training_args = TrainingArguments(
        output_dir=f"{data_root}/debug_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=2,
        logging_steps=1,
        eval_steps=2,
        eval_strategy="steps",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
    )
    
    trainer = create_trainer_with_coordinator(
        training_args=training_args, 
        config=config
    )
    
    # Get the dataset
    train_dataset = trainer.train_dataset
    logger.info(f"📊 Dataset size: {len(train_dataset)}")
    logger.info(f"📊 Dataset type: {type(train_dataset)}")
    
    # Create a wrapper for the dataset to log calls
    class DatasetWrapper:
        def __init__(self, original_dataset, logger):
            self.original_dataset = original_dataset
            self.logger = logger
            self.call_count = 0
            
        def __len__(self):
            return len(self.original_dataset)
            
        def __getitem__(self, idx):
            self.call_count += 1
            self.logger.info(f"🔍 Dataset.__getitem__({idx}) - Call #{self.call_count}")
            
            try:
                # Call original dataset
                result = self.original_dataset[idx]
                
                if isinstance(result, dict):
                    if not result:
                        self.logger.error(f"   ❌ Dataset returned EMPTY DICT for index {idx}!")
                        # Let's debug the original dataset state
                        self.logger.error(f"   Original dataset type: {type(self.original_dataset)}")
                        self.logger.error(f"   Original dataset length: {len(self.original_dataset)}")
                        
                        # Try to access the underlying data
                        if hasattr(self.original_dataset, 'data'):
                            self.logger.error(f"   Original dataset.data length: {len(getattr(self.original_dataset, 'data', []))}")
                        
                        # Get stack trace to see where this is called from
                        import inspect
                        frame = inspect.currentframe()
                        try:
                            stack_info = []
                            current_frame = frame
                            for j in range(15):  # Get 15 levels of stack
                                if current_frame is None:
                                    break
                                stack_info.append(f"Frame {j}: {current_frame.f_code.co_filename}:{current_frame.f_lineno} in {current_frame.f_code.co_name}")
                                current_frame = current_frame.f_back
                            self.logger.error("Call stack leading to empty dataset result:")
                            for stack_line in stack_info:
                                self.logger.error(f"     {stack_line}")
                        finally:
                            del frame
                            
                        return result  # Return the empty dict to trigger the error
                    else:
                        self.logger.info(f"   ✅ Dataset returned dict with keys: {list(result.keys())}")
                else:
                    self.logger.info(f"   ✅ Dataset returned: {type(result)}")
                    
                return result
                
            except Exception as e:
                self.logger.error(f"   ❌ Dataset.__getitem__({idx}) failed: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # Test direct access first
    logger.info("🔍 Testing direct dataset access...")
    for i in range(min(4, len(train_dataset))):
        try:
            item = train_dataset[i]
            if isinstance(item, dict) and item:
                logger.info(f"   Direct access [{i}]: ✅ dict with keys {list(item.keys())}")
            else:
                logger.error(f"   Direct access [{i}]: ❌ empty or invalid: {type(item)}")
        except Exception as e:
            logger.error(f"   Direct access [{i}]: ❌ exception: {e}")
    
    # Now test with trainer dataloader
    logger.info("🔍 Testing with trainer dataloader...")
    
    # Wrap the dataset
    wrapped_dataset = DatasetWrapper(train_dataset, logger)
    trainer.train_dataset = wrapped_dataset
    
    # Try to get a batch manually using the trainer's dataloader
    try:
        train_dataloader = trainer.get_train_dataloader()
        logger.info(f"📊 Dataloader created with {len(train_dataloader)} batches")
        
        # Get first batch
        logger.info("🔍 Getting first batch...")
        for i, batch in enumerate(train_dataloader):
            logger.info(f"   Batch {i}: {type(batch)} with keys {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}")
            break
            
    except Exception as e:
        logger.error(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_dataset_getitem()