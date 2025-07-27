#!/usr/bin/env python
"""
Debug script to understand what's happening between dataset and collator.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from transformers.training_args import TrainingArguments
from src.config import init_config, load_config
from src.training.trainer_factory import create_trainer_with_coordinator
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_collator_wrapper():
    """Debug the trainer's collator wrapper behavior."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_collator_wrapper")
    
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
    
    # Create trainer to get access to the internals
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
        remove_unused_columns=False,  # This is critical!
    )
    
    trainer = create_trainer_with_coordinator(
        training_args=training_args, 
        config=config
    )
    
    logger.info(f"🔍 Trainer data_collator type: {type(trainer.data_collator)}")
    logger.info(f"🔍 Training args remove_unused_columns: {training_args.remove_unused_columns}")
    
    # Check what the trainer actually does to wrap our collator
    original_collator = trainer.data_collator
    
    # Get the raw dataloader without going through trainer.train()
    train_dataloader = trainer.get_train_dataloader()
    logger.info(f"🔍 Dataloader collate_fn type: {type(train_dataloader.collate_fn)}")
    
    # Let's try to understand the dataloader's collate_fn
    import inspect
    logger.info(f"🔍 Dataloader collate_fn: {train_dataloader.collate_fn}")
    
    # Try calling the dataloader's collate_fn directly with test data
    logger.info("🔍 Testing dataloader's collate_fn directly...")
    
    # Get some test data from the dataset
    dataset = trainer.train_dataset
    test_data = [dataset[0], dataset[1]]  # Get 2 samples
    
    logger.info("🔍 Test data sample keys:")
    for i, sample in enumerate(test_data):
        if isinstance(sample, dict):
            logger.info(f"   Sample {i}: {list(sample.keys())}")
        else:
            logger.info(f"   Sample {i}: {type(sample)}")
    
    try:
        # Call the dataloader's collate_fn directly
        collated_result = train_dataloader.collate_fn(test_data)
        logger.info(f"✅ Dataloader collate_fn succeeded: {type(collated_result)}")
        if isinstance(collated_result, dict):
            logger.info(f"   Result keys: {list(collated_result.keys())}")
    except Exception as e:
        logger.error(f"❌ Dataloader collate_fn failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check if there are any column filtering happening
    logger.info("🔍 Checking for column filtering...")
    if hasattr(trainer, '_signature_columns'):
        logger.info(f"   Trainer signature columns: {trainer._signature_columns}")
    
    if hasattr(trainer, '_set_signature_columns'):
        logger.info("   Trainer has _set_signature_columns method")
    
    # Check training arguments that might affect data processing
    data_related_args = [
        'remove_unused_columns', 'dataloader_drop_last', 'dataloader_num_workers',
        'dataloader_pin_memory', 'dataloader_persistent_workers'
    ]
    
    logger.info("🔍 Data-related training arguments:")
    for arg in data_related_args:
        if hasattr(training_args, arg):
            value = getattr(training_args, arg)
            logger.info(f"   {arg}: {value}")
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_collator_wrapper()