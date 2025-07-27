#!/usr/bin/env python
"""
Debug script to understand trainer data flow and the compatibility issue.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from transformers.training_args import TrainingArguments
from src.config import init_config, load_config
from src.training.trainer_factory import create_trainer_with_coordinator
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_data_flow():
    """Debug the data flow to understand why the trainer clears data."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_trainer")
    
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
    
    logger.info("🔍 Testing dataset and collator directly (without trainer)...")
    
    # Load model and processor for actual testing
    from src.models.model_loader import load_model_and_processor_unified
    model, tokenizer, processor = load_model_and_processor_unified(
        model_path=config.model_path,
        for_inference=False,
        attn_implementation=config.attn_implementation,
        config=config,
    )
    
    # Test dataset and collator directly (should work)
    from src.core.data_processor import DataProcessor
    data_processor = DataProcessor(tokenizer, processor, model, config=config)
    train_dataset, eval_dataset = data_processor.create_datasets()
    data_collator = data_processor.create_data_collator()
    
    logger.info(f"📊 Dataset sizes: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    
    # Test direct collation (should work)
    try:
        sample1 = train_dataset[0]
        sample2 = train_dataset[1] if len(train_dataset) > 1 else train_dataset[0]
        
        logger.info(f"🔍 Sample 1 keys: {list(sample1.keys())}")
        logger.info(f"🔍 Sample 2 keys: {list(sample2.keys())}")
        
        batch = data_collator([sample1, sample2])
        logger.info(f"✅ Direct collation works! Batch keys: {list(batch.keys())}")
        
        # Test empty dict scenario (should trigger error)
        try:
            empty_batch = data_collator([{}, {}])
            logger.error("❌ Empty batch should have triggered error!")
        except ValueError as e:
            logger.info(f"✅ Empty batch correctly triggered error: {str(e)[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Direct collation failed: {e}")
        return
    
    logger.info("🔍 Testing trainer data flow...")
    
    # Create minimal training arguments
    training_args = TrainingArguments(
        output_dir=f"{data_root}/debug_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,  # This might be key!
    )
    
    # Create trainer
    trainer = create_trainer_with_coordinator(
        training_args=training_args, 
        config=config
    )
    
    logger.info("🔍 Checking trainer's dataloader setup...")
    
    # Try to get the dataloader manually
    try:
        train_dataloader = trainer.get_train_dataloader()
        logger.info(f"📊 Train dataloader created with {len(train_dataloader)} batches")
        
        # Try to get first batch manually
        logger.info("🔍 Attempting to get first batch...")
        for i, batch in enumerate(train_dataloader):
            logger.info(f"✅ Got batch {i}: {type(batch)}")
            if isinstance(batch, dict):
                logger.info(f"   Batch keys: {list(batch.keys())}")
            else: 
                logger.info(f"   Batch is not dict: {batch}")
            break
            
    except Exception as e:
        logger.error(f"❌ Trainer dataloader failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_data_flow()