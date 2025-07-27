#!/usr/bin/env python
"""
Debug script to understand the exact vision data format that Qwen2.5-VL expects.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from src.config import init_config, load_config
from src.core.data_processor import DataProcessor
from src.models.model_loader import load_model_and_processor_unified
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def debug_vision_format():
    """Debug the exact vision format expected by Qwen2.5-VL."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("debug_vision_format")
    
    # Create test data and config
    data_generator = SyntheticDataGenerator(num_samples=8)
    train_path, val_path, teacher_path, all_samples_path = data_generator.generate_complete_dataset()
    data_root = str(data_generator.temp_dir)
    
    config_factory = ConfigFactory()
    config_path = config_factory.create_coordinate_enabled_config(data_root, "standard")
    
    # Initialize configuration
    init_config(config_path)
    config = load_config(config_path)
    
    # Load model and create data processor
    model, tokenizer, processor = load_model_and_processor_unified(
        model_path=config.model_path,
        for_inference=False,
        attn_implementation=config.attn_implementation,
    )
    
    data_processor = DataProcessor(tokenizer, processor, model, config)
    train_dataset, _ = data_processor.create_datasets()
    data_collator = data_processor.create_data_collator()
    
    # Get a normal sample and batch it
    logger.info("🔍 Examining normal sample format...")
    normal_sample = train_dataset[0]
    
    logger.info(f"📊 Normal sample keys: {list(normal_sample.keys())}")
    for key, value in normal_sample.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"   {key}: tensor shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, list):
            logger.info(f"   {key}: list with {len(value)} items")
            if value and isinstance(value[0], (list, tuple)):
                logger.info(f"      First item: {value[0]}")
        else:
            logger.info(f"   {key}: {type(value)} = {value}")
    
    # Create a normal batch
    logger.info("🔍 Examining normal batch format...")
    normal_batch = data_collator([normal_sample])
    
    logger.info(f"📊 Normal batch keys: {list(normal_batch.keys())}")
    for key, value in normal_batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"   {key}: tensor shape {value.shape}, dtype {value.dtype}")
            if key == "image_grid_thw":
                logger.info(f"      image_grid_thw values: {value}")
        elif isinstance(value, list):
            logger.info(f"   {key}: list with {len(value)} items")
            if value:
                logger.info(f"      First item: {value[0]}")
        else:
            logger.info(f"   {key}: {type(value)} = {value}")
    
    # Test the normal batch with the model to understand the expected format
    logger.info("🔍 Testing normal batch with model...")
    device = next(model.parameters()).device
    for key, value in normal_batch.items():
        if isinstance(value, torch.Tensor):
            normal_batch[key] = value.to(device)
    
    model.train()
    try:
        with torch.no_grad():
            model_inputs = {
                k: v for k, v in normal_batch.items()
                if k in ["input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"]
            }
            logger.info(f"📊 Model inputs keys: {list(model_inputs.keys())}")
            
            # Let's see what pixel_values and image_grid_thw look like in detail
            if "pixel_values" in model_inputs:
                pv = model_inputs["pixel_values"]
                logger.info(f"   pixel_values: shape {pv.shape}, min {pv.min():.3f}, max {pv.max():.3f}")
                
            if "image_grid_thw" in model_inputs:
                igt = model_inputs["image_grid_thw"]
                logger.info(f"   image_grid_thw: shape {igt.shape}, values {igt}")
            
            outputs = model(**model_inputs)
            logger.info(f"✅ Normal batch worked! Loss: {outputs.loss}")
            
    except Exception as e:
        logger.error(f"❌ Normal batch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    debug_vision_format()