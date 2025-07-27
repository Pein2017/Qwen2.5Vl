#!/usr/bin/env python
"""
Test script to validate the recovery batch format works with the model.
"""

import sys
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch
from src.config import init_config, load_config
from src.core.data_processor import DataProcessor
from src.models.model_loader import load_model_and_processor_unified
from tests.fixtures import ConfigFactory, SyntheticDataGenerator
from src.logger_utils import configure_global_logging, get_logger

def test_recovery_batch():
    """Test if the recovery batch format works with the model."""
    configure_global_logging(rank=0, world_size=1)
    logger = get_logger("test_recovery_batch")
    
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
    
    # Create recovery batch manually to test it
    logger.info("🔍 Creating recovery batch manually...")
    
    batch_size = 2
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    sequence_length = 8
    
    recovery_batch = {
        # Text inputs - all padding tokens
        "input_ids": torch.full((batch_size, sequence_length), 
                              fill_value=pad_token_id, dtype=torch.long),
        "labels": torch.full((batch_size, sequence_length), 
                           fill_value=-100, dtype=torch.long),  # All ignored
        "attention_mask": torch.zeros((batch_size, sequence_length), dtype=torch.bool),
        
        # Vision inputs - use minimal tokenized vision format that Qwen2.5-VL expects
        "pixel_values": torch.zeros((288, 1176), dtype=torch.bfloat16),  # Minimal flattened vision tokens
        "image_grid_thw": torch.tensor([[1, 12, 12], [1, 12, 12]], dtype=torch.long),  # Standard grid format
        
        # Span data - empty but properly structured
        "teacher_assistant_spans": [[] for _ in range(batch_size)],
        "student_assistant_spans": [[] for _ in range(batch_size)],
    }
    
    logger.info(f"📊 Recovery batch keys: {list(recovery_batch.keys())}")
    for key, value in recovery_batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"   {key}: tensor shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, list):
            logger.info(f"   {key}: list with {len(value)} items")
    
    # Test the recovery batch with the model
    logger.info("🔍 Testing recovery batch with model...")
    device = next(model.parameters()).device
    for key, value in recovery_batch.items():
        if isinstance(value, torch.Tensor):
            recovery_batch[key] = value.to(device)
    
    model.train()
    try:
        with torch.no_grad():
            model_inputs = {
                k: v for k, v in recovery_batch.items()
                if k in ["input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"]
            }
            logger.info(f"📊 Model inputs keys: {list(model_inputs.keys())}")
            
            outputs = model(**model_inputs)
            logger.info(f"✅ Recovery batch worked! Loss: {outputs.loss}")
            
    except Exception as e:
        logger.error(f"❌ Recovery batch failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's also test if we need different dimensions
        logger.info("🔍 Testing with different pixel_values dimensions...")
        
        # Try with batch dimension included
        recovery_batch_v2 = recovery_batch.copy()
        recovery_batch_v2["pixel_values"] = torch.zeros((batch_size, 288, 1176), dtype=torch.bfloat16).to(device)
        
        try:
            model_inputs_v2 = {
                k: v for k, v in recovery_batch_v2.items()
                if k in ["input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"]
            }
            logger.info(f"📊 Model inputs v2 pixel_values shape: {model_inputs_v2['pixel_values'].shape}")
            
            outputs_v2 = model(**model_inputs_v2)
            logger.info(f"✅ Recovery batch v2 worked! Loss: {outputs_v2.loss}")
            
        except Exception as e2:
            logger.error(f"❌ Recovery batch v2 also failed: {e2}")
            
            # Try with different image_grid_thw dimensions
            logger.info("🔍 Testing with different image_grid_thw dimensions...")
            recovery_batch_v3 = recovery_batch_v2.copy()
            recovery_batch_v3["image_grid_thw"] = torch.tensor([[[1, 12, 12]] * batch_size], dtype=torch.long).to(device)
            
            try:
                model_inputs_v3 = {
                    k: v for k, v in recovery_batch_v3.items()
                    if k in ["input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"]
                }
                logger.info(f"📊 Model inputs v3 image_grid_thw shape: {model_inputs_v3['image_grid_thw'].shape}")
                
                outputs_v3 = model(**model_inputs_v3)
                logger.info(f"✅ Recovery batch v3 worked! Loss: {outputs_v3.loss}")
                
            except Exception as e3:
                logger.error(f"❌ Recovery batch v3 also failed: {e3}")
    
    # Cleanup
    data_generator.cleanup()
    config_factory.cleanup_test_configs()

if __name__ == "__main__":
    test_recovery_batch()