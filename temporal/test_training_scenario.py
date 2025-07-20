#!/usr/bin/env python3
"""
Test script to reproduce the training scenario and fix token loss errors.
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
from transformers import AutoTokenizer
from src.config import config, init_config
from src.models.wrapper import Qwen25VLWithDetection, CoordinateConfig

def test_training_scenario():
    """Test the actual training scenario."""
    print("🚀 Testing Training Scenario\n")
    
    # Load config
    init_config("configs/base_flat_v2.yaml")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    print(f"📋 Configuration:")
    print(f"   Config model_vocab_size: {config.model_vocab_size}")
    print(f"   Tokenizer vocab size: {len(tokenizer.get_vocab())}")
    
    # Create coordinate config
    coordinate_config = CoordinateConfig(
        enable_coordinate_tokens=config.coordinate_tokens_enabled,
        max_coord_value=config.coordinate_config_max_coord_value,
        coord_token_init_std=config.coordinate_config_coord_token_init_std,
        coordinate_loss_weight=config.coordinate_config_coordinate_loss_weight,
        regular_loss_weight=config.coordinate_config_regular_loss_weight,
        soft_expectation_temperature=config.coordinate_config_soft_expectation_temperature,
        focal_loss_alpha=config.coordinate_config_focal_loss_alpha,
        focal_loss_gamma=config.coordinate_config_focal_loss_gamma,
    )
    
    print(f"\n🔧 Creating model...")
    
    # Create model wrapper
    model = Qwen25VLWithDetection(
        base_model_path=config.model_path,
        num_queries=100,
        max_caption_length=32,
        tokenizer=tokenizer,
        coordinate_config=coordinate_config,
    )
    
    print(f"✅ Model created successfully!")
    print(f"   Extended vocab size: {model.extended_vocab_size}")
    
    # Test with actual training inputs
    print(f"\n🧪 Testing with training-like inputs...")
    
    # Create realistic training input
    test_input = {
        "input_ids": torch.tensor([[1, 2, 3, 151648, 151665, 151666, 151667, 151668, 151649, 4, 5]], device="cuda:0"),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device="cuda:0"),
        "pixel_values": torch.randn(1, 3, 224, 224, device="cuda:0"),
        "image_grid_thw": torch.tensor([[1, 224, 224]], device="cuda:0"),
    }
    
    original_inputs = {
        "labels": torch.tensor([[1, 2, 3, 151648, 151665, 151666, 151667, 151668, 151649, 4, 5]], device="cuda:0"),
        "ground_truth_objects": [[]],  # Empty for testing
    }
    
    print(f"   Input shape: {test_input['input_ids'].shape}")
    print(f"   Max token ID: {torch.max(test_input['input_ids']).item()}")
    print(f"   Model vocab size: {model.extended_vocab_size}")
    
    # Test forward pass with coordinate tokens ENABLED
    try:
        # Test with coordinate tokens enabled
        model.coordinate_tokens_enabled = True
        print(f"   Testing with coordinate tokens enabled: {model.coordinate_tokens_enabled}")
        
        with torch.no_grad():
            outputs = model(**test_input, **original_inputs)
            
        print(f"   ✅ Coordinate tokens enabled forward pass successful!")
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(f"   Loss: {outputs.loss}")
        
        # Test with coordinate tokens disabled
        model.coordinate_tokens_enabled = False
        print(f"   Testing with coordinate tokens disabled: {model.coordinate_tokens_enabled}")
        
        with torch.no_grad():
            outputs = model(**test_input, **original_inputs)
            
        print(f"   ✅ Coordinate tokens disabled forward pass successful!")
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(f"   Loss: {outputs.loss}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training forward pass failed: {e}")
        
        # Debug the issue
        print(f"\n🔍 Debugging the issue...")
        
        # Check if token IDs are within vocab size
        max_token_id = torch.max(test_input['input_ids']).item()
        print(f"   Max token ID in input: {max_token_id}")
        print(f"   Model vocab size: {model.extended_vocab_size}")
        print(f"   Token ID within range: {max_token_id < model.extended_vocab_size}")
        
        # Check embedding layer size
        if hasattr(model, 'extended_embeddings'):
            print(f"   Extended embeddings size: {model.extended_embeddings.weight.shape[0]}")
        
        # Check LM head size
        if hasattr(model, 'extended_lm_head'):
            print(f"   Extended LM head size: {model.extended_lm_head.weight.shape[0]}")
        
        return False

if __name__ == "__main__":
    try:
        success = test_training_scenario()
        
        if success:
            print(f"\n🎉 Training scenario test passed!")
        else:
            print(f"\n❌ Training scenario test failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()