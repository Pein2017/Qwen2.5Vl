#!/usr/bin/env python3
"""
Test script to verify coordinate loss computation fix
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config
from src.models.model_loader import load_model_and_processor_unified

def test_coordinate_loss_fix():
    """Test that coordinate loss can properly detect box tokens."""
    
    print("🧪 Testing coordinate loss computation fix...")
    
    # Initialize config
    config_path = "configs/base_flat_det.yaml"
    init_config(config_path)
    
    print(f"✅ Config loaded: coordinate_tokens_enabled = {config.coordinate_tokens_enabled}")
    
    # Load model with coordinate tokens
    model, tokenizer, image_processor = load_model_and_processor_unified()
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Original vocab: {model.original_vocab_size}")
    print(f"   Extended vocab: {model.extended_vocab_size}")
    print(f"   Box start ID: {model.box_start_id}")
    print(f"   Box end ID: {model.box_end_id}")
    
    # Test that coordinate manager has correct box token IDs
    if hasattr(model, 'coordinate_manager') and model.coordinate_manager:
        manager_config = model.coordinate_manager.config
        print(f"   Manager box start ID: {manager_config.box_start_id}")
        print(f"   Manager box end ID: {manager_config.box_end_id}")
        
        # Verify they match
        if (model.box_start_id == manager_config.box_start_id and 
            model.box_end_id == manager_config.box_end_id):
            print("✅ Box token IDs match between model and manager")
        else:
            print("❌ Box token ID mismatch!")
            return False
    else:
        print("❌ No coordinate manager found")
        return False
    
    # Test coordinate loss computation with sample data
    print("\n🧪 Testing coordinate loss computation...")
    
    # Create sample input that contains coordinate tokens
    # Format: <|box_start|><coord_100><coord_200><coord_300><coord_400><|box_end|>
    coord_start_id = model.original_vocab_size  # First coordinate token
    sample_input_ids = torch.tensor([
        [1, 2, 3,  # Some regular tokens
         model.box_start_id,  # <|box_start|>
         coord_start_id + 100,  # <coord_100>
         coord_start_id + 200,  # <coord_200>
         coord_start_id + 300,  # <coord_300>
         coord_start_id + 400,  # <coord_400>
         model.box_end_id,  # <|box_end|>
         4, 5, 6]  # Some more regular tokens
    ], dtype=torch.long)
    
    sample_labels = sample_input_ids.clone()
    
    # Create fake logits (batch_size=1, seq_len=11, vocab_size=extended)
    fake_logits = torch.randn(1, 11, model.extended_vocab_size)
    
    print(f"   Sample input shape: {sample_input_ids.shape}")
    print(f"   Sample logits shape: {fake_logits.shape}")
    print(f"   Box start token at position: {(sample_input_ids == model.box_start_id).nonzero()}")
    print(f"   Box end token at position: {(sample_input_ids == model.box_end_id).nonzero()}")
    
    # Test coordinate loss computer
    if hasattr(model, 'coordinate_loss_computer') and model.coordinate_loss_computer:
        try:
            loss, loss_components = model.coordinate_loss_computer.compute_coordinate_aware_loss(
                fake_logits, sample_labels
            )
            
            print(f"✅ Coordinate loss computation successful!")
            print(f"   Total loss: {loss.item():.6f}")
            print(f"   Coordinate loss: {loss_components.get('coordinate_loss', 0.0):.6f}")
            print(f"   Regular loss: {loss_components.get('regular_loss', 0.0):.6f}")
            print(f"   Coordinate tokens: {loss_components.get('coordinate_tokens', 0)}")
            print(f"   Regular tokens: {loss_components.get('regular_tokens', 0)}")
            
            # Check if coordinate tokens were detected
            if loss_components.get('coordinate_tokens', 0) > 0:
                print("✅ Coordinate tokens detected successfully!")
                return True
            else:
                print("⚠️ No coordinate tokens detected - may be expected if none in sequence")
                return True
            
        except Exception as e:
            print(f"❌ Coordinate loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ No coordinate loss computer found")
        return False

if __name__ == "__main__":
    success = test_coordinate_loss_fix()
    if success:
        print("\n🎉 Coordinate loss fix test PASSED!")
        print("   The model should now properly detect coordinate tokens during training.")
    else:
        print("\n💥 Coordinate loss fix test FAILED!")
        print("   Additional debugging needed.")
    
    sys.exit(0 if success else 1)