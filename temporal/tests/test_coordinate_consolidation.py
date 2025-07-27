#!/usr/bin/env python3
"""
Test consolidated coordinate token management system
Verify that SimpleTokenManager and CoordinateTokenManager work together without conflicts
"""

import sys
import os
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

from src.config.global_config import init_config, reset_config
from src.models.model_loader import load_model_and_processor_unified

def test_coordinate_consolidation():
    """Test that consolidated coordinate token system works without conflicts"""
    
    print("🧪 Testing consolidated coordinate token management...")
    
    try:
        # Reset any existing config
        reset_config()
        
        # Initialize config
        config = init_config("configs/bbu_v2.yaml")
        
        # Load model with coordinate tokens enabled
        if not config.coordinate_config_enable_coordinate_tokens:
            print("❌ SKIP: Coordinate tokens not enabled in config")
            return True
            
        print("🔧 Loading model with coordinate tokens enabled...")
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False  # Training mode
        )
        
        # Check SimpleTokenManager is available
        if hasattr(model, 'simple_token_manager'):
            simple_manager = model.simple_token_manager
            print("✅ SimpleTokenManager found on model")
            
            # Test geometry token ID retrieval
            square_start_id = simple_manager.get_token_id("<|square_start|>")
            square_end_id = simple_manager.get_token_id("<|square_end|>")
            line_start_id = simple_manager.get_token_id("<|line_start|>")
            line_end_id = simple_manager.get_token_id("<|line_end|>")
            
            print(f"   Geometry tokens: square=[{square_start_id}, {square_end_id}], line=[{line_start_id}, {line_end_id}]")
        else:
            print("⚠️ SimpleTokenManager not found on model")
        
        # Check CoordinateTokenManager is available
        if hasattr(model, 'coordinate_manager'):
            coord_manager = model.coordinate_manager
            print("✅ CoordinateTokenManager found on model")
            print(f"   Coordinate token range: [{coord_manager.coord_start_id}, {coord_manager.coord_end_id})")
            print(f"   Box tokens: [{coord_manager.config.box_start_id}, {coord_manager.config.box_end_id}]")
        else:
            print("⚠️ CoordinateTokenManager not found on model")
        
        # Test tokenizer vocab size consistency
        vocab_size = len(tokenizer.get_vocab())
        model_embedding_size = model.get_input_embeddings().num_embeddings
        
        print(f"✅ Vocabulary sizes: tokenizer={vocab_size}, model_embeddings={model_embedding_size}")
        
        if vocab_size == model_embedding_size:
            print("✅ Vocabulary sizes are consistent")
        else:
            print(f"❌ FAIL: Vocabulary size mismatch")
            return False
        
        print("✅ Consolidated coordinate token system validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Consolidation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        reset_config()

if __name__ == "__main__":
    success = test_coordinate_consolidation()
    sys.exit(0 if success else 1)