#!/root/miniconda3/envs/ms/bin/python
"""
Test that the geometry token fix works correctly
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
from src.config.global_config import init_config, reset_config

def test_geometry_token_detection():
    """Test that square and line tokens are properly detected after the fix."""
    
    print("🔍 Testing geometry token detection fix...")
    
    # Initialize config
    reset_config()
    config = init_config('configs/bbu_v2.yaml')
    
    # Load model and tokenizer (minimal setup)
    from src.models.model_loader import create_model_and_tokenizer
    
    try:
        model, tokenizer, image_processor = create_model_and_tokenizer()
        print("✅ Model and tokenizer loaded")
        
        # Create chat processor
        from src.chat_processor import ChatProcessor
        chat_processor = ChatProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            enable_simple_tokens=True,
            enable_coordinate_tokens=True,  # Enable legacy coordinate system for testing
        )
        
        # Initialize simple tokens (this should update coordinate manager)
        chat_processor.initialize_simple_tokens(model)
        
        # Check if geometry tokens were added to tokenizer
        geometry_tokens = {
            "square_start": tokenizer.convert_tokens_to_ids("<|square_start|>"),
            "square_end": tokenizer.convert_tokens_to_ids("<|square_end|>"),
            "line_start": tokenizer.convert_tokens_to_ids("<|line_start|>"),
            "line_end": tokenizer.convert_tokens_to_ids("<|line_end|>"),
        }
        
        unk_id = tokenizer.unk_token_id
        print(f"\n📋 Geometry Token IDs:")
        for name, token_id in geometry_tokens.items():
            status = "✅" if token_id != unk_id else "❌"
            print(f"   {status} {name}: {token_id}")
        
        # Check if coordinate manager was updated (if it exists)
        if hasattr(chat_processor, 'coordinate_manager') and chat_processor.coordinate_manager:
            coord_manager = chat_processor.coordinate_manager
            print(f"\n🔧 Coordinate Manager Geometry Tokens:")
            for name, token_id in coord_manager.geometry_token_ids.items():
                print(f"   🎯 {name}: {token_id}")
                
            # Test geometry span detection with fake token sequences
            print(f"\n🧪 Testing geometry span detection...")
            
            # Create fake sequences with different geometry types
            square_seq = torch.tensor([[
                geometry_tokens["square_start"],
                151665, 151666, 151667, 151668, 151669, 151670, 151671, 151672,  # 8 coord tokens
                geometry_tokens["square_end"]
            ]])
            
            line_seq = torch.tensor([[
                geometry_tokens["line_start"], 
                151665, 151666, 151667, 151668, 151669, 151670,  # 6 coord tokens
                geometry_tokens["line_end"]
            ]])
            
            # Test span detection
            square_spans = coord_manager.detect_geometry_spans(square_seq)
            line_spans = coord_manager.detect_geometry_spans(line_seq)
            
            print(f"   🟩 Square spans detected: {square_spans}")
            print(f"   🟦 Line spans detected: {line_spans}")
            
            # Check if spans are detected correctly
            if square_spans and len(square_spans[0]) > 0:
                _, _, geom_type = square_spans[0][0]
                print(f"   ✅ Square geometry detected as: {geom_type}")
            else:
                print(f"   ❌ No square spans detected!")
                
            if line_spans and len(line_spans[0]) > 0:
                _, _, geom_type = line_spans[0][0]
                print(f"   ✅ Line geometry detected as: {geom_type}")
            else:
                print(f"   ❌ No line spans detected!")
        else:
            print(f"\n⚠️ No coordinate manager found - only SimpleTokenManager active")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_geometry_token_detection()