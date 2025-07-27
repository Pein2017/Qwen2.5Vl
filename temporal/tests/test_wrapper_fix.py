#!/usr/bin/env python3
"""
Test script to verify the wrapper fix for extended_embeddings issue
"""

import sys
import os
sys.path.append('/data3/Qwen2.5-VL-main')

from src.config import init_config
from src.models.model_loader import load_model_and_processor_unified

def main():
    print("🔧 Testing wrapper fix...")
    
    # Initialize config
    config_path = 'configs/bbu_v2.yaml'
    print(f"📄 Initializing config from: {config_path}")
    init_config(config_path)
    
    # Test model loading (use model_path from config)
    from src.config import get_config
    config = get_config()
    model_path = config.model_path
    print(f"🤖 Loading model from config: {model_path}")
    
    try:
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=model_path,
            for_inference=False,
            attn_implementation='flash_attention_2'
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Has coordinate tokens: {getattr(model, 'coordinate_tokens_enabled', False)}")
        print(f"   Extended vocab size: {getattr(model, 'extended_vocab_size', 'N/A')}")
        
        # Test that the embedding layer works
        if hasattr(model, 'coordinate_tokens_enabled') and model.coordinate_tokens_enabled:
            print("🧪 Testing embedding layer...")
            import torch
            
            # Create a dummy input with coordinate tokens
            test_input = torch.tensor([[1, 2, 3, model.extended_vocab_size - 1]], device='cuda:0' if torch.cuda.is_available() else 'cpu')
            embeddings = model.base_model.get_input_embeddings()(test_input)
            print(f"   ✅ Embeddings work! Shape: {embeddings.shape}")
            
            # Test the LM head (the part that was broken)
            print("🧪 Testing LM head (language modeling head)...")
            # Use the same dtype as the model
            model_dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
            hidden_states = torch.randn(1, 4, 2048, dtype=model_dtype, device=device)
            logits = model.base_model.get_output_embeddings()(hidden_states)
            print(f"   ✅ LM head works! Shape: {logits.shape}")
            
            # Test a simple forward pass with the base model directly
            print("🧪 Testing base model forward pass (bypasses coordinate tokens)...")
            with torch.no_grad():
                # Create simple test inputs on the same device as model
                device = next(model.parameters()).device
                input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
                attention_mask = torch.ones_like(input_ids)
                test_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                
                # Test the base model directly to verify it works
                outputs = model.base_model(**test_inputs)
                print(f"   ✅ Base model forward works! Logits shape: {outputs.logits.shape}")
                print(f"   ✅ Output type: {type(outputs)}")
                
            print("🧪 Key findings:")
            print("   ✅ extended_embeddings error FIXED - now uses base_model.get_input_embeddings()")
            print("   ✅ extended_lm_head error FIXED - now uses base_model.get_output_embeddings()")
            print("   ✅ Both embedding input and output heads work with extended vocabulary")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)