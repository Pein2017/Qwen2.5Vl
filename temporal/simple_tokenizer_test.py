#!/usr/bin/env python3
"""
Simple test to identify Chinese character truncation in tokenization.
"""

import os
import sys
sys.path.append('/data3/Qwen2.5-VL-main')

def test_tokenizer_directly():
    """Test the tokenizer directly without config dependencies."""
    
    try:
        from transformers import Qwen2VLProcessor
        
        # Load tokenizer directly from the model cache
        model_path = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        
        if not os.path.exists(model_path):
            print(f"Model path not found: {model_path}")
            return
            
        processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
        print("✅ Tokenizer loaded successfully")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Vocab size: {len(tokenizer.get_vocab())}")
        
        # Test the problematic texts
        test_cases = [
            "明白！我会仔细学习参考示例中的检测模式、标注风格和判断标准，然后应用到目标图像的检测中。",
            "机柜空间/满载", 
            "BBU基带处理单元/中兴"
        ]
        
        for i, text in enumerate(test_cases):
            print(f"\n=== Test Case {i+1}: '{text}' ===")
            
            # Test encoding/decoding
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens, skip_special_tokens=False)
            
            print(f"Tokens: {tokens}")
            print(f"Decoded: '{decoded}'")
            print(f"Match: {text == decoded}")
            
            # Test first character specifically
            if len(text) > 0:
                first_char = text[0]
                first_tokens = tokenizer.encode(first_char, add_special_tokens=False)
                first_decoded = tokenizer.decode(first_tokens, skip_special_tokens=False)
                print(f"First char '{first_char}' -> {first_tokens} -> '{first_decoded}'")
            
            # Simulate common training scenario: skip first token
            if len(tokens) > 1:
                without_first = tokens[1:]
                without_first_decoded = tokenizer.decode(without_first, skip_special_tokens=False)
                print(f"Skip first token: {without_first} -> '{without_first_decoded}'")
                
            # Check if any tokens are coordinate tokens (problematic range)
            original_vocab_size = 151936  # From logs
            coord_start = 153713          # From logs
            
            for j, token_id in enumerate(tokens):
                if token_id >= coord_start:
                    print(f"⚠️  Token {j} (ID: {token_id}) is in coordinate range!")
                elif token_id >= original_vocab_size:
                    print(f"⚠️  Token {j} (ID: {token_id}) is beyond original vocab!")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer_directly()