#!/usr/bin/env python3
"""
Debug script to check text truncation issues in the training data.
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

from src.logger_utils import get_logger

def check_text_encoding():
    """Check for text encoding issues that might cause truncation."""
    
    # Test text that should appear correctly
    expected_texts = [
        "机柜空间/满载",
        "基带处理单元/中兴", 
        "BBU基带处理单元"
    ]
    
    print("=== Text Encoding Test ===")
    for text in expected_texts:
        print(f"Original: {text}")
        print(f"Encoded (UTF-8): {text.encode('utf-8')}")
        print(f"Length: {len(text)} chars")
        
        # Check if first character is being lost
        if len(text) > 1:
            print(f"First char: '{text[0]}' (U+{ord(text[0]):04X})")
            print(f"Without first: '{text[1:]}'")
        print()

def check_tokenizer_issues():
    """Check if tokenizer is causing truncation."""
    print("=== Tokenizer Test ===")
    
    try:
        from transformers import AutoTokenizer
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data3/Qwen2.5-VL-main/model_cache/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        test_texts = [
            "机柜空间/满载",
            "基带处理单元/中兴",
            "BBU基带处理单元"
        ]
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            
            print(f"Original: '{text}'")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"Decoded: '{decoded}'")
            print(f"Match: {text == decoded}")
            print()
            
    except Exception as e:
        print(f"Tokenizer test failed: {e}")

def check_coordinate_token_collision():
    """Check if coordinate tokens are interfering with text."""
    print("=== Coordinate Token Collision Test ===")
    
    # Check if any Chinese characters have token IDs that collide with coordinate tokens
    chinese_chars = "机柜空间基带处理单元中兴BBU"
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            "/data3/Qwen2.5-VL-main/model_cache/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        original_vocab_size = 151936  # From logs
        coordinate_start = 153713     # From logs
        
        for char in chinese_chars:
            token_id = tokenizer.encode(char, add_special_tokens=False)[0] if tokenizer.encode(char, add_special_tokens=False) else None
            
            if token_id:
                if token_id >= coordinate_start:
                    print(f"⚠️  COLLISION: '{char}' -> ID {token_id} (coordinate range)")
                else:
                    print(f"✅ '{char}' -> ID {token_id} (safe)")
            else:
                print(f"❌ '{char}' -> No token ID")
                
    except Exception as e:
        print(f"Collision test failed: {e}")

if __name__ == "__main__":
    check_text_encoding()
    check_tokenizer_issues() 
    check_coordinate_token_collision()