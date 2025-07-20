#!/usr/bin/env python3
"""
Debug vocabulary mismatch causing CUDA index out of bounds error.

This script investigates the vocabulary size mismatch between:
1. Config file specification
2. Actual tokenizer vocabulary
3. Model embedding layer size
4. Coordinate token extension
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
import yaml
from transformers import AutoTokenizer
from src.config import config, init_config

def investigate_vocab_mismatch():
    """Investigate vocabulary size mismatch."""
    print("🔍 Investigating vocabulary size mismatch...")
    
    # Load config
    print("\n📋 Loading configuration...")
    init_config("configs/base_flat_det.yaml")
    
    # Get config values
    config_vocab_size = config.model_vocab_size
    print(f"   Config model_vocab_size: {config_vocab_size}")
    
    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    actual_vocab_size = len(tokenizer.get_vocab())
    print(f"   Actual tokenizer vocab size: {actual_vocab_size}")
    
    # Check mismatch
    if config_vocab_size != actual_vocab_size:
        print(f"❌ VOCAB SIZE MISMATCH FOUND!")
        print(f"   Config expects: {config_vocab_size}")
        print(f"   Tokenizer has: {actual_vocab_size}")
        print(f"   Difference: {config_vocab_size - actual_vocab_size}")
    else:
        print(f"✅ Vocab sizes match")
    
    # Check coordinate token configuration
    print(f"\n🎯 Checking coordinate token configuration...")
    coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
    print(f"   Coordinate tokens enabled: {coordinate_tokens_enabled}")
    
    if coordinate_tokens_enabled:
        max_coord_value = getattr(config, 'coordinate_config_max_coord_value', 2048)
        print(f"   Max coordinate value: {max_coord_value}")
        
        extended_vocab_size = actual_vocab_size + max_coord_value
        print(f"   Extended vocab size: {extended_vocab_size}")
        
        # Check if model expects extended vocab
        if config_vocab_size == extended_vocab_size:
            print(f"✅ Config expects extended vocabulary")
        elif config_vocab_size == actual_vocab_size:
            print(f"❌ Config expects original vocabulary but coordinate tokens are enabled!")
        else:
            print(f"❌ Config vocab size doesn't match either original or extended!")
    
    return config_vocab_size, actual_vocab_size, coordinate_tokens_enabled

def check_model_embedding_size():
    """Check the model's actual embedding layer size."""
    print("\n🤖 Checking model embedding layer...")
    
    try:
        from transformers import AutoModel
        
        # Load model to check embedding size
        model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Get embedding layer
        embeddings = model.get_input_embeddings()
        embedding_vocab_size = embeddings.weight.shape[0]
        
        print(f"   Model embedding vocab size: {embedding_vocab_size}")
        
        return embedding_vocab_size
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def check_coordinate_token_setup():
    """Check coordinate token setup in wrapper."""
    print("\n🎯 Checking coordinate token setup in wrapper...")
    
    # Check if coordinate tokens are configured
    coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
    
    if not coordinate_tokens_enabled:
        print("   ❌ coordinate_tokens_enabled not found in config")
        return False
        
    # Get coordinate token configuration
    max_coord_value = getattr(config, 'coordinate_config_max_coord_value', None)
    if max_coord_value is None:
        print("   ❌ coordinate_config_max_coord_value not found in config")
        return False
        
    print(f"   ✅ Coordinate tokens enabled with max_coord_value: {max_coord_value}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    original_vocab_size = len(tokenizer.get_vocab())
    extended_vocab_size = original_vocab_size + max_coord_value
    
    print(f"   Original vocab size: {original_vocab_size}")
    print(f"   Extended vocab size: {extended_vocab_size}")
    
    # Check if config matches extended size
    config_vocab_size = config.model_vocab_size
    if config_vocab_size == extended_vocab_size:
        print(f"   ✅ Config vocab size matches extended vocabulary")
        return True
    else:
        print(f"   ❌ Config vocab size ({config_vocab_size}) doesn't match extended vocabulary ({extended_vocab_size})")
        return False

def suggest_fixes():
    """Suggest fixes for the vocabulary mismatch."""
    print("\n🔧 Suggested fixes:")
    
    # Get actual tokenizer vocab size
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    actual_vocab_size = len(tokenizer.get_vocab())
    
    # Check if coordinate tokens are enabled
    coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
    
    if coordinate_tokens_enabled:
        max_coord_value = getattr(config, 'coordinate_config_max_coord_value', 2048)
        extended_vocab_size = actual_vocab_size + max_coord_value
        
        print(f"1. Update config model_vocab_size to: {extended_vocab_size}")
        print(f"   (Original: {actual_vocab_size} + Coordinate tokens: {max_coord_value})")
    else:
        print(f"1. Update config model_vocab_size to: {actual_vocab_size}")
        print(f"   (Match actual tokenizer vocabulary)")
    
    print(f"2. Ensure coordinate token configuration is consistent:")
    print(f"   - coordinate_tokens_enabled: {coordinate_tokens_enabled}")
    if coordinate_tokens_enabled:
        print(f"   - coordinate_config_max_coord_value: {getattr(config, 'coordinate_config_max_coord_value', 'NOT SET')}")
    
    print(f"3. Check model wrapper initialization:")
    print(f"   - Ensure extended vocabulary is properly configured")
    print(f"   - Verify coordinate tokens are added to tokenizer")
    print(f"   - Check model embedding layer is resized correctly")

if __name__ == "__main__":
    print("🚀 Debugging Vocabulary Mismatch Issue\n")
    
    try:
        # Step 1: Investigate vocabulary mismatch
        config_vocab, actual_vocab, coord_enabled = investigate_vocab_mismatch()
        
        # Step 2: Check model embedding size
        embedding_vocab = check_model_embedding_size()
        
        # Step 3: Check coordinate token setup
        coord_setup_ok = check_coordinate_token_setup()
        
        # Step 4: Suggest fixes
        suggest_fixes()
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Config vocab size: {config_vocab}")
        print(f"   Actual vocab size: {actual_vocab}")
        print(f"   Model embedding size: {embedding_vocab}")
        print(f"   Coordinate tokens enabled: {coord_enabled}")
        print(f"   Coordinate setup OK: {coord_setup_ok}")
        
        if config_vocab != actual_vocab:
            print(f"\n❌ CRITICAL: Vocabulary size mismatch detected!")
            print(f"   This is causing the CUDA index out of bounds error.")
            print(f"   The model is trying to access token IDs beyond the vocabulary size.")
        else:
            print(f"\n✅ Vocabulary sizes match - issue may be elsewhere")
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()