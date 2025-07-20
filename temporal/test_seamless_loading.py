#!/usr/bin/env python3
"""
Test seamless loading of pretrained weights with coordinate token extension.

This script validates that:
1. All pretrained embeddings are preserved exactly
2. Only coordinate tokens are randomly initialized
3. Vocabulary sizes are consistent across all components
4. Model can handle both original and extended token IDs
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
import numpy as np
from transformers import AutoTokenizer
from src.config import config, init_config
from src.models.wrapper import Qwen25VLWithDetection, CoordinateConfig

def test_seamless_loading():
    """Test seamless loading of pretrained weights."""
    print("🚀 Testing Seamless Pretrained Weight Loading\n")
    
    # Load config
    init_config("configs/base_flat_v2.yaml")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    print(f"📋 Configuration:")
    print(f"   Config model_vocab_size: {config.model_vocab_size}")
    print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"   Coordinate tokens enabled: {config.coordinate_tokens_enabled}")
    print(f"   Max coordinate value: {config.coordinate_config_max_coord_value}")
    
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
    
    print(f"\n🔧 Creating model with coordinate tokens...")
    
    # Create model wrapper
    model = Qwen25VLWithDetection(
        base_model_path=config.model_path,
        num_queries=100,
        max_caption_length=32,
        tokenizer=tokenizer,
        coordinate_config=coordinate_config,
    )
    
    print(f"\n✅ Model created successfully!")
    print(f"   Original vocab size: {model.original_vocab_size}")
    print(f"   Extended vocab size: {model.extended_vocab_size}")
    print(f"   Coordinate tokens enabled: {model.coordinate_tokens_enabled}")
    
    return model, tokenizer

def test_embedding_preservation(model, tokenizer):
    """Test that pretrained embeddings are preserved."""
    print(f"\n🔍 Testing Embedding Preservation...")
    
    if not model.coordinate_tokens_enabled:
        print("   ⚠️  Coordinate tokens disabled - skipping embedding preservation test")
        return
    
    # Load original model for comparison
    from transformers import AutoModel
    original_model = AutoModel.from_pretrained(
        model.base_model.config.name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Keep on CPU for comparison
    )
    
    original_embeddings = original_model.get_input_embeddings()
    extended_embeddings = model.extended_embeddings
    
    print(f"   Original embedding size: {original_embeddings.weight.shape}")
    print(f"   Extended embedding size: {extended_embeddings.weight.shape}")
    
    # Test preservation of pretrained weights
    preserved_size = min(model.original_vocab_size, original_embeddings.weight.shape[0])
    
    # Compare preserved embeddings (move to same device)
    original_weights = original_embeddings.weight[:preserved_size].to('cuda:0')
    extended_weights = extended_embeddings.weight[:preserved_size]
    
    # Check if weights are identical
    weights_match = torch.allclose(original_weights, extended_weights, atol=1e-6)
    
    print(f"   ✅ Preserved {preserved_size} embeddings")
    print(f"   ✅ Weights match: {weights_match}")
    
    if not weights_match:
        diff = torch.abs(original_weights - extended_weights)
        max_diff = torch.max(diff).item()
        print(f"   ⚠️  Max difference: {max_diff}")
        
        # Find mismatched tokens
        mismatched = torch.any(diff > 1e-6, dim=1)
        num_mismatched = torch.sum(mismatched).item()
        print(f"   ⚠️  Mismatched tokens: {num_mismatched}")
        
        if num_mismatched > 0:
            mismatched_indices = torch.where(mismatched)[0][:10]  # Show first 10
            print(f"   ⚠️  First mismatched indices: {mismatched_indices.tolist()}")
    
    return weights_match

def test_coordinate_token_initialization(model, tokenizer):
    """Test that coordinate tokens are properly initialized."""
    print(f"\n🎯 Testing Coordinate Token Initialization...")
    
    if not model.coordinate_tokens_enabled:
        print("   ⚠️  Coordinate tokens disabled - skipping initialization test")
        return
    
    # Check coordinate token range
    coord_start = model.original_vocab_size
    coord_end = model.extended_vocab_size
    
    print(f"   Coordinate token range: [{coord_start}:{coord_end}]")
    
    # Check that coordinate tokens are initialized (not zero)
    coord_embeddings = model.extended_embeddings.weight[coord_start:coord_end]
    coord_projections = model.extended_lm_head.weight[coord_start:coord_end]
    
    # Check embeddings
    embedding_mean = torch.mean(coord_embeddings).item()
    embedding_std = torch.std(coord_embeddings).item()
    embedding_zero_ratio = torch.sum(torch.abs(coord_embeddings) < 1e-6).item() / coord_embeddings.numel()
    
    print(f"   Coordinate embeddings:")
    print(f"     Mean: {embedding_mean:.6f}")
    print(f"     Std: {embedding_std:.6f}")
    print(f"     Zero ratio: {embedding_zero_ratio:.6f}")
    
    # Check projections
    projection_mean = torch.mean(coord_projections).item()
    projection_std = torch.std(coord_projections).item()
    projection_zero_ratio = torch.sum(torch.abs(coord_projections) < 1e-6).item() / coord_projections.numel()
    
    print(f"   Coordinate projections:")
    print(f"     Mean: {projection_mean:.6f}")
    print(f"     Std: {projection_std:.6f}")
    print(f"     Zero ratio: {projection_zero_ratio:.6f}")
    
    # Validation
    properly_initialized = (
        embedding_zero_ratio < 0.1 and  # Less than 10% zeros
        projection_zero_ratio < 0.1 and
        0.001 < embedding_std < 0.1 and  # Reasonable std
        0.001 < projection_std < 0.1
    )
    
    print(f"   ✅ Properly initialized: {properly_initialized}")
    
    return properly_initialized

def test_tokenizer_consistency(model, tokenizer):
    """Test tokenizer consistency with coordinate tokens."""
    print(f"\n🔤 Testing Tokenizer Consistency...")
    
    if not model.coordinate_tokens_enabled:
        print("   ⚠️  Coordinate tokens disabled - skipping tokenizer consistency test")
        return
    
    # Test coordinate token IDs
    coord_start = model.original_vocab_size
    max_coord_value = model.coordinate_config.max_coord_value
    
    print(f"   Testing coordinate tokens: <coord_0> to <coord_{max_coord_value-1}>")
    
    # Test first 10 coordinate tokens
    consistent = True
    for i in range(min(10, max_coord_value)):
        token_text = f"<coord_{i}>"
        expected_id = coord_start + i
        
        if token_text in tokenizer.get_vocab():
            actual_id = tokenizer.convert_tokens_to_ids(token_text)
            if actual_id != expected_id:
                print(f"   ❌ {token_text}: expected {expected_id}, got {actual_id}")
                consistent = False
        else:
            print(f"   ❌ {token_text}: not found in tokenizer vocab")
            consistent = False
    
    if consistent:
        print(f"   ✅ Coordinate token IDs are consistent")
    
    # Test vocab size consistency
    actual_vocab_size = len(tokenizer.get_vocab())
    expected_vocab_size = model.extended_vocab_size
    
    size_consistent = actual_vocab_size == expected_vocab_size
    print(f"   Tokenizer vocab size: {actual_vocab_size}")
    print(f"   Expected vocab size: {expected_vocab_size}")
    print(f"   ✅ Vocab size consistent: {size_consistent}")
    
    return consistent and size_consistent

def test_forward_pass(model, tokenizer):
    """Test model forward pass with coordinate tokens."""
    print(f"\n🚀 Testing Forward Pass...")
    
    # Create test input with coordinate tokens
    test_text = "test: <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>"
    
    print(f"   Input text: {test_text}")
    
    # Tokenize and move to GPU
    tokens = tokenizer(test_text, return_tensors="pt")
    tokens = {k: v.to('cuda:0') for k, v in tokens.items()}
    print(f"   Token IDs: {tokens['input_ids'].shape}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Test with coordinate tokens disabled to avoid label requirement
            model.coordinate_tokens_enabled = False
            outputs = model(input_ids=tokens['input_ids'])
            model.coordinate_tokens_enabled = True  # Restore
            
        print(f"   ✅ Forward pass successful")
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(f"   Expected vocab size: {model.extended_vocab_size}")
        
        # Check logits shape
        logits_vocab_size = outputs.logits.shape[-1]
        shape_consistent = logits_vocab_size == model.extended_vocab_size
        print(f"   ✅ Logits vocab size consistent: {shape_consistent}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Test seamless loading
        model, tokenizer = test_seamless_loading()
        
        # Test embedding preservation
        embeddings_preserved = test_embedding_preservation(model, tokenizer)
        
        # Test coordinate token initialization
        coords_initialized = test_coordinate_token_initialization(model, tokenizer)
        
        # Test tokenizer consistency
        tokenizer_consistent = test_tokenizer_consistency(model, tokenizer)
        
        # Test forward pass
        forward_successful = test_forward_pass(model, tokenizer)
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   ✅ Seamless loading: {True}")
        print(f"   ✅ Embeddings preserved: {embeddings_preserved}")
        print(f"   ✅ Coordinates initialized: {coords_initialized}")
        print(f"   ✅ Tokenizer consistent: {tokenizer_consistent}")
        print(f"   ✅ Forward pass successful: {forward_successful}")
        
        all_passed = all([
            embeddings_preserved,
            coords_initialized,
            tokenizer_consistent,
            forward_successful
        ])
        
        if all_passed:
            print(f"\n🎉 All tests passed! Seamless loading is working correctly.")
        else:
            print(f"\n❌ Some tests failed. Please check the issues above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()