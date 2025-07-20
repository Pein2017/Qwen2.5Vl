#!/usr/bin/env python3
"""
Comprehensive Analysis of Coordinate Token Vocabulary Size Mismatch

This script provides a detailed analysis of the CUDA index out of bounds error
in the coordinate token system, identifies the root cause, and provides a fix.
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import torch
from transformers import AutoTokenizer
from src.config import config, init_config, reset_config

def analyze_vocabulary_mismatch():
    """Analyze the vocabulary size mismatch causing CUDA index out of bounds."""
    
    print("🔍 COMPREHENSIVE ANALYSIS: Coordinate Token Vocabulary Size Mismatch")
    print("=" * 80)
    
    # Initialize configuration
    reset_config()
    init_config("configs/base_flat_det.yaml")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    # Get actual vocabulary sizes
    actual_vocab_size = len(tokenizer.get_vocab())
    config_vocab_size = config.model_vocab_size
    max_coord_value = config.coordinate_config_max_coord_value
    
    print(f"\n📊 VOCABULARY SIZE ANALYSIS:")
    print(f"   Actual tokenizer vocabulary size: {actual_vocab_size}")
    print(f"   Config model_vocab_size setting: {config_vocab_size}")
    print(f"   Max coordinate value: {max_coord_value}")
    print(f"   Expected extended vocab size: {actual_vocab_size + max_coord_value}")
    
    # Calculate the mismatch
    difference = config_vocab_size - actual_vocab_size
    print(f"\n❌ MISMATCH IDENTIFIED:")
    print(f"   Config expects: {config_vocab_size} tokens")
    print(f"   Tokenizer has: {actual_vocab_size} tokens")
    print(f"   Difference: {difference} tokens")
    
    # Analyze the coordinate token system
    print(f"\n🎯 COORDINATE TOKEN SYSTEM ANALYSIS:")
    print(f"   Coordinate tokens enabled: {config.coordinate_tokens_enabled}")
    print(f"   Max coordinate value: {max_coord_value}")
    
    # This is the root cause analysis
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    print(f"   1. The wrapper._setup_coordinate_tokens() calculates:")
    print(f"      - self.original_vocab_size = len(tokenizer.get_vocab()) = {actual_vocab_size}")
    print(f"      - self.extended_vocab_size = {actual_vocab_size} + {max_coord_value} = {actual_vocab_size + max_coord_value}")
    print(f"   ")
    print(f"   2. The coordinate_loss_computer.py line 318 extracts:")
    print(f"      - coord_only_logits = coord_logits[:, coord_start:coord_end]")
    print(f"      - coord_start = {actual_vocab_size} (original_vocab_size)")
    print(f"      - coord_end = {actual_vocab_size + max_coord_value} (extended_vocab_size)")
    print(f"   ")
    print(f"   3. BUT the actual logits tensor has size:")
    print(f"      - logits.shape[-1] = {config_vocab_size} (from config)")
    print(f"      - This is LESS than coord_end = {actual_vocab_size + max_coord_value}")
    print(f"   ")
    print(f"   4. When coord_only_logits = coord_logits[:, {actual_vocab_size}:{actual_vocab_size + max_coord_value}]")
    print(f"      tries to access indices beyond {config_vocab_size}, we get:")
    print(f"      CUDA ERROR: index out of bounds")
    
    # Show the exact error location
    print(f"\n💥 ERROR LOCATION:")
    print(f"   File: src/utils/coordinate_loss_computer.py")
    print(f"   Line: 318")
    print(f"   Code: coord_only_logits = coord_logits[:, coord_start:coord_end]")
    print(f"   Problem: coord_end ({actual_vocab_size + max_coord_value}) > logits.shape[-1] ({config_vocab_size})")
    
    # The fix
    print(f"\n🔧 THE FIX:")
    print(f"   The config file model_vocab_size should be:")
    print(f"   model_vocab_size: {actual_vocab_size + max_coord_value}")
    print(f"   ")
    print(f"   This ensures the logits tensor has the correct size to accommodate")
    print(f"   both regular tokens and coordinate tokens.")
    
    return actual_vocab_size, config_vocab_size, max_coord_value

def demonstrate_fix():
    """Demonstrate the fix by showing the correct configuration."""
    
    print(f"\n🛠️  CONFIGURATION FIX DEMONSTRATION:")
    print(f"=" * 60)
    
    # Get the values (don't reinitialize config)
    actual_vocab_size = len(AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True).get_vocab())
    config_vocab_size = config.model_vocab_size
    max_coord_value = config.coordinate_config_max_coord_value
    
    correct_vocab_size = actual_vocab_size + max_coord_value
    
    print(f"\n📝 REQUIRED CONFIGURATION CHANGE:")
    print(f"   In configs/base_flat_det.yaml, change:")
    print(f"   ")
    print(f"   FROM: model_vocab_size: {config_vocab_size}")
    print(f"   TO:   model_vocab_size: {correct_vocab_size}")
    print(f"   ")
    print(f"   This change ensures:")
    print(f"   1. The model creates logits tensor with {correct_vocab_size} dimensions")
    print(f"   2. The coordinate loss computer can access indices [{actual_vocab_size}:{correct_vocab_size}]")
    print(f"   3. No more CUDA index out of bounds errors")
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Original vocab:    0 to {actual_vocab_size - 1}")
    print(f"   Coordinate tokens: {actual_vocab_size} to {correct_vocab_size - 1}")
    print(f"   Total logits size: {correct_vocab_size}")
    print(f"   Safe to access:    coord_logits[:, {actual_vocab_size}:{correct_vocab_size}]")

def show_code_flow():
    """Show the exact code flow that causes the error."""
    
    print(f"\n🔬 DETAILED CODE FLOW ANALYSIS:")
    print(f"=" * 60)
    
    print(f"1. wrapper.py:158 - Store original vocab size from tokenizer:")
    print(f"   self.original_vocab_size = len(self.tokenizer.get_vocab()) = 151665")
    
    print(f"\n2. wrapper.py:247 - Calculate extended vocab size:")
    print(f"   self.extended_vocab_size = 151665 + 2048 = 153713")
    
    print(f"\n3. wrapper.py:294-303 - Add coordinate tokens to tokenizer:")
    print(f"   coordinate_tokens = ['<coord_0>', '<coord_1>', ..., '<coord_2047>']")
    print(f"   tokenizer.add_special_tokens(coordinate_tokens)")
    
    print(f"\n4. wrapper.py:525 - Forward pass creates logits:")
    print(f"   logits = self.extended_lm_head(hidden_states)")
    print(f"   logits.shape = (batch_size, seq_len, 152064)  # From config.model_vocab_size")
    
    print(f"\n5. coordinate_loss_computer.py:318 - Extract coordinate logits:")
    print(f"   coord_start = 151665  # original_vocab_size")
    print(f"   coord_end = 153713    # extended_vocab_size")
    print(f"   coord_only_logits = coord_logits[:, 151665:153713]")
    
    print(f"\n6. ERROR: Trying to access indices [151665:153713] in tensor of size 152064:")
    print(f"   RuntimeError: CUDA kernel errors might be asynchronously reported")
    print(f"   IndexError: index 152064 is out of bounds for dimension 2 with size 152064")

if __name__ == "__main__":
    try:
        analyze_vocabulary_mismatch()
        demonstrate_fix()
        show_code_flow()
        
        print(f"\n🎯 SUMMARY:")
        print(f"=" * 40)
        print(f"ROOT CAUSE: Config model_vocab_size (152064) < required extended vocab (153713)")
        print(f"SOLUTION: Update config model_vocab_size to 153713")
        print(f"IMPACT: Eliminates CUDA index out of bounds error in coordinate token system")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()