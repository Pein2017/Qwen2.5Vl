#!/usr/bin/env python3
"""Test script to verify coordinate losses survive ModelOutput operations."""

import sys
import os
sys.path.append('src')

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


def test_output_fix():
    print("🧪 Testing ModelOutput coordinate loss preservation with fix...")
    
    # Create output and add coordinate losses to its dictionary (like our fix)
    output = Qwen2_5_VLCausalLMOutputWithPast(
        loss=torch.tensor(15.0),
        logits=torch.randn(1, 10, 100),
    )
    
    # Add coordinate losses to the output's dictionary
    output['_coordinate_l1_loss'] = torch.tensor(10.0)
    output['_geometry_focal_loss'] = torch.tensor(5.0)
    
    print(f"✅ Original output has _coordinate_l1_loss: {hasattr(output, '_coordinate_l1_loss')}")
    print(f"✅ Original output _coordinate_l1_loss value: {getattr(output, '_coordinate_l1_loss', 'MISSING')}")
    print(f"✅ Original output id: {id(output)}")
    
    # Test dictionary operations (what Transformers does)
    dict_output = dict(output)
    print(f"📚 Dict conversion keys: {list(dict_output.keys())}")
    print(f"📚 Dict has _coordinate_l1_loss: {'_coordinate_l1_loss' in dict_output}")
    
    # Test creating new instance from dict (Transformers reconstruction)
    new_output = Qwen2_5_VLCausalLMOutputWithPast(**dict_output)
    print(f"🆕 Reconstructed output has _coordinate_l1_loss: {hasattr(new_output, '_coordinate_l1_loss')}")
    print(f"🆕 Reconstructed output _coordinate_l1_loss value: {getattr(new_output, '_coordinate_l1_loss', 'MISSING')}")
    print(f"🆕 Reconstructed output id: {id(new_output)}")
    
    if hasattr(new_output, '_coordinate_l1_loss'):
        print("✅ SUCCESS: Coordinate losses survive Transformers reconstruction!")
    else:
        print("❌ FAILURE: Coordinate losses still lost!")


if __name__ == "__main__":
    test_output_fix()