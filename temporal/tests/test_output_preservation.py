#!/usr/bin/env python3
"""Test script to understand why coordinate losses are lost in ModelOutput."""

import sys
import os
sys.path.append('src')

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


def test_output_preservation():
    print("🧪 Testing ModelOutput coordinate loss preservation...")
    
    # Create a custom output class like in our wrapper
    class CoordinateLossOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
        def __init__(self, coordinate_losses=None, **kwargs):
            super().__init__(**kwargs)
            self._coordinate_losses_data = coordinate_losses or {}
            if coordinate_losses:
                for key, value in coordinate_losses.items():
                    setattr(self, key, value)
    
    # Test basic creation
    coordinate_losses = {
        '_coordinate_l1_loss': torch.tensor(10.0),
        '_geometry_focal_loss': torch.tensor(5.0),
    }
    
    output = CoordinateLossOutputWithPast(
        coordinate_losses=coordinate_losses,
        loss=torch.tensor(15.0),
        logits=torch.randn(1, 10, 100),
    )
    
    print(f"✅ Original output has _coordinate_l1_loss: {hasattr(output, '_coordinate_l1_loss')}")
    print(f"✅ Original output _coordinate_l1_loss value: {getattr(output, '_coordinate_l1_loss', 'MISSING')}")
    print(f"✅ Original output id: {id(output)}")
    
    # Test copy operation
    copied_output = output.copy()
    print(f"📋 Copied output has _coordinate_l1_loss: {hasattr(copied_output, '_coordinate_l1_loss')}")
    print(f"📋 Copied output id: {id(copied_output)}")
    
    # Test dictionary operations (what Transformers might do)
    dict_output = dict(output)
    print(f"📚 Dict conversion keys: {list(dict_output.keys())}")
    
    # Test creating new instance from dict
    new_output = Qwen2_5_VLCausalLMOutputWithPast(**dict_output)
    print(f"🆕 New output from dict has _coordinate_l1_loss: {hasattr(new_output, '_coordinate_l1_loss')}")
    print(f"🆕 New output id: {id(new_output)}")
    
    # This is likely what's happening - Transformers is reconstructing from dict!
    print("\n🔍 CONCLUSION: If Transformers uses dict() + new constructor, custom attributes are lost!")


if __name__ == "__main__":
    test_output_preservation()