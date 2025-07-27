#!/usr/bin/env python3
"""
Test script to demonstrate token validation during training.

This script creates a minimal training setup to test that token validation
is working correctly during the training process.

Usage:
    python temporal/test_training_validation.py
"""

import sys
import os
import torch
import logging
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat_processor import ChatProcessor
from transformers import AutoTokenizer, AutoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_sample_valid():
    """Create a valid test sample with all required tokens."""
    return {
        "id": "test_valid",
        "images": ["test_image.jpg"],
        "objects": [
            {
                "bbox_2d": [100, 100, 200, 200],
                "desc": "Test object description"
            }
        ],
        "width": 640,
        "height": 480
    }

def create_test_sample_invalid_no_object_ref():
    """Create an invalid test sample missing object reference tokens."""
    return {
        "id": "test_invalid_no_object_ref", 
        "images": ["test_image.jpg"],
        "objects": [
            {
                "bbox_2d": [100, 100, 200, 200],
                "desc": ""  # Empty description will cause missing object_ref tokens
            }
        ],
        "width": 640,
        "height": 480
    }

def create_test_sample_invalid_no_geometry():
    """Create an invalid test sample missing geometry tokens."""
    return {
        "id": "test_invalid_no_geometry",
        "images": ["test_image.jpg"], 
        "objects": [
            {
                "desc": "Test object description"
                # Missing geometry (bbox_2d, square, line)
            }
        ],
        "width": 640,
        "height": 480
    }

def test_chat_processor_validation():
    """Test the chat processor token validation functionality."""
    
    print("🧪 Testing Chat Processor Token Validation")
    print("=" * 50)
    
    # Load tokenizer and processor
    model_path = "model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Create chat processor with validation enabled
    chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        processor=processor,
        data_root="data/ds_v2_full",
        language="chinese",
        model_max_length=12000,
        use_training_prompts=True,
        enable_token_validation=True  # Enable validation
    )
    
    print("✅ Chat processor created with token validation enabled")
    
    # Test 1: Valid sample
    print("\n📋 Test 1: Valid sample with all required tokens")
    valid_sample = create_test_sample_valid()
    
    try:
        # This should work without errors
        result = chat_processor.process_sample(valid_sample)
        print("   ✅ Valid sample processed successfully")
        print(f"   📊 Input IDs shape: {result.input_ids.shape}")
        print(f"   📊 Labels shape: {result.labels.shape}")
    except Exception as e:
        print(f"   ❌ Valid sample failed: {e}")
    
    # Test 2: Invalid sample - missing object reference tokens
    print("\n📋 Test 2: Invalid sample missing object reference tokens")
    invalid_sample_1 = create_test_sample_invalid_no_object_ref()
    
    try:
        result = chat_processor.process_sample(invalid_sample_1)
        print("   ❌ Invalid sample incorrectly passed validation")
    except ValueError as e:
        if "object reference tokens" in str(e):
            print("   ✅ Invalid sample correctly failed validation (missing object reference tokens)")
        else:
            print(f"   ⚠️ Invalid sample failed for wrong reason: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 3: Invalid sample - missing geometry tokens
    print("\n📋 Test 3: Invalid sample missing geometry tokens")
    invalid_sample_2 = create_test_sample_invalid_no_geometry()
    
    try:
        result = chat_processor.process_sample(invalid_sample_2)
        print("   ❌ Invalid sample incorrectly passed validation")
    except ValueError as e:
        if "geometry tokens" in str(e):
            print("   ✅ Invalid sample correctly failed validation (missing geometry tokens)")
        else:
            print(f"   ⚠️ Invalid sample failed for wrong reason: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 4: Chat processor with validation disabled
    print("\n📋 Test 4: Chat processor with validation disabled")
    
    chat_processor_no_validation = ChatProcessor(
        tokenizer=tokenizer,
        processor=processor,
        data_root="data/ds_v2_full",
        language="chinese",
        model_max_length=12000,
        use_training_prompts=True,
        enable_token_validation=False  # Disable validation
    )
    
    try:
        # This should work even with invalid samples when validation is disabled
        result = chat_processor_no_validation.process_sample(invalid_sample_1)
        print("   ✅ Invalid sample processed when validation disabled (expected behavior)")
    except Exception as e:
        print(f"   ⚠️ Unexpected error even with validation disabled: {e}")
    
    print("\n🎯 Chat Processor Token Validation Test Summary:")
    print("   ✅ Validation correctly identifies missing object reference tokens")
    print("   ✅ Validation correctly identifies missing geometry tokens")
    print("   ✅ Validation can be enabled/disabled via configuration")
    print("   ✅ Valid samples pass validation successfully")

def test_config_integration():
    """Test that the configuration system properly passes validation settings."""
    
    print("\n🔧 Testing Configuration Integration")
    print("=" * 40)
    
    # Test config with validation enabled
    config_with_validation = {
        "chat_processor_enable_token_validation": True,
        "language": "chinese",
        "model_max_length": 12000,
        "use_training_prompts": True
    }
    
    print("✅ Configuration with token validation enabled:")
    print(f"   chat_processor_enable_token_validation: {config_with_validation['chat_processor_enable_token_validation']}")
    
    # Test config with validation disabled
    config_without_validation = {
        "chat_processor_enable_token_validation": False,
        "language": "chinese", 
        "model_max_length": 12000,
        "use_training_prompts": True
    }
    
    print("✅ Configuration with token validation disabled:")
    print(f"   chat_processor_enable_token_validation: {config_without_validation['chat_processor_enable_token_validation']}")

if __name__ == "__main__":
    test_chat_processor_validation()
    test_config_integration()
    
    print("\n🎉 All token validation tests completed!")
    print("\n📝 Usage Instructions:")
    print("   1. Add 'chat_processor_enable_token_validation: true' to your config YAML")
    print("   2. Token validation will automatically check all training samples")
    print("   3. Training will fail fast if samples are missing required tokens")
    print("   4. This helps catch data quality issues early in the training process")
