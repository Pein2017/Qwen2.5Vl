#!/usr/bin/env python3
"""
Test script for the unified model loading mechanism.

This script demonstrates how to use the new unified loading system
that automatically detects checkpoint types and loads appropriately.

Usage:
    cd /data4/Qwen2.5-VL-main
    python notebook/test_unified_loading.py <model_path>
"""

import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

from src.models.wrapper import Qwen25VLWithDetection


def test_checkpoint_inspection(model_path: str):
    """Test checkpoint inspection functionality."""
    print("🔍 CHECKPOINT INSPECTION TEST")
    print("=" * 50)

    # Print detailed checkpoint information
    Qwen25VLWithDetection.print_checkpoint_info(model_path)

    # Get programmatic information
    info = Qwen25VLWithDetection.inspect_checkpoint(model_path)
    print(f"Programmatic info: {info}")
    print()


def test_unified_loading(model_path: str):
    """Test unified loading mechanism."""
    print("🚀 UNIFIED LOADING TEST")
    print("=" * 50)

    # Load tokenizer (needed for model)
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
        )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # Test unified loading
    print(f"\nLoading model using unified mechanism...")
    try:
        model = Qwen25VLWithDetection.from_pretrained(
            model_path=model_path,
            num_queries=50,  # Will be overridden if checkpoint has config
            max_caption_length=32,  # Will be overridden if checkpoint has config
            tokenizer=tokenizer,
        )
        print("✅ Model loaded successfully")

        # Print model information
        print(f"\nModel Information:")
        print(f"   Base model device: {model.base_model.device}")
        print(f"   Detection head device: {model.detection_head.device}")
        print(f"   Detection queries: {model.detection_head.num_queries}")
        print(f"   Max caption length: {model.detection_head.max_caption_length}")
        print(f"   Hidden size: {model.detection_head.hidden_size}")
        print(f"   Vocab size: {model.detection_head.vocab_size}")

        # Test model forward pass
        print(f"\nTesting model forward pass...")
        import torch

        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (1, 10)).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"✅ Forward pass successful")
            print(f"   Output logits shape: {outputs.logits.shape}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_unified_loading.py <model_path>")
        print("\nExamples:")
        print("  # Test with base model")
        print("  python test_unified_loading.py Qwen/Qwen2.5-VL-3B-Instruct")
        print("  # Test with checkpoint")
        print("  python test_unified_loading.py /path/to/checkpoint-100")
        return

    model_path = sys.argv[1]

    print("🧪 UNIFIED LOADING MECHANISM TEST")
    print("=" * 60)
    print(f"Testing with: {model_path}")
    print()

    # Test 1: Checkpoint inspection
    test_checkpoint_inspection(model_path)

    # Test 2: Unified loading
    test_unified_loading(model_path)

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
