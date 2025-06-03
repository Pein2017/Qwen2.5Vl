#!/usr/bin/env python3
"""
Test script to verify training-aligned inference settings.
This script tests a single sample to ensure the inference exactly matches training.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from eval.infer_dataset import TrainingAlignedInferenceEngine


def test_training_alignment():
    """Test that inference settings exactly match training."""

    # Configuration
    model_path = "output/qwen3B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"
    base_model_path = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    validation_file = "521_qwen_val.jsonl"

    print("🧪 Testing training-aligned inference...")

    # Load a single sample
    with open(validation_file, "r") as f:
        sample = json.loads(f.readline())

    # Extract test data
    conversations = sample.get("conversations", [])
    image_path = sample.get("image", "")

    if not conversations or not image_path:
        print("❌ Invalid sample data")
        return

    # Extract prompts
    system_prompt = ""
    user_prompt = ""
    ground_truth = ""

    for conv in conversations:
        if conv.get("role") == "system":
            system_prompt = conv.get("content", "")
        elif conv.get("role") == "user":
            user_prompt = conv.get("content", "")
        elif conv.get("role") == "assistant":
            ground_truth = conv.get("content", "")

    print(f"📁 Image: {image_path}")
    print(f"📝 User prompt length: {len(user_prompt)} chars")

    # Initialize training-aligned inference engine
    try:
        engine = TrainingAlignedInferenceEngine(
            model_path=model_path,
            base_model_path=base_model_path,
            device="auto",
            max_new_tokens=1024,  # Reduced for testing
        )
        print("✅ Model loaded with training alignment")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Verify training alignment
    print("\n🔍 Verifying training alignment:")
    print(f"   - Flash attention: {engine.model.config.attn_implementation}")
    print(f"   - Torch dtype: {engine.model.dtype}")
    print(f"   - Use cache: {engine.model.config.use_cache}")
    print(f"   - Model max length: {engine.tokenizer.model_max_length}")
    print(f"   - Max pixels: {engine.image_processor.max_pixels}")
    print(f"   - Min pixels: {engine.image_processor.min_pixels}")
    print(f"   - Merge size: {engine.image_processor.merge_size}")

    # Expected training values
    expected_values = {
        "attn_implementation": "flash_attention_2",
        "dtype": "torch.bfloat16",
        "use_cache": False,
        "model_max_length": 2048,
        "max_pixels": 1003520,
        "min_pixels": 784,
    }

    # Check alignment
    alignment_ok = True

    if (
        engine.model.config.attn_implementation
        != expected_values["attn_implementation"]
    ):
        print(
            f"❌ Attention implementation mismatch: {engine.model.config.attn_implementation} != {expected_values['attn_implementation']}"
        )
        alignment_ok = False

    if str(engine.model.dtype) != expected_values["dtype"]:
        print(f"❌ Dtype mismatch: {engine.model.dtype} != {expected_values['dtype']}")
        alignment_ok = False

    if engine.model.config.use_cache != expected_values["use_cache"]:
        print(
            f"❌ Use cache mismatch: {engine.model.config.use_cache} != {expected_values['use_cache']}"
        )
        alignment_ok = False

    if engine.tokenizer.model_max_length != expected_values["model_max_length"]:
        print(
            f"❌ Model max length mismatch: {engine.tokenizer.model_max_length} != {expected_values['model_max_length']}"
        )
        alignment_ok = False

    if engine.image_processor.max_pixels != expected_values["max_pixels"]:
        print(
            f"❌ Max pixels mismatch: {engine.image_processor.max_pixels} != {expected_values['max_pixels']}"
        )
        alignment_ok = False

    if engine.image_processor.min_pixels != expected_values["min_pixels"]:
        print(
            f"❌ Min pixels mismatch: {engine.image_processor.min_pixels} != {expected_values['min_pixels']}"
        )
        alignment_ok = False

    if alignment_ok:
        print("✅ All settings match training configuration!")
    else:
        print("❌ Some settings don't match training configuration!")
        return False

    # Generate response
    print("\n🔄 Generating response with training-aligned settings...")
    try:
        response, metadata = engine.generate_response(
            image_path, system_prompt, user_prompt
        )

        print("✅ Generation completed!")
        print(f"📊 Generated tokens: {metadata.get('generated_tokens', 0)}")
        print(f"📏 Response length: {len(response)} chars")

        # Verify generation config
        gen_config = metadata.get("generation_config", {})
        print("\n🎯 Generation config verification:")
        print(f"   - do_sample: {gen_config.get('do_sample')} (should be False)")
        print(f"   - temperature: {gen_config.get('temperature')} (should be None)")
        print(f"   - use_cache: {gen_config.get('use_cache')} (should be False)")
        print(f"   - num_beams: {gen_config.get('num_beams')} (should be 1)")

        # Check for problematic patterns
        if "addCriterion" in response:
            addcriterion_count = response.count("addCriterion")
            print(f"⚠️  Found {addcriterion_count} 'addCriterion' repetitions")
            if addcriterion_count > 5:
                print("❌ Still has repetitive generation issue!")
                return False
        else:
            print("✅ No 'addCriterion' repetitions found")

        # Check if response looks like valid JSON
        if response.strip().startswith("[") and (
            response.strip().endswith("]") or response.strip().endswith("}")
        ):
            print("✅ Response has valid JSON structure")
        else:
            print("⚠️  Response may not be valid JSON")

        # Show first 200 chars of response
        print("\n📄 Response preview:")
        print(f"   {response[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_training_alignment()
    if success:
        print("\n🎉 Training alignment test passed! Settings match training exactly.")
    else:
        print("\n💥 Training alignment test failed! Check configuration.")
