#!/usr/bin/env python3
"""
Quick test script to verify fixed inference parameters.
Tests the model on a single sample to ensure no repetitive generation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from eval.infer_dataset import SimpleInferenceEngine


def test_single_sample():
    """Test inference on a single sample to verify fixes."""

    # Configuration
    model_path = "output/qwen3B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"
    base_model_path = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    validation_file = "521_qwen_val.jsonl"

    print("🧪 Testing fixed inference parameters...")

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

    # Initialize inference engine with reduced max_new_tokens for testing
    try:
        engine = SimpleInferenceEngine(
            model_path=model_path,
            base_model_path=base_model_path,
            device="auto",
            max_new_tokens=1024,  # Reduced for testing
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Generate response
    print("🔄 Generating response...")
    try:
        response, metadata = engine.generate_response(
            image_path, system_prompt, user_prompt
        )

        print("✅ Generation completed!")
        print(f"📊 Generated tokens: {metadata.get('generated_tokens', 0)}")
        print(f"📏 Response length: {len(response)} chars")

        # Check for repetitive patterns
        if "addCriterion" in response:
            addcriterion_count = response.count("addCriterion")
            print(f"⚠️  Found {addcriterion_count} 'addCriterion' repetitions")
            if addcriterion_count > 10:
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
        print(f"📄 Response preview: {response[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_single_sample()
    if success:
        print("\n🎉 Test passed! Fixed parameters appear to work correctly.")
    else:
        print("\n💥 Test failed! Need further parameter adjustments.")
