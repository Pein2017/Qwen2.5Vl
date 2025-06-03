#!/usr/bin/env python3
"""
Minimal test script with eager attention and reduced parameters.
This helps isolate whether the issue is with flash attention or generation parameters.
"""

import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chat template from training
from src.utils import CHAT_TEMPLATE


def minimal_inference_test():
    """Minimal inference test with safe parameters."""

    print("🧪 MINIMAL INFERENCE TEST")
    print("=" * 50)
    print(f"⏰ Start time: {datetime.now()}")

    try:
        # Import required modules
        import json

        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Configuration - use safe settings
        model_path = "output/qwen7B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"
        base_model_path = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-7B-Instruct"
        validation_file = "521_qwen_val.jsonl"

        print(f"📁 Model path: {model_path}")

        # Load sample data
        print("\n📊 Loading sample data...")
        with open(validation_file, "r") as f:
            sample = json.loads(f.readline())

        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])

        # Extract prompts
        user_prompt = ""
        for conv in conversations:
            if conv.get("role") == "user":
                user_prompt = conv.get("content", "")
                break

        print(f"🖼️  Image: {image_path}")
        print(f"📝 Prompt length: {len(user_prompt)} chars")

        # Load model with EAGER attention (safer)
        print("\n🤖 Loading model with eager attention...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Use eager instead of flash_attention_2
            device_map="auto",
        )

        # Load processor from base model (has chat template)
        print("📦 Loading processor from base model...")
        processor = AutoProcessor.from_pretrained(base_model_path)

        # Set chat template manually
        processor.chat_template = CHAT_TEMPLATE
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.chat_template = CHAT_TEMPLATE

        print("✅ Chat template set successfully")

        # Load image
        print("🖼️  Loading image...")
        image = Image.open(image_path).convert("RGB")

        # Create simple conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        print("📝 Applying chat template...")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"📝 Template applied, text length: {len(text)} chars")

        # Process inputs
        print("⚙️  Processing inputs...")
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )

        # Move to GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            print("🚀 Moved inputs to GPU")

        print(f"📊 Input shapes: {inputs.input_ids.shape}")

        # Test forward pass first
        print("\n🧪 Testing forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"✅ Forward pass successful, logits shape: {outputs.logits.shape}")

        # Test generation with MINIMAL parameters
        print("\n🎲 Testing generation with minimal parameters...")
        print("⚠️  Using very conservative settings...")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,  # Very small for testing
                do_sample=False,  # Deterministic
                num_beams=1,  # No beam search
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,  # Try with cache first
            )

        # Decode output
        print("📄 Decoding output...")
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        print("✅ Generation successful!")
        print(f"📏 Output length: {len(output_text)} chars")
        print(f"📄 Output: {output_text[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_without_cache():
    """Test with use_cache=False like training."""

    print("\n🧪 TESTING WITHOUT CACHE (like training)...")

    try:
        # Same setup as above but with use_cache=False
        import json

        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = "output/qwen7B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"
        base_model_path = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-7B-Instruct"
        validation_file = "521_qwen_val.jsonl"

        # Load sample
        with open(validation_file, "r") as f:
            sample = json.loads(f.readline())

        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])

        user_prompt = ""
        for conv in conversations:
            if conv.get("role") == "user":
                user_prompt = conv.get("content", "")
                break

        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        )

        # Set use_cache=False like training
        model.config.use_cache = False

        processor = AutoProcessor.from_pretrained(base_model_path)
        processor.chat_template = CHAT_TEMPLATE
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.chat_template = CHAT_TEMPLATE

        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("🎲 Testing generation with use_cache=False...")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                use_cache=False,  # Same as training
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        print("✅ Generation with use_cache=False successful!")
        print(f"📄 Output: {output_text[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Test with use_cache=False failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 MINIMAL INFERENCE TESTING")
    print("=" * 60)

    # Test 1: Basic inference with cache
    success1 = minimal_inference_test()

    # Test 2: Inference without cache (like training)
    success2 = test_without_cache()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Basic inference (with cache): {'PASSED' if success1 else 'FAILED'}")
    print(
        f"✅ Training-like inference (no cache): {'PASSED' if success2 else 'FAILED'}"
    )

    if success1 and success2:
        print("\n🎉 Both tests passed! The model works correctly.")
        print("💡 The issue might be with flash attention or longer generation.")
    elif success1 and not success2:
        print("\n⚠️  Cache-enabled works, but training-like fails.")
        print("💡 The issue is likely with use_cache=False setting.")
    elif not success1 and not success2:
        print("\n❌ Both tests failed.")
        print("💡 There's a fundamental issue with the model or environment.")
    else:
        print("\n🤔 Unexpected result pattern.")

    print("=" * 60)
