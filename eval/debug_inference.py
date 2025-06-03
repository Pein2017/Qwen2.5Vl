#!/usr/bin/env python3
"""
Debug script to identify where inference gets stuck.
Tests each step of the inference pipeline with detailed logging.
"""

import os
import sys
import time
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.core import Config, ModelWrapper
from src.rope2d import get_rope_index_25
from src.utils import (
    UnifiedLogger,
    extract_prompts_from_conversation,
    find_ground_truth_response,
    load_jsonl,
)


def debug_inference_step_by_step():
    """Debug inference step by step to find where it gets stuck."""

    print("🔍 Starting step-by-step inference debugging...")
    print(f"⏰ Start time: {datetime.now()}")

    # Configuration
    model_path = "output/qwen7B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"
    validation_file = "521_qwen_val.jsonl"

    # Step 1: Load validation data
    print("\n📊 Step 1: Loading validation data...")
    start_time = time.time()
    try:
        samples = load_jsonl(validation_file)
        sample = samples[0]  # Use first sample
        print(f"✅ Loaded {len(samples)} samples in {time.time() - start_time:.2f}s")
        print(f"📄 Using sample: {sample.get('image', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load validation data: {e}")
        return False

    # Step 2: Extract conversation data
    print("\n💬 Step 2: Extracting conversation data...")
    start_time = time.time()
    try:
        conversations = sample.get("conversations", [])
        image_path = sample.get("image", "")

        system_prompt, user_prompt = extract_prompts_from_conversation(conversations)
        ground_truth = find_ground_truth_response(conversations)

        print(f"✅ Extracted conversation data in {time.time() - start_time:.2f}s")
        print(f"📝 System prompt: {len(system_prompt)} chars")
        print(f"📝 User prompt: {len(user_prompt)} chars")
        print(f"📝 Ground truth: {len(ground_truth)} chars")
        print(f"🖼️  Image path: {image_path}")
    except Exception as e:
        print(f"❌ Failed to extract conversation data: {e}")
        return False

    # Step 3: Create config and logger
    print("\n⚙️  Step 3: Creating config and logger...")
    start_time = time.time()
    try:
        config = Config()
        config.model_path = model_path
        config.cache_dir = "/data4/swift/model_cache"
        config.model_max_length = 8192
        config.max_pixels = 1003520
        config.min_pixels = 784

        logger = UnifiedLogger(
            log_dir="logs",
            verbose=True,
            log_level=20,
            is_training=False,
        )

        print(f"✅ Created config and logger in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to create config/logger: {e}")
        return False

    # Step 4: Load model components
    print("\n🤖 Step 4: Loading model components...")
    start_time = time.time()
    try:
        model_wrapper = ModelWrapper(config, logger)
        print(f"📦 ModelWrapper created in {time.time() - start_time:.2f}s")

        # Load all components
        load_start = time.time()
        model, tokenizer, image_processor = model_wrapper.load_all()
        print(f"✅ Model components loaded in {time.time() - load_start:.2f}s")

        # Set critical settings
        model.config.use_cache = False
        model.eval()

        # Move to GPU
        if torch.cuda.is_available():
            device_start = time.time()
            model = model.to("cuda:0")
            print(f"🚀 Moved to GPU in {time.time() - device_start:.2f}s")

        print(f"📊 Total model loading time: {time.time() - start_time:.2f}s")
        print(
            f"🔧 Flash attention: {getattr(model.config, 'attn_implementation', 'unknown')}"
        )
        print(f"🔧 Torch dtype: {model.dtype}")
        print(f"🔧 Use cache: {model.config.use_cache}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 5: Load and preprocess image
    print("\n🖼️  Step 5: Loading and preprocessing image...")
    start_time = time.time()
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"📷 Image loaded: {image.size} in {time.time() - start_time:.2f}s")

        # Preprocess image
        preprocess_start = time.time()
        visual_processed = image_processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]

        grid_thw = visual_processed["image_grid_thw"][0]
        grid_thw_merged = grid_thw.prod() // image_processor.merge_size**2

        print(f"✅ Image preprocessed in {time.time() - preprocess_start:.2f}s")
        print(f"📐 Grid THW: {grid_thw.tolist()}")
        print(f"📐 Grid THW merged: {grid_thw_merged.item()}")
        print(f"📐 Image tensor shape: {image_tensor.shape}")

    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        return False

    # Step 6: Process text and create inputs
    print("\n📝 Step 6: Processing text and creating model inputs...")
    start_time = time.time()
    try:
        # Replace vision tokens
        if "<image>" in user_prompt:
            vision_tokens = (
                "<|vision_start|>"
                + "<|image_pad|>" * grid_thw_merged.item()
                + "<|vision_end|>"
            )
            user_prompt_processed = user_prompt.replace("<image>", vision_tokens)
            print(f"🔄 Replaced <image> with {grid_thw_merged.item()} vision tokens")
        else:
            user_prompt_processed = user_prompt
            print("ℹ️  No <image> token found in user prompt")

        # Create conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_processed},
        ]

        # Apply chat template
        template_start = time.time()
        input_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        print(f"📝 Chat template applied in {time.time() - template_start:.2f}s")
        print(f"📏 Input IDs shape: {input_ids.shape}")

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Calculate position IDs
        pos_start = time.time()
        position_ids, _ = get_rope_index_25(
            spatial_merge_size=image_processor.merge_size,
            input_ids=input_ids,
            image_grid_thw=grid_thw.unsqueeze(0),
        )
        print(f"📐 Position IDs calculated in {time.time() - pos_start:.2f}s")
        print(f"📏 Position IDs shape: {position_ids.shape}")

        print(f"✅ Text processing completed in {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"❌ Failed to process text: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 7: Prepare model inputs
    print("\n🎯 Step 7: Preparing model inputs...")
    start_time = time.time()
    try:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_tensor.unsqueeze(0),
            "image_grid_thw": grid_thw.unsqueeze(0),
            "position_ids": position_ids,
        }

        # Move to device
        device_start = time.time()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"🚀 Moved inputs to device in {time.time() - device_start:.2f}s")

        print(f"✅ Model inputs prepared in {time.time() - start_time:.2f}s")
        print("📊 Input shapes:")
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(f"   {k}: {v.shape}")
            else:
                print(f"   {k}: {type(v)}")

    except Exception as e:
        print(f"❌ Failed to prepare model inputs: {e}")
        return False

    # Step 8: Test model forward pass (without generation)
    print("\n🧪 Step 8: Testing model forward pass...")
    start_time = time.time()
    try:
        with torch.no_grad():
            # Test forward pass without generation
            forward_start = time.time()
            outputs = model(**inputs)
            forward_time = time.time() - forward_start

            print(f"✅ Forward pass completed in {forward_time:.2f}s")
            print(f"📊 Logits shape: {outputs.logits.shape}")
            print(
                f"💾 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"💾 GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 9: Test generation (the critical step)
    print("\n🎲 Step 9: Testing generation (CRITICAL STEP)...")
    print("⚠️  This is where the model might get stuck...")
    start_time = time.time()

    try:
        # Set a timeout for generation
        max_new_tokens = 512  # Reduced for testing
        print("🔧 Generation settings:")
        print(f"   max_new_tokens: {max_new_tokens}")
        print("   do_sample: False")
        print("   use_cache: False")
        print("   num_beams: 1")

        with torch.no_grad():
            print(f"⏰ Starting generation at {datetime.now()}")
            generation_start = time.time()

            # Add progress tracking
            def progress_callback(step, total_steps):
                if step % 10 == 0:
                    elapsed = time.time() - generation_start
                    print(f"   🔄 Step {step}/{total_steps}, elapsed: {elapsed:.1f}s")

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=None,
                length_penalty=None,
                early_stopping=False,
                min_new_tokens=1,
                no_repeat_ngram_size=None,
            )

            generation_time = time.time() - generation_start
            print(f"✅ Generation completed in {generation_time:.2f}s")
            print(f"📏 Output shape: {output_ids.shape}")

            # Decode output
            decode_start = time.time()
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]

            output_text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            decode_time = time.time() - decode_start
            print(f"✅ Decoding completed in {decode_time:.2f}s")
            print(f"📏 Output length: {len(output_text)} chars")
            print(f"📄 Output preview: {output_text[:200]}...")

            # Check for problematic patterns
            if "addCriterion" in output_text:
                count = output_text.count("addCriterion")
                print(f"⚠️  Found {count} 'addCriterion' repetitions")
            else:
                print("✅ No repetitive patterns detected")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Final summary
    total_time = time.time() - start_time
    print("\n🎉 All steps completed successfully!")
    print(f"⏰ Total time: {total_time:.2f}s")
    print(f"⏰ End time: {datetime.now()}")

    return True


if __name__ == "__main__":
    print("🔍 INFERENCE DEBUGGING TOOL")
    print("=" * 60)

    # Check environment
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🚀 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("=" * 60)

    success = debug_inference_step_by_step()

    if success:
        print("\n✅ DEBUG PASSED: Inference pipeline works correctly")
    else:
        print("\n❌ DEBUG FAILED: Found issues in inference pipeline")
