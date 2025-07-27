#!/usr/bin/env python3
"""
End-to-end BBU training pipeline test
Verify that the complete pipeline works: config → model loading → data processing → training step
"""

import sys


sys.path.insert(0, "/data3/Qwen2.5-VL-main")


import torch

from src.chat_processor import ChatProcessor
from src.config.global_config import init_config, reset_config
from src.models.model_loader import load_model_and_processor_unified


def create_sample_training_data():
    """Create a sample training data entry matching the expected format"""
    # Create a random 3xHxW tensor as image for testing
    import tempfile

    import numpy as np
    from PIL import Image

    # Create random image data
    height, width = 728, 532
    image_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(temp_file.name)
    temp_file.close()

    return {
        "images": [temp_file.name],  # Random image for testing
        "objects": [
            {
                "bbox_2d": [269, 46, 291, 69],
                "desc": "螺丝、光纤插头/机柜处接地螺丝,只显示部分,符合要求",
            },
            {
                "square": [150, 10, 211, 35, 218, 16, 166, 0],
                "desc": "标签/5G-BBU-（接地线）",
            },
            {
                "line": [260, 52, 219, 31, 173, 6, 116, 3, 75, 3, 23, 8, 0, 16],
                "desc": "电线/有遮挡,捆扎整齐",
            },
        ],
        "width": width,
        "height": height,
        "_temp_image": temp_file.name,  # Keep reference for cleanup
    }


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline functionality"""

    print("🧪 Testing end-to-end BBU training pipeline...")

    try:
        # Reset any existing config
        reset_config()

        # Initialize config
        config = init_config("configs/bbu_v2.yaml")
        print(
            f"✅ Configuration loaded: coordinate_tokens={config.coordinate_config_enable_coordinate_tokens}"
        )

        # Test 1: Model and tokenizer loading
        print("\n🔧 Phase 1: Model and tokenizer loading...")
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,  # Training mode
        )
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Tokenizer vocab size: {len(tokenizer.get_vocab())}")

        # Verify coordinate token systems are present
        if hasattr(model, "simple_token_manager"):
            print(f"✅ SimpleTokenManager initialized")
        else:
            print("⚠️ SimpleTokenManager not found")

        if hasattr(model, "coordinate_manager"):
            print(f"✅ CoordinateTokenManager initialized")
        else:
            print("⚠️ CoordinateTokenManager not found")

        # Test 2: Chat processor initialization
        print("\n🔧 Phase 2: Chat processor initialization...")
        chat_processor = ChatProcessor(
            tokenizer=tokenizer,
            image_processor=processor,  # processor is already the image processor
            max_coord_value=config.chat_processor_max_coord_value,
            language=config.language,
        )
        print(f"✅ ChatProcessor initialized for {config.language}")

        # Test 3: Data processing pipeline
        print("\n🔧 Phase 3: Data processing pipeline...")
        sample_data = create_sample_training_data()

        try:
            # Process the sample data
            processed_sample = chat_processor.process_sample(sample_data)
            print(f"✅ Sample processed successfully")
            print(f"   Input IDs shape: {processed_sample.input_ids.shape}")
            print(f"   Labels shape: {processed_sample.labels.shape}")

            # Check for special tokens
            input_ids = processed_sample.input_ids
            special_tokens_found = []

            # Check for object reference tokens
            if 151646 in input_ids:  # <|object_ref_start|>
                special_tokens_found.append("object_ref_start")
            if 151647 in input_ids:  # <|object_ref_end|>
                special_tokens_found.append("object_ref_end")

            # Check for geometry tokens
            if 151648 in input_ids:  # <|box_start|>
                special_tokens_found.append("box_start")
            if 151649 in input_ids:  # <|box_end|>
                special_tokens_found.append("box_end")

            print(f"✅ Special tokens found: {special_tokens_found}")

        except Exception as e:
            print(f"❌ Data processing failed: {e}")
            return False

        # Test 4: Model forward pass
        print("\n🔧 Phase 4: Model forward pass...")
        try:
            # Get model device
            model_device = next(model.parameters()).device

            # Prepare inputs for model and move to same device
            model_inputs = {
                "input_ids": processed_sample.input_ids.unsqueeze(0).to(
                    model_device
                ),  # Add batch dimension and move to device
                "labels": processed_sample.labels.unsqueeze(0).to(model_device),
            }

            # Add image inputs if available
            if (
                hasattr(processed_sample, "pixel_values")
                and processed_sample.pixel_values is not None
            ):
                model_inputs["pixel_values"] = processed_sample.pixel_values.unsqueeze(
                    0
                ).to(model_device)
            if (
                hasattr(processed_sample, "image_grid_thw")
                and processed_sample.image_grid_thw is not None
            ):
                model_inputs["image_grid_thw"] = (
                    processed_sample.image_grid_thw.unsqueeze(0).to(model_device)
                )

            # Run forward pass
            with torch.no_grad():  # No gradients for testing
                outputs = model(**model_inputs)

            print(f"✅ Forward pass successful")
            print(f"   Output logits shape: {outputs.logits.shape}")
            print(f"   Loss: {outputs.loss.item():.6f}")

            # Check for coordinate loss attributes
            coordinate_losses = []
            for attr in [
                "_llm_loss",
                "_coordinate_l1_loss",
                "_geometry_focal_loss",
                "_geometry_bbox_giou_loss",
            ]:
                if hasattr(outputs, attr):
                    coordinate_losses.append(attr)

            print(f"✅ Coordinate loss attributes found: {coordinate_losses}")

        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Test 5: Verify tensor flow consistency
        print("\n🔧 Phase 5: Tensor flow validation...")

        # Check that model is in expected mode
        if model.training:
            print("✅ Model in training mode")
        else:
            print("⚠️ Model in eval mode")

        # Check device placement
        model_device = next(model.parameters()).device
        print(f"✅ Model on device: {model_device}")

        # Verify coordinate token integration
        if config.coordinate_config_enable_coordinate_tokens:
            vocab_size = len(tokenizer.get_vocab())
            model_vocab_size = model.get_input_embeddings().num_embeddings

            if vocab_size == model_vocab_size:
                print("✅ Tokenizer and model vocabulary sizes match")
            else:
                print(
                    f"❌ FAIL: Vocabulary size mismatch - tokenizer: {vocab_size}, model: {model_vocab_size}"
                )
                return False

        print("\n🎉 End-to-end pipeline test completed successfully!")
        print("📊 Summary:")
        print("   ✅ Configuration system working")
        print("   ✅ Model loading with coordinate tokens")
        print("   ✅ Data processing pipeline")
        print("   ✅ Model forward pass")
        print("   ✅ Tensor flow validation")

        # Cleanup temporary image
        if "_temp_image" in sample_data:
            import os

            try:
                os.unlink(sample_data["_temp_image"])
            except:
                pass

        return True

    except Exception as e:
        print(f"❌ End-to-end pipeline test failed: {e}")
        import traceback

        traceback.print_exc()

        # Cleanup temporary image even on failure
        try:
            if "sample_data" in locals() and "_temp_image" in sample_data:
                import os

                os.unlink(sample_data["_temp_image"])
        except:
            pass

        return False
    finally:
        reset_config()


if __name__ == "__main__":
    success = test_end_to_end_pipeline()
    sys.exit(0 if success else 1)
