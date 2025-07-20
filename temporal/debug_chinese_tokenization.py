#!/usr/bin/env python3
"""
Debug script to identify the Chinese character truncation issue.
"""

import sys


sys.path.append("/data3/Qwen2.5-VL-main")


def test_chinese_tokenization():
    """Test tokenization of the specific problematic Chinese text."""

    # Import the model loader to get the actual tokenizer
    from src.config.global_config import init_config
    from src.models.model_loader import load_model_and_processor_unified

    # Initialize with the correct config
    init_config("/data3/Qwen2.5-VL-main/configs/base_flat_det.yaml")
    from src.config import config

    try:
        # Load the actual tokenizer used in training
        model, tokenizer, image_processor = load_model_and_processor_unified(
            config.base_model_path, for_inference=False, force_detection=True
        )

        print("✅ Tokenizer loaded successfully")
        print(f"Tokenizer type: {type(tokenizer)}")
        print(f"Vocab size: {len(tokenizer.get_vocab())}")

        # Test the problematic texts
        test_cases = [
            "明白！我会仔细学习参考示例中的检测模式、标注风格和判断标准，然后应用到目标图像的检测中。",
            "机柜空间/满载",
            "BBU基带处理单元/中兴",
        ]

        for i, text in enumerate(test_cases):
            print(f"\n=== Test Case {i + 1} ===")
            print(f"Original: '{text}'")

            # Encode with labels (like in training)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            print(f"Token IDs: {tokens}")

            # Decode normally
            decoded = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"Decoded: '{decoded}'")
            print(f"Match: {text == decoded}")

            # Test individual character tokenization
            if len(text) > 0:
                first_char = text[0]
                first_char_tokens = tokenizer.encode(
                    first_char, add_special_tokens=False
                )
                first_char_decoded = tokenizer.decode(
                    first_char_tokens, skip_special_tokens=False
                )
                print(
                    f"First char '{first_char}' -> {first_char_tokens} -> '{first_char_decoded}'"
                )

            # Test partial decoding (simulate label filtering)
            if len(tokens) > 1:
                # Skip first token (common in training when first token is ignored)
                partial_tokens = tokens[1:]
                partial_decoded = tokenizer.decode(
                    partial_tokens, skip_special_tokens=False
                )
                print(f"Without first token: {partial_tokens} -> '{partial_decoded}'")

            print()

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        import traceback

        traceback.print_exc()


def simulate_training_label_extraction():
    """Simulate the exact label extraction process from training."""

    from src.config.global_config import init_config
    from src.models.model_loader import load_model_and_processor_unified

    # Initialize with the correct config
    init_config("/data3/Qwen2.5-VL-main/configs/base_flat_det.yaml")
    from src.config import config

    try:
        model, tokenizer, image_processor = load_model_and_processor_unified(
            config.base_model_path, for_inference=False, force_detection=True
        )

        print("\n=== Simulating Training Label Extraction ===")

        # Create a sample like in training
        text = "明白！我会仔细学习参考示例中的检测模式、标注风格和判断标准，然后应用到目标图像的检测中。"

        # Encode the text
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"Input IDs: {input_ids}")

        # Simulate label creation (common pattern: ignore first few tokens, keep rest)
        # This is how training typically works - some tokens are IGNORE_INDEX (-100)
        labels = [-100, -100] + input_ids[2:]  # Ignore first 2 tokens

        print(f"Labels: {labels}")

        # Extract target IDs (what the trainer does)
        target_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
        print(f"Target IDs: {target_ids}")

        # Decode target IDs
        decoded_target = tokenizer.decode(target_ids, skip_special_tokens=False)
        print(f"Decoded target: '{decoded_target}'")

        # This should show the truncation issue!

    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_chinese_tokenization()
    simulate_training_label_extraction()
