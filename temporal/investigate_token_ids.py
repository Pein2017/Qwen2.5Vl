#!/usr/bin/env python3
"""
Investigation script for token ID mismatch issue.

This script analyzes the tokenizer vocabulary structure to understand
why coordinate tokens are getting unexpected IDs.
"""

from transformers import AutoTokenizer


def investigate_tokenizer_vocab():
    """Investigate tokenizer vocabulary structure."""
    print("🔍 Investigating tokenizer vocabulary structure...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )

    print(f"✅ Tokenizer loaded: {type(tokenizer)}")
    print(f"   Original vocab size: {len(tokenizer.get_vocab())}")

    # Get vocab dictionary
    vocab = tokenizer.get_vocab()

    # Find box tokens
    box_start_candidates = [k for k in vocab.keys() if "box_start" in k.lower()]
    box_end_candidates = [k for k in vocab.keys() if "box_end" in k.lower()]

    print(f"\n🔍 Box token candidates:")
    print(f"   box_start candidates: {box_start_candidates}")
    print(f"   box_end candidates: {box_end_candidates}")

    # Check specific tokens
    specific_tokens = ["<|box_start|>", "<|box_end|>", "<|im_start|>", "<|im_end|>"]
    print(f"\n🔍 Specific token IDs:")
    for token in specific_tokens:
        if token in vocab:
            print(f"   {token}: {vocab[token]}")
        else:
            print(f"   {token}: NOT FOUND")

    # Check highest token IDs
    max_id = max(vocab.values())
    min_id = min(vocab.values())
    print(f"\n🔍 Token ID range:")
    print(f"   Min ID: {min_id}")
    print(f"   Max ID: {max_id}")

    # Find tokens with highest IDs
    high_id_tokens = {k: v for k, v in vocab.items() if v >= max_id - 100}
    print(f"\n🔍 Tokens with highest IDs (top 100):")
    for token, id in sorted(high_id_tokens.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]:
        print(f"   {token}: {id}")

    return tokenizer, vocab


def test_coordinate_token_addition():
    """Test adding coordinate tokens to tokenizer."""
    print("\n🧪 Testing coordinate token addition...")

    tokenizer, original_vocab = investigate_tokenizer_vocab()
    original_size = len(original_vocab)

    # Add coordinate tokens
    coordinate_tokens = [f"<coord_{i}>" for i in range(10)]  # Test with 10 tokens

    print(f"\n🔧 Adding coordinate tokens: {coordinate_tokens}")

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": coordinate_tokens}
    )

    print(f"   Reported added tokens: {num_added}")

    # Check new vocab size
    new_vocab = tokenizer.get_vocab()
    new_size = len(new_vocab)
    print(f"   New vocab size: {new_size}")
    print(f"   Size increase: {new_size - original_size}")

    # Check actual IDs assigned
    print(f"\n🔍 Actual token IDs assigned:")
    for token in coordinate_tokens:
        if token in new_vocab:
            print(f"   {token}: {new_vocab[token]}")
        else:
            print(f"   {token}: NOT FOUND")

    # Check if tokens are sequential
    coord_ids = [new_vocab[token] for token in coordinate_tokens if token in new_vocab]
    if coord_ids:
        print(f"\n🔍 Token ID sequence analysis:")
        print(f"   First coord token ID: {min(coord_ids)}")
        print(f"   Last coord token ID: {max(coord_ids)}")
        print(
            f"   Expected sequential: {list(range(min(coord_ids), max(coord_ids) + 1))}"
        )
        print(f"   Actual IDs: {sorted(coord_ids)}")
        print(
            f"   Sequential: {sorted(coord_ids) == list(range(min(coord_ids), max(coord_ids) + 1))}"
        )

    return tokenizer, new_vocab


def test_tokenizer_consistency():
    """Test tokenizer consistency with coordinate tokens."""
    print("\n🧪 Testing tokenizer consistency...")

    tokenizer, vocab = test_coordinate_token_addition()

    # Test tokenization and conversion
    test_text = "test: <|box_start|><coord_0><coord_1><coord_2><coord_3><|box_end|>"

    print(f"\n🔍 Testing tokenization:")
    print(f"   Input text: {test_text}")

    # Tokenize
    tokens = tokenizer(test_text, return_tensors="pt")
    token_ids = tokens["input_ids"][0]

    print(f"   Token IDs: {token_ids.tolist()}")

    # Convert back to tokens
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"   Token strings: {token_strings}")

    # Check if coordinate tokens are properly identified
    coord_token_ids = []
    for i, token_str in enumerate(token_strings):
        if token_str.startswith("<coord_"):
            coord_token_ids.append((i, token_ids[i].item(), token_str))

    print(f"\n🔍 Coordinate tokens found:")
    for pos, id, token in coord_token_ids:
        print(f"   Position {pos}: {token} -> ID {id}")

    return tokenizer, token_ids


def analyze_coordinate_manager_logic():
    """Analyze the coordinate token manager logic."""
    print("\n🔍 Analyzing coordinate token manager logic...")

    from src.utils.coordinate_token_manager import (
        CoordinateTokenConfig,
        CoordinateTokenManager,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )

    original_vocab_size = len(tokenizer.get_vocab())
    print(f"   Original vocab size: {original_vocab_size}")

    # Create coordinate config
    config = CoordinateTokenConfig(
        enable_coordinate_tokens=True,
        max_coord_value=10,  # Small number for testing
    )

    # Create manager
    manager = CoordinateTokenManager(
        tokenizer=tokenizer, config=config, original_vocab_size=original_vocab_size
    )

    print(f"   Manager original_vocab_size: {manager.original_vocab_size}")
    print(f"   Manager extended_vocab_size: {manager.extended_vocab_size}")
    print(f"   Manager coord_start_id: {manager.coord_start_id}")
    print(f"   Manager coord_end_id: {manager.coord_end_id}")

    # Check actual tokenizer vocab after manager initialization
    actual_vocab = tokenizer.get_vocab()
    actual_size = len(actual_vocab)
    print(f"   Actual vocab size after manager init: {actual_size}")

    # Check coordinate token IDs
    print(f"\n🔍 Checking coordinate token IDs:")
    for i in range(config.max_coord_value):
        token = f"<coord_{i}>"
        if token in actual_vocab:
            actual_id = actual_vocab[token]
            expected_id = manager.coord_start_id + i
            print(
                f"   {token}: actual={actual_id}, expected={expected_id}, match={actual_id == expected_id}"
            )
        else:
            print(f"   {token}: NOT FOUND in vocab")

    return manager


if __name__ == "__main__":
    print("🚀 Investigating Token ID Mismatch Issue\n")

    try:
        # Step 1: Investigate basic tokenizer structure
        investigate_tokenizer_vocab()

        # Step 2: Test coordinate token addition
        test_coordinate_token_addition()

        # Step 3: Test tokenizer consistency
        test_tokenizer_consistency()

        # Step 4: Analyze coordinate manager logic
        analyze_coordinate_manager_logic()

        print("\n🎉 Investigation completed!")

    except Exception as e:
        print(f"\n❌ Investigation failed with error: {e}")
        import traceback

        traceback.print_exc()
