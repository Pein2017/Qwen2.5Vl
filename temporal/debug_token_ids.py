#!/usr/bin/env python3
"""
Debug script to check token IDs and geometry span detection issue.
"""

import sys
from pathlib import Path

import torch


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.chat_processor import ChatProcessor
from src.config.global_config import init_config


def debug_token_ids():
    """Debug token IDs and coordinate manager setup."""
    print("🔍 Debugging token IDs and coordinate manager...")

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "bbu_v2.yaml"
    config = init_config(str(config_path))

    # Create chat processor
    chat_processor = ChatProcessor(config)

    # Check tokenizer vocabulary
    vocab = chat_processor.tokenizer.get_vocab()
    print(f"\n📊 Tokenizer vocabulary size: {len(vocab)}")

    # Check for geometry tokens
    geometry_tokens = {
        "box_start": "<|box_start|>",
        "box_end": "<|box_end|>",
        "square_start": "<|square_start|>",
        "square_end": "<|square_end|>",
        "line_start": "<|line_start|>",
        "line_end": "<|line_end|>",
    }

    print(f"\n🎯 Geometry token IDs:")
    for name, token in geometry_tokens.items():
        token_id = vocab.get(token)
        print(f"   {name}: {token} -> {token_id}")

    # Check coordinate manager configuration
    if (
        hasattr(chat_processor, "coordinate_manager")
        and chat_processor.coordinate_manager
    ):
        coord_manager = chat_processor.coordinate_manager
        print(f"\n🔧 Coordinate Manager Configuration:")
        print(f"   box_start_id: {coord_manager.config.box_start_id}")
        print(f"   box_end_id: {coord_manager.config.box_end_id}")
        print(
            f"   coord_start_id: {getattr(coord_manager, 'coord_start_id', 'NOT_SET')}"
        )
        print(f"   coord_end_id: {getattr(coord_manager, 'coord_end_id', 'NOT_SET')}")
        print(f"   max_coord_value: {coord_manager.config.max_coord_value}")

        # Check coordinate token range
        if hasattr(coord_manager, "coord_start_id") and hasattr(
            coord_manager, "coord_end_id"
        ):
            print(
                f"   coordinate range: [{coord_manager.coord_start_id}, {coord_manager.coord_end_id})"
            )

            # Test coordinate token detection
            test_coord_ids = [2113, 2114, 2115, 2116]  # From the error log
            print(f"\n🧪 Testing coordinate token detection:")
            for coord_id in test_coord_ids:
                is_coord = coord_manager.is_coordinate_token(coord_id)
                print(f"   Token {coord_id}: is_coordinate = {is_coord}")

        # Test geometry span detection with a simple sequence
        print(f"\n🧪 Testing geometry span detection:")

        # Create a test sequence similar to the error case
        # Box end at position 3635, coordinate tokens at various positions
        test_sequence = torch.zeros(3660, dtype=torch.long)

        # Add box_end at position 3635 (from error log)
        if coord_manager.config.box_end_id is not None:
            test_sequence[3635] = coord_manager.config.box_end_id
            print(
                f"   Added box_end ({coord_manager.config.box_end_id}) at position 3635"
            )

        # Add some coordinate tokens (from error log positions)
        coord_positions = [2113, 2114, 2115, 2116]
        if hasattr(coord_manager, "coord_start_id"):
            for i, pos in enumerate(coord_positions):
                test_sequence[pos] = coord_manager.coord_start_id + i
                print(
                    f"   Added coordinate token {coord_manager.coord_start_id + i} at position {pos}"
                )

        # Test span detection
        test_batch = test_sequence.unsqueeze(0)  # Add batch dimension
        geometry_spans = coord_manager.detect_geometry_spans(test_batch)
        print(f"   Detected spans: {geometry_spans}")

        # Check what happens if we add box_start
        if coord_manager.config.box_start_id is not None:
            print(f"\n🔧 Testing with box_start added:")
            test_sequence_with_start = test_sequence.clone()
            test_sequence_with_start[2110] = (
                coord_manager.config.box_start_id
            )  # Before coordinates
            test_batch_with_start = test_sequence_with_start.unsqueeze(0)
            geometry_spans_with_start = coord_manager.detect_geometry_spans(
                test_batch_with_start
            )
            print(
                f"   Added box_start ({coord_manager.config.box_start_id}) at position 2110"
            )
            print(f"   Detected spans: {geometry_spans_with_start}")
    else:
        print(f"\n❌ No coordinate manager found!")

    return True


if __name__ == "__main__":
    try:
        success = debug_token_ids()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
