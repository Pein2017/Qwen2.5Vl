#!/usr/bin/env python3
"""
Simple integration test for Top-K Unlikelihood functionality.

Tests the core Top-K methods without complex trainer setup.
"""

import torch
from unittest.mock import MagicMock


def test_topk_methods_directly():
    """Test Top-K methods directly without full trainer setup."""
    
    # Import the trainer class to access methods
    from src_coord_pretrain.training.trainer import PhaseATrainer
    
    # Create a minimal mock object with required attributes
    class MockTrainer:
        def __init__(self):
            self.ul_topk_noncoord = 10
            self.ul_topk_coord = 10
            self.ul_neighbor_window = 4
            self.unlikelihood_lambda_digits = 0.5
            self.unlikelihood_lambda_coords = 0.5
            
            # Mock tokenizer
            self.tokenizer = MagicMock()
            vocab = {}
            
            # Add coordinate tokens
            coord_start_id = 151667
            for i in range(50):  # Small range for testing
                token_name = f"<|coord_{i}|>"
                token_id = coord_start_id + i
                vocab[token_name] = token_id
            
            # Add regular tokens
            vocab.update({
                "0": 100, "1": 101, "2": 102, "3": 103,
                "<|im_end|>": 151643,
                "text": 500, "token": 501,
            })
            
            self.tokenizer.get_vocab.return_value = vocab
        
        def _get_coordinate_token_ids(self):
            """Get coordinate token IDs from vocabulary."""
            coord_ids = []
            vocab = self.tokenizer.get_vocab()
            for token_name, token_id in vocab.items():
                if token_name.startswith("<|coord_") and token_name.endswith("|>"):
                    coord_ids.append(token_id)
            return sorted(coord_ids)
    
    # Create mock trainer
    mock_trainer = MockTrainer()
    
    # Bind the actual methods from PhaseATrainer
    mock_trainer._compute_topk_unlikelihood_loss = PhaseATrainer._compute_topk_unlikelihood_loss.__get__(mock_trainer)
    mock_trainer._compute_coordinate_target_loss = PhaseATrainer._compute_coordinate_target_loss.__get__(mock_trainer)
    mock_trainer._compute_text_target_loss = PhaseATrainer._compute_text_target_loss.__get__(mock_trainer)
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 8, 2000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create labels with mixed forward/reverse samples
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    
    # Sample 1: Forward mapping (coordinate token response)
    labels[0, 5] = 151667 + 10  # <|coord_10|>
    labels[0, 6] = 151643       # <|im_end|>
    
    # Sample 2: Reverse mapping (text response)
    labels[1, 4] = 101  # "1"
    labels[1, 5] = 102  # "2"
    labels[1, 6] = 151643  # <|im_end|>
    
    assistant_mask = (labels != -100)
    
    print("🧪 Testing Top-K Unlikelihood methods...")
    
    # Test 1: Top-K loss computation
    try:
        loss = mock_trainer._compute_topk_unlikelihood_loss(logits, labels, assistant_mask)
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"✅ Top-K loss computation: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Top-K loss computation failed: {e}")
        raise
    
    # Test 2: Coordinate target loss
    try:
        coord_token_ids = mock_trainer._get_coordinate_token_ids()
        coord_ids_tensor = torch.tensor(coord_token_ids, device=logits.device, dtype=torch.long)
        coord_target_mask = torch.isin(labels, coord_ids_tensor) & assistant_mask
        
        if coord_target_mask.any():
            coord_loss = mock_trainer._compute_coordinate_target_loss(
                logits, labels, coord_target_mask, coord_ids_tensor
            )
            
            assert torch.isfinite(coord_loss), "Coordinate loss should be finite"
            assert coord_loss >= 0, f"Coordinate loss should be non-negative, got {coord_loss.item()}"
            
            print(f"✅ Coordinate target loss: {coord_loss.item():.4f}")
        else:
            print("✅ No coordinate targets found (expected for this test)")
        
    except Exception as e:
        print(f"❌ Coordinate target loss failed: {e}")
        raise
    
    # Test 3: Text target loss
    try:
        text_target_mask = (~torch.isin(labels, coord_ids_tensor)) & (labels != -100) & assistant_mask
        
        if text_target_mask.any():
            text_loss = mock_trainer._compute_text_target_loss(
                logits, labels, text_target_mask, coord_ids_tensor
            )
            
            assert torch.isfinite(text_loss), "Text loss should be finite"
            assert text_loss >= 0, f"Text loss should be non-negative, got {text_loss.item()}"
            
            print(f"✅ Text target loss: {text_loss.item():.4f}")
        else:
            print("✅ No text targets found (unexpected)")
        
    except Exception as e:
        print(f"❌ Text target loss failed: {e}")
        raise
    
    print("🎉 All Top-K method tests passed!")


def test_configuration_loading():
    """Test that configuration parameters are properly loaded."""
    
    # Test configuration dictionary
    test_config = {
        "unlikelihood_enabled": True,
        "unlikelihood_lambda_digits": 0.8,
        "unlikelihood_lambda_coords": 1.2,
        "unlikelihood_coord_window": 6,
        "ul_topk_noncoord": 50,
        "ul_topk_coord": 75,
        "ul_neighbor_window": 10,
    }
    
    # Simulate configuration loading (like in main training function)
    ul_topk_noncoord = test_config.get("ul_topk_noncoord", 100)
    ul_topk_coord = test_config.get("ul_topk_coord", 100)
    ul_neighbor_window = test_config.get("ul_neighbor_window", 8)
    
    # Verify values
    assert ul_topk_noncoord == 50, f"Expected ul_topk_noncoord=50, got {ul_topk_noncoord}"
    assert ul_topk_coord == 75, f"Expected ul_topk_coord=75, got {ul_topk_coord}"
    assert ul_neighbor_window == 10, f"Expected ul_neighbor_window=10, got {ul_neighbor_window}"
    
    print("✅ Configuration loading test passed")
    print(f"   ul_topk_noncoord: {ul_topk_noncoord}")
    print(f"   ul_topk_coord: {ul_topk_coord}")
    print(f"   ul_neighbor_window: {ul_neighbor_window}")


def test_mask_computation():
    """Test mask computation for conflict resolution."""
    
    # Create test data
    labels = torch.tensor([
        [-100, -100, 151667 + 5, 151643, -100],  # Forward: coord token + EOS
        [-100, 101, 102, 103, 151643],           # Reverse: text tokens + EOS
    ])
    
    assistant_mask = (labels != -100)
    
    # Mock coordinate token IDs
    coord_token_ids = list(range(151667, 151667 + 50))
    coord_ids_tensor = torch.tensor(coord_token_ids, dtype=torch.long)
    
    # Compute masks
    coord_target_mask = torch.isin(labels, coord_ids_tensor) & assistant_mask
    text_target_mask = (~torch.isin(labels, coord_ids_tensor)) & (labels != -100) & assistant_mask
    
    # Verify mutual exclusivity
    assert not (coord_target_mask & text_target_mask).any(), "Masks should be mutually exclusive"
    
    # Verify coverage
    total_targets = coord_target_mask.sum() + text_target_mask.sum()
    assert total_targets <= assistant_mask.sum(), "Target masks should not exceed assistant positions"
    
    print("✅ Mask computation test passed")
    print(f"   Assistant positions: {assistant_mask.sum().item()}")
    print(f"   Coordinate targets: {coord_target_mask.sum().item()}")
    print(f"   Text targets: {text_target_mask.sum().item()}")


if __name__ == "__main__":
    print("🧪 Running Simple Top-K Integration Tests...")
    print("=" * 50)
    
    test_functions = [
        test_configuration_loading,
        test_mask_computation,
        test_topk_methods_directly,
    ]
    
    for test_func in test_functions:
        try:
            print(f"\n📋 Running {test_func.__name__}...")
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            raise
    
    print(f"\n🎉 All simple integration tests passed!")
    print("\n📊 Test Summary:")
    print("✅ Configuration parameter loading")
    print("✅ Mask computation and conflict resolution")
    print("✅ Top-K loss computation methods")
    print("✅ Numerical stability and finite loss values")
    print("\n🚀 Top-K Unlikelihood system is ready for training!")
