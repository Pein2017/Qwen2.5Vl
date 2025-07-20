#!/usr/bin/env python3
"""
Debug script to test coordinate loss computation with safer bounds checking.
This will help identify the exact cause of the CUDA index out of bounds error.
"""

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_coordinate_loss_bounds():
    """Test coordinate loss computation with various tensor configurations."""
    
    # Test case 1: Regular scenario
    print("=== Test Case 1: Regular scenario ===")
    
    # Simulate typical dimensions
    batch_size, seq_len = 2, 100
    vocab_size = 153713  # Example original vocab size from logs
    extended_vocab_size = vocab_size + 2048  # With coordinate tokens
    
    # Create sample tensors
    logits = torch.randn(batch_size, seq_len, extended_vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Add some coordinate tokens (simulate actual training data)
    coord_start = vocab_size
    coord_end = vocab_size + 2048
    
    # Replace some labels with coordinate tokens
    labels[0, 10:14] = torch.randint(coord_start, coord_end, (4,))
    labels[1, 20:24] = torch.randint(coord_start, coord_end, (4,))
    
    # Add ignore index
    labels[:, 0] = -100  # Start token typically ignored
    
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"Extended vocab size: {extended_vocab_size}")
    print(f"Labels range: [{labels[labels != -100].min()}, {labels[labels != -100].max()}]")
    
    # Simulate coordinate loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, extended_vocab_size)
    flat_labels = shift_labels.view(-1)
    
    valid_mask = flat_labels != -100
    
    if not valid_mask.any():
        print("ERROR: No valid labels found!")
        return
    
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    print(f"Valid logits shape: {valid_logits.shape}")
    print(f"Valid labels shape: {valid_labels.shape}")
    print(f"Valid labels range: [{valid_labels.min()}, {valid_labels.max()}]")
    
    # Create coordinate mask (simulate bbox-aware detection)
    coordinate_mask = (valid_labels >= coord_start) & (valid_labels < coord_end)
    
    coord_indices = torch.nonzero(coordinate_mask, as_tuple=False).squeeze(-1)
    regular_indices = torch.nonzero(~coordinate_mask, as_tuple=False).squeeze(-1)
    
    print(f"Coordinate tokens: {coord_indices.size(0)}")
    print(f"Regular tokens: {regular_indices.size(0)}")
    
    # Check bounds
    max_valid_idx = valid_logits.size(0) - 1
    print(f"Max valid index: {max_valid_idx}")
    
    if coord_indices.numel() > 0:
        print(f"Coord indices range: [{coord_indices.min()}, {coord_indices.max()}]")
        if coord_indices.max() > max_valid_idx:
            print(f"ERROR: Coord indices out of bounds! Max: {coord_indices.max()}, Valid: {max_valid_idx}")
    
    if regular_indices.numel() > 0:
        print(f"Regular indices range: [{regular_indices.min()}, {regular_indices.max()}]")
        if regular_indices.max() > max_valid_idx:
            print(f"ERROR: Regular indices out of bounds! Max: {regular_indices.max()}, Valid: {max_valid_idx}")
    
    # Test regular token loss computation
    if regular_indices.size(0) > 0:
        print("\n=== Testing regular token loss ===")
        
        # Extract regular tokens
        regular_logits = valid_logits[regular_indices]
        regular_labels = valid_labels[regular_indices]
        
        print(f"Regular logits shape: {regular_logits.shape}")
        print(f"Regular labels shape: {regular_labels.shape}")
        print(f"Regular labels range: [{regular_labels.min()}, {regular_labels.max()}]")
        
        # Check if labels are within vocab bounds
        if regular_labels.max() >= vocab_size:
            print(f"ERROR: Regular labels out of vocab bounds! Max: {regular_labels.max()}, Vocab: {vocab_size}")
        
        # Simulate loss computation
        try:
            regular_logits_truncated = regular_logits[:, :vocab_size]
            clamped_labels = torch.clamp(regular_labels, 0, vocab_size - 1)
            
            loss = torch.nn.functional.cross_entropy(regular_logits_truncated, clamped_labels)
            print(f"Regular loss computed successfully: {loss.item()}")
        except Exception as e:
            print(f"ERROR in regular loss computation: {e}")
    
    print("=== Test completed ===\n")

if __name__ == "__main__":
    test_coordinate_loss_bounds()