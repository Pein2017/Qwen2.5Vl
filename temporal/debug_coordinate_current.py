#!/usr/bin/env python3
"""
Debug script to check current coordinate token processing state.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config
from src.chat_processor import ChatProcessor
from src.utils.tokens import SpecialTokens
from src.utils.coordinate_token_manager import create_coordinate_token_manager

def test_coordinate_token_processing():
    """Test how coordinate tokens are processed in labels."""
    
    print("🧪 Testing coordinate token processing in labels...")
    
    # Initialize config
    config_path = "configs/base_flat_det.yaml"
    init_config(config_path)
    
    print(f"✅ Config loaded: coordinate_tokens_enabled = {config.coordinate_tokens_enabled}")
    
    # Initialize basic components
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    print(f"✅ Tokenizer loaded: vocab_size = {tokenizer.vocab_size}")
    
    # Create coordinate token manager
    manager = create_coordinate_token_manager(
        tokenizer=tokenizer,
        enable_coordinate_tokens=config.coordinate_tokens_enabled,
        max_coord_value=config.max_coord_value,
        box_start_id=config.box_start_id,
        box_end_id=config.box_end_id,
    )
    
    print(f"✅ Coordinate manager created: enabled = {manager.enabled}")
    print(f"   Box start ID: {manager.box_start_id}")
    print(f"   Box end ID: {manager.box_end_id}")
    print(f"   Coord token range: [{manager.coord_token_start}, {manager.coord_token_end})")
    
    # Test coordinate token conversion
    sample_json = '[{"bbox_2d": [100, 200, 300, 400], "label": "Object 1"}]'
    print(f"\n🧪 Testing coordinate conversion...")
    print(f"   Input JSON: {sample_json}")
    
    # This should create coordinate tokens
    coord_format = manager.convert_json_to_coordinate_format(sample_json)
    print(f"   Coordinate format: {coord_format}")
    
    # Test tokenization of coordinate format
    tokenized_coords = tokenizer.encode(coord_format, add_special_tokens=False)
    print(f"   Tokenized coordinate tokens: {tokenized_coords}")
    
    # Check if coordinate tokens are in the expected range
    coord_tokens = []
    for token_id in tokenized_coords:
        if manager.coord_token_start <= token_id < manager.coord_token_end:
            coord_tokens.append(token_id)
    
    print(f"   Coordinate token IDs found: {coord_tokens}")
    
    # Test chat processor with coordinate tokens
    print(f"\n🧪 Testing chat processor with coordinate tokens...")
    
    # Create chat processor
    chat_processor = ChatProcessor(
        tokenizer=tokenizer,
        image_processor=None,
        data_root=Path("/tmp"),
        enable_coordinate_tokens=True,
        max_coord_value=config.max_coord_value,
        box_start_id=config.box_start_id,
        box_end_id=config.box_end_id,
    )
    
    # Create simple conversation with detection
    from src.utils.schema import ChatMessage
    
    conversation = [
        ChatMessage(role="user", content="Detect objects in this image."),
        ChatMessage(role="assistant", content=coord_format)
    ]
    
    print(f"   Assistant message: {conversation[1].content}")
    
    # This should produce labels with coordinate tokens
    try:
        output = chat_processor.process_conversation(
            conversation=conversation,
            image_paths=[],
            ground_truth_objects=[]
        )
        
        print(f"   Input IDs shape: {output.input_ids.shape}")
        print(f"   Labels shape: {output.labels.shape}")
        
        # Check labels for coordinate tokens
        labels_1d = output.labels.view(-1)
        valid_labels = labels_1d[labels_1d != -100]
        
        print(f"   Valid labels count: {len(valid_labels)}")
        print(f"   Valid labels range: [{valid_labels.min()}, {valid_labels.max()}]")
        
        # Find coordinate tokens in labels
        coord_in_labels = []
        for token_id in valid_labels:
            if manager.coord_token_start <= token_id < manager.coord_token_end:
                coord_in_labels.append(token_id.item())
        
        print(f"   Coordinate tokens in labels: {coord_in_labels}")
        
        if coord_in_labels:
            print("✅ Coordinate tokens found in labels!")
            return True
        else:
            print("❌ No coordinate tokens found in labels!")
            
            # Debug: check what's in the labels
            print("   All valid label values:")
            unique_labels = torch.unique(valid_labels)
            for label in unique_labels[:20]:  # Show first 20 unique labels
                print(f"     {label.item()}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error processing conversation: {e}")
        return False

if __name__ == "__main__":
    success = test_coordinate_token_processing()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}")
