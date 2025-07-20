#!/usr/bin/env python3
"""
Test coordinate token conversion in training data
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import json
from pathlib import Path
from src.config import init_config
from src.core.data_processor import DataProcessor
from src.models.model_loader import load_model_and_processor_unified

def test_coordinate_conversion():
    """Test coordinate token conversion during data processing."""
    
    print("🔍 Testing Coordinate Token Conversion")
    print("=" * 50)
    
    # Initialize config
    init_config("configs/base_flat_det.yaml")
    
    # Create model and tokenizer
    print("🤖 Loading model and tokenizer...")
    from src.config import config
    model, tokenizer, image_processor = load_model_and_processor_unified(config.model_path)
    print(f"✅ Model loaded, vocab size: {len(tokenizer.get_vocab())}")
    
    # Create data processor
    print("📊 Creating data processor...")
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        use_new_config=False
    )
    
    # Check if coordinate tokens are enabled in chat processor
    print(f"📋 ChatProcessor coordinate tokens enabled: {data_processor.chat_processor.coordinate_processor.enabled}")
    print(f"📋 ChatProcessor max coord value: {data_processor.chat_processor.coordinate_processor.config.max_coord_value}")
    
    # Load a sample from training data
    train_file = Path("data/train.jsonl")
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return
    
    # Read first sample
    with open(train_file, 'r', encoding='utf-8') as f:
        sample_line = f.readline().strip()
        sample = json.loads(sample_line)
    
    print(f"\n📄 Original sample objects:")
    for i, obj in enumerate(sample.get('objects', [])):
        print(f"   {i+1}. {obj}")
    
    # Process the sample through chat processor
    print(f"\n🔄 Processing sample through ChatProcessor...")
    try:
        processed_output = data_processor.chat_processor.process_sample(sample)
        
        # Decode the input_ids to see the actual text
        input_text = tokenizer.decode(processed_output.input_ids.squeeze(), skip_special_tokens=False)
        
        # Look for coordinate tokens in the text
        print(f"\n📝 Processed text sample (last 1000 chars):")
        print(f"   {repr(input_text[-1000:])}")
        
        # Check for coordinate token patterns
        coord_token_count = input_text.count('<coord_')
        box_start_count = input_text.count('<|box_start|>')
        box_end_count = input_text.count('<|box_end|>')
        
        print(f"\n🎯 Coordinate Token Analysis:")
        print(f"   <coord_* tokens found: {coord_token_count}")
        print(f"   <|box_start|> tokens found: {box_start_count}")
        print(f"   <|box_end|> tokens found: {box_end_count}")
        
        if coord_token_count > 0:
            print("✅ SUCCESS: Coordinate tokens are being generated!")
        else:
            print("❌ ISSUE: No coordinate tokens found in processed text")
            
            # Try manual conversion to see if the processor works
            print(f"\n🔧 Testing manual coordinate conversion...")
            test_json = '[{"bbox_2d": [204, 409, 1638, 1843], "label": "test object"}]'
            converted = data_processor.chat_processor.coordinate_processor.convert_json_to_coordinate_format(test_json)
            print(f"   Test JSON: {test_json}")
            print(f"   Converted: {converted}")
            
    except Exception as e:
        print(f"❌ Error processing sample: {e}")
        import traceback
        traceback.print_exc()
    
    # Also test the coordinate processor directly
    print(f"\n🧪 Testing coordinate processor directly...")
    processor = data_processor.chat_processor.coordinate_processor
    
    # Create test objects from the actual sample
    test_objects = sample.get('objects', [])[:1]  # Just test first object
    if test_objects:
        obj = test_objects[0]
        bbox = obj.get('bbox_2d', [0, 0, 100, 100])
        desc = obj.get('desc', 'test')
        
        print(f"   Original bbox: {bbox}")
        print(f"   Original desc: {desc}")
        
        # Convert to coordinate format
        test_json = json.dumps([{"bbox_2d": bbox, "label": desc}], ensure_ascii=False)
        coord_format = processor.convert_json_to_coordinate_format(test_json)
        
        print(f"   JSON format: {test_json}")
        print(f"   Coordinate format: {coord_format}")
        
        # Convert back
        back_to_json = processor.convert_coordinate_to_json_format(coord_format)
        print(f"   Back to JSON: {back_to_json}")

if __name__ == "__main__":
    test_coordinate_conversion()