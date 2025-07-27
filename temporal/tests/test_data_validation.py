#!/usr/bin/env python3
"""
Test script to validate training samples in data/ds_v2_full/all_samples.jsonl
Checks for coordinate token generation and data format issues.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

from src.chat_processor import ChatProcessor
from src.config import init_config, config

def load_samples(file_path: str, limit: int = 10) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                samples.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(samples)} samples from {file_path}")
        return samples
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return []

def analyze_sample_structure(sample: Dict) -> Dict[str, Any]:
    """Analyze the structure of a single sample."""
    analysis = {
        'has_images': 'images' in sample and len(sample['images']) > 0,
        'has_objects': 'objects' in sample and len(sample['objects']) > 0,
        'image_count': len(sample.get('images', [])),
        'object_count': len(sample.get('objects', [])),
        'width': sample.get('width'),
        'height': sample.get('height'),
        'geometry_types': set(),
        'object_types': [],
        'coordinate_formats': set()
    }
    
    # Analyze objects
    for obj in sample.get('objects', []):
        analysis['object_types'].append(obj.get('desc', 'unknown'))
        
        # Check geometry types
        for key in obj.keys():
            if key != 'desc':
                analysis['geometry_types'].add(key)
                
                # Check coordinate format
                coords = obj[key]
                if isinstance(coords, list):
                    if len(coords) == 4:  # bbox_2d format
                        analysis['coordinate_formats'].add('bbox_2d')
                    elif len(coords) == 8:  # square format
                        analysis['coordinate_formats'].add('square_8points')
                    elif len(coords) > 8:  # line or complex polygon
                        analysis['coordinate_formats'].add('multi_point')
    
    return analysis

def test_chat_processor_conversion(sample: Dict) -> Dict[str, Any]:
    """Test the chat processor conversion for coordinate tokens."""
    try:  
        from src.config import get_config
        current_config = get_config()
        processor = ChatProcessor(current_config)
        
        # Process the sample
        processed = processor.process_sample(sample)
        
        result = {
            'success': True,
            'has_input_ids': 'input_ids' in processed,
            'has_labels': 'labels' in processed,
            'input_length': len(processed.get('input_ids', [])),
            'labels_length': len(processed.get('labels', [])),
            'coordinate_tokens_found': False,
            'coordinate_token_count': 0,
            'conversation_structure': []
        }
        
        # Check for coordinate tokens in the conversation
        if 'conversation' in processed:
            for turn in processed['conversation']:
                result['conversation_structure'].append({
                    'role': turn.get('role'),
                    'content_length': len(turn.get('content', '')),
                    'has_coordinate_markers': '<coordinate>' in turn.get('content', '') or '</coordinate>' in turn.get('content', '')
                })
        
        # Check input_ids for coordinate tokens
        input_ids = processed.get('input_ids', [])
        if input_ids:
            # Look for coordinate token patterns (typically high-value token IDs)
            # Based on the coordinate token manager, coordinate tokens are usually in a specific range
            coordinate_tokens = [token for token in input_ids if token > 150000]  # Rough estimate
            result['coordinate_token_count'] = len(coordinate_tokens)
            result['coordinate_tokens_found'] = len(coordinate_tokens) > 0
            
            # Sample some token IDs for analysis
            result['token_id_sample'] = input_ids[:20] if len(input_ids) > 20 else input_ids
            result['high_value_tokens'] = coordinate_tokens[:10] if coordinate_tokens else []
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def main():
    """Main validation function."""
    print("🔍 BBU Training Data Validation Script")
    print("=" * 50)
    
    # Check if data file exists
    data_file = "data/ds_v2_full/all_samples.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        
        # Try alternative locations
        alternative_paths = [
            "data_conversion/output/ds_v2/all_samples.jsonl",
            "ds_v2/all_samples.jsonl"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found alternative data file: {alt_path}")
                data_file = alt_path
                break
        else:
            print("❌ No valid data file found")
            return
    
    print(f"📂 Using data file: {data_file}")
    
    # Load and analyze samples
    samples = load_samples(data_file, limit=5)  # Test with first 5 samples
    
    if not samples:
        print("❌ No samples loaded. Exiting.")
        return
    
    # Load configuration
    try:
        loaded_config = init_config("configs/bbu_v2.yaml")
        print(f"✅ Loaded config with coordinate_tokens_enabled: {loaded_config.coordinate_config_enable_coordinate_tokens}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    print("\n📊 Sample Analysis Results:")
    print("-" * 30)
    
    all_analyses = []
    processing_results = []
    
    for i, sample in enumerate(samples):
        print(f"\n🔍 Sample {i+1}:")
        
        # Analyze structure
        analysis = analyze_sample_structure(sample)
        all_analyses.append(analysis)
        
        print(f"  📷 Images: {analysis['image_count']}")
        print(f"  🎯 Objects: {analysis['object_count']}")
        print(f"  📐 Geometry types: {analysis['geometry_types']}")
        print(f"  📏 Coordinate formats: {analysis['coordinate_formats']}")
        
        # Test chat processor
        proc_result = test_chat_processor_conversion(sample)
        processing_results.append(proc_result)
        
        if proc_result['success']:
            print(f"  ✅ Chat processing: SUCCESS")
            print(f"  📝 Input length: {proc_result['input_length']}")
            print(f"  🏷️ Labels length: {proc_result['labels_length']}")
            print(f"  🎯 Coordinate tokens found: {proc_result['coordinate_tokens_found']}")
            print(f"  🔢 Coordinate token count: {proc_result['coordinate_token_count']}")
            
            if proc_result['high_value_tokens']:
                print(f"  🔍 Sample coordinate tokens: {proc_result['high_value_tokens']}")
            
            # Check conversation structure
            if proc_result['conversation_structure']:
                for j, turn in enumerate(proc_result['conversation_structure']):
                    print(f"    Turn {j}: {turn['role']} (len={turn['content_length']}, coords={turn['has_coordinate_markers']})")
        else:
            print(f"  ❌ Chat processing: FAILED - {proc_result['error']}")
    
    # Summary analysis
    print("\n📋 Summary Analysis:")
    print("-" * 20)
    
    successful_processing = sum(1 for r in processing_results if r['success'])
    coordinate_token_samples = sum(1 for r in processing_results if r.get('coordinate_tokens_found', False))
    
    print(f"✅ Successfully processed: {successful_processing}/{len(samples)}")
    print(f"🎯 Samples with coordinate tokens: {coordinate_token_samples}/{len(samples)}")
    
    if coordinate_token_samples == 0:
        print("\n❌ CRITICAL ISSUE: No coordinate tokens found in any sample!")
        print("   This explains why coordinate losses are zero during training.")
        print("   Possible causes:")
        print("   1. Chat processor not generating coordinate tokens")
        print("   2. Coordinate token manager not working properly") 
        print("   3. Data format incompatible with coordinate processing")
        print("   4. Configuration issue with coordinate token generation")
    
    # Check for specific patterns that might cause issues
    all_geometry_types = set()
    for analysis in all_analyses:
        all_geometry_types.update(analysis['geometry_types'])
    
    print(f"\n📐 All geometry types found: {all_geometry_types}")
    
    # Check for problematic patterns
    problematic_patterns = []
    for i, sample in enumerate(samples):
        # Check for missing coordinates
        if not sample.get('objects'):
            problematic_patterns.append(f"Sample {i+1}: No objects")
        
        # Check for invalid coordinate formats
        for obj in sample.get('objects', []):
            for key, coords in obj.items():
                if key != 'desc' and not isinstance(coords, list):
                    problematic_patterns.append(f"Sample {i+1}: Invalid coordinates for {key}")
    
    if problematic_patterns:
        print(f"\n⚠️ Problematic patterns found:")
        for pattern in problematic_patterns[:5]:  # Show first 5
            print(f"   - {pattern}")
    
    print(f"\n🏁 Analysis complete. Check results above for coordinate token issues.")

if __name__ == "__main__":
    main()