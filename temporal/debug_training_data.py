#!/usr/bin/env python3
"""
Debug script to examine the training data and understand coordinate token usage.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer


def analyze_training_data():
    """Analyze training data to understand coordinate token usage."""
    print("🔍 Analyzing training data coordinate token usage...")
    
    # Load tokenizer
    model_path = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"📊 Base tokenizer vocabulary size: {len(tokenizer.get_vocab())}")
    
    # Check training data
    data_file = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full/train.jsonl")
    if not data_file.exists():
        print(f"❌ Training data file not found: {data_file}")
        return
    
    print(f"📄 Analyzing training data: {data_file}")
    
    # Analyze a few samples
    coordinate_tokens_found = set()
    geometry_tokens_found = set()
    sample_count = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 5:  # Analyze first 5 samples
                break
                
            try:
                sample = json.loads(line.strip())
                sample_count += 1
                
                # Get the conversation text
                conversations = sample.get('conversations', [])
                for conv in conversations:
                    content = conv.get('value', '')
                    
                    # Tokenize the content
                    tokens = tokenizer.encode(content)
                    
                    # Check for coordinate and geometry tokens
                    for token_id in tokens:
                        try:
                            token_str = tokenizer.decode([token_id])
                            
                            # Check for coordinate tokens
                            if token_str.startswith('<coord_') and token_str.endswith('>'):
                                coordinate_tokens_found.add((token_id, token_str))
                            
                            # Check for geometry tokens
                            if any(geom in token_str for geom in ['box_start', 'box_end', 'square_start', 'square_end', 'line_start', 'line_end']):
                                geometry_tokens_found.add((token_id, token_str))
                                
                        except Exception:
                            continue
                            
            except json.JSONDecodeError:
                continue
    
    print(f"\n📊 Analysis Results (from {sample_count} samples):")
    print(f"   Coordinate tokens found: {len(coordinate_tokens_found)}")
    print(f"   Geometry tokens found: {len(geometry_tokens_found)}")
    
    if coordinate_tokens_found:
        print(f"\n🎯 Coordinate tokens in training data:")
        sorted_coords = sorted(coordinate_tokens_found, key=lambda x: x[0])
        for token_id, token_str in sorted_coords[:20]:  # Show first 20
            print(f"   {token_str} -> ID {token_id}")
        if len(sorted_coords) > 20:
            print(f"   ... and {len(sorted_coords) - 20} more")
            
        # Analyze coordinate token ranges
        coord_ids = [token_id for token_id, _ in coordinate_tokens_found]
        min_coord_id = min(coord_ids)
        max_coord_id = max(coord_ids)
        print(f"\n📊 Coordinate token ID range in data: [{min_coord_id}, {max_coord_id}]")
        
        # Check if they're consecutive
        expected_ids = set(range(min_coord_id, max_coord_id + 1))
        actual_ids = set(coord_ids)
        missing_ids = expected_ids - actual_ids
        
        if missing_ids:
            print(f"   ❌ Non-consecutive range, missing {len(missing_ids)} IDs")
            print(f"   Missing IDs: {sorted(list(missing_ids))[:10]}...")
        else:
            print(f"   ✅ Consecutive range")
    else:
        print(f"\n❌ No coordinate tokens found in training data!")
    
    if geometry_tokens_found:
        print(f"\n🎯 Geometry tokens in training data:")
        for token_id, token_str in sorted(geometry_tokens_found, key=lambda x: x[0]):
            print(f"   {token_str} -> ID {token_id}")
    else:
        print(f"\n❌ No geometry tokens found in training data!")
    
    # Compare with expected ranges from current system
    print(f"\n🔧 Expected coordinate token range (current system):")
    print(f"   Expected range: [151669, 153717) (from logs)")
    print(f"   Expected geometry tokens: <|box_start|> (151648), <|box_end|> (151649)")
    
    if coordinate_tokens_found:
        data_min = min(token_id for token_id, _ in coordinate_tokens_found)
        data_max = max(token_id for token_id, _ in coordinate_tokens_found)
        expected_min = 151669
        expected_max = 153717
        
        print(f"\n⚠️  MISMATCH ANALYSIS:")
        print(f"   Data coordinate range: [{data_min}, {data_max}]")
        print(f"   Expected coordinate range: [{expected_min}, {expected_max})")
        print(f"   Offset: {data_min - expected_min}")
        
        if data_min < expected_min:
            print(f"   🚨 ISSUE: Training data uses OLDER coordinate token configuration!")
            print(f"   🔧 SOLUTION: Reprocess training data with current tokenizer")
        elif data_min > expected_min:
            print(f"   🚨 ISSUE: Training data uses NEWER coordinate token configuration!")
            print(f"   🔧 SOLUTION: Update coordinate manager to match data")
        else:
            print(f"   ✅ Coordinate ranges match!")
    
    return coordinate_tokens_found, geometry_tokens_found


if __name__ == "__main__":
    try:
        coord_tokens, geom_tokens = analyze_training_data()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
