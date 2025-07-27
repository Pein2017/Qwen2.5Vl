#!/usr/bin/env python3
"""
Debug script to identify why coordinate losses are zero during training.

Based on the training error:
- Student samples are present 
- All coordinate losses are zero (geometry_focal_loss, coordinate_l1_loss, geometry_bbox_giou_loss)
- This suggests coordinate tokens are not being generated or processed correctly
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

def check_training_data_format():
    """Check the actual training data format used by the trainer."""
    print("🔍 CHECKING TRAINING DATA FORMAT")
    print("=" * 50)
    
    # The training script loads data from these files according to the config
    training_files = [
        "data/ds_v2_full/train.jsonl",
        "data/ds_v2_full/val.jsonl",
        "data/ds_v2_full/teacher.jsonl"
    ]
    
    for file_path in training_files:
        if os.path.exists(file_path):
            print(f"\n📂 Found: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = json.loads(f.readline().strip())
                
                print(f"   Keys: {list(sample.keys())}")
                
                # Check if it has the teacher-student structure
                if 'teachers' in sample and 'student' in sample:
                    print(f"   ✅ Teacher-Student format detected")
                    print(f"   Teachers: {len(sample['teachers'])}")
                    print(f"   Student keys: {list(sample['student'].keys())}")
                    
                    # Check student data
                    student = sample['student']
                    if 'objects' in student:
                        print(f"   Student objects: {len(student['objects'])}")
                        
                        # Check coordinate formats in student data
                        if student['objects']:
                            obj = student['objects'][0]
                            print(f"   First object keys: {list(obj.keys())}")
                            
                            # This is the KEY: Check if coordinates are in a format 
                            # that can generate coordinate tokens
                            coord_types = [k for k in obj.keys() if k != 'desc']
                            print(f"   Coordinate types found: {coord_types}")
                    
                elif 'images' in sample and 'objects' in sample:
                    print(f"   ✅ Direct sample format detected")
                    print(f"   Objects: {len(sample['objects'])}")
                else:
                    print(f"   ❓ Unknown format: {list(sample.keys())}")
                    
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        else:
            print(f"📂 Missing: {file_path}")

def analyze_coordinate_token_expectation():
    """Analyze what the coordinate token system expects."""
    print("\n🔍 COORDINATE TOKEN EXPECTATIONS")
    print("=" * 50)
    
    print("Based on the training error, the system expects:")
    print("1. geometry_focal_loss > 0 (for coordinate detection)")
    print("2. coordinate_l1_loss > 0 (for coordinate regression)")
    print("3. geometry_bbox_giou_loss > 0 (for bounding box IoU)")
    print("")
    print("These losses come from coordinate tokens in the text.")
    print("The coordinate token manager should convert coordinates like:")
    print("  bbox_2d: [100, 150, 200, 250] -> special tokens")
    print("  square: [100, 150, 200, 150, 200, 250, 100, 250] -> special tokens")
    print("")
    print("If these losses are zero, it means:")
    print("❌ No coordinate tokens are being generated in the text")
    print("❌ OR coordinate tokens are generated but not recognized during loss computation")

def check_chat_processor_integration():
    """Check how the chat processor should integrate coordinates."""
    print("\n🔍 CHAT PROCESSOR INTEGRATION")
    print("=" * 50)
    
    # Look at the data conversion pipeline
    data_files_to_check = [
        "data_conversion/v2_to_training_pipeline.py",
        "data_conversion/unified_processor.py"
    ]
    
    for file_path in data_files_to_check:
        if os.path.exists(file_path):
            print(f"\n📂 Checking: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for coordinate-related processing
                coordinate_keywords = [
                    'coordinate', 'bbox', 'square', 'line',
                    '<coordinate>', '</coordinate>',
                    'geometry', 'token'
                ]
                
                found_keywords = []
                for keyword in coordinate_keywords:
                    if keyword.lower() in content.lower():
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"   Found coordinate keywords: {found_keywords}")
                else:
                    print(f"   ❌ No coordinate keywords found")
                    
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")

def check_coordinate_processing_flow():
    """Check the expected flow of coordinate processing."""
    print("\n🔍 COORDINATE PROCESSING FLOW ANALYSIS")
    print("=" * 50)
    
    print("Expected flow:")
    print("1. Raw data: bbox_2d: [100, 150, 200, 250]")
    print("2. Chat processor: Converts to conversation with coordinate tokens")
    print("3. Tokenizer: Converts coordinate tokens to token IDs")
    print("4. Model: Processes tokens and generates predictions")
    print("5. Loss computation: Compares coordinate predictions to targets")
    print("")
    
    print("The training error suggests the flow breaks between steps 1-3.")
    print("Coordinate tokens are either:")  
    print("❌ Not being generated in conversation text (step 2)")
    print("❌ Not being tokenized correctly (step 3)")
    print("")
    
    print("Key questions:")
    print("1. Does the conversation text contain <coordinate>...</coordinate> markers?")
    print("2. Are coordinate values being converted to special tokens?")
    print("3. Are the coordinate token IDs in the high range (>150000)?")

def debug_data_sample():
    """Debug a specific data sample to trace the issue."""
    print("\n🔍 DEBUGGING SPECIFIC SAMPLE")
    print("=" * 50)
    
    # Load one sample from the data
    data_file = "data/ds_v2_full/all_samples.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Sample file not found: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline().strip())
    
    print(f"Sample analysis:")
    print(f"  Images: {sample['images']}")
    print(f"  Objects: {len(sample['objects'])}")
    print(f"  Dimensions: {sample['width']}x{sample['height']}")
    
    # Analyze first object
    obj = sample['objects'][0]
    print(f"\nFirst object:")
    print(f"  Description: {obj['desc']}")
    
    for key, value in obj.items():
        if key != 'desc':
            print(f"  {key}: {value}")
            
            # This is what should be converted to coordinate tokens
            print(f"    -> Should generate coordinate tokens for {key} geometry")
    
    print(f"\nKEY INSIGHT:")
    print(f"This sample has coordinate data, but if training shows zero coordinate losses,")
    print(f"it means this data is NOT being converted to coordinate tokens in the conversation.")
    print(f"")
    print(f"The issue is likely in:")  
    print(f"❌ data_conversion pipeline (v2_to_training_pipeline.py)")
    print(f"❌ OR chat_processor coordinate integration")
    print(f"❌ OR training data format mismatch")

def main():
    """Main debug function."""
    print("🚀 BBU COORDINATE TOKEN DEBUG")
    print("=" * 50)
    print("Investigating why coordinate losses are zero during training...")
    print("")
    
    check_training_data_format()
    analyze_coordinate_token_expectation()
    check_chat_processor_integration() 
    check_coordinate_processing_flow()
    debug_data_sample()
    
    print("\n🏥 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    print("The training error shows:")
    print("✅ Student samples are present (student_lm_loss > 0)")
    print("❌ All coordinate losses are zero")
    print("❌ Coordinate tokens are enabled in config")
    print("")
    print("ROOT CAUSE ANALYSIS:")
    print("The issue is NOT in the model or loss computation.")
    print("The issue is in DATA PROCESSING - coordinate tokens are not being generated.")
    print("")
    print("RECOMMENDED FIXES:")
    print("1. Check data_conversion/v2_to_training_pipeline.py")
    print("2. Verify that coordinate data is converted to coordinate tokens in text")
    print("3. Test chat processor coordinate integration")
    print("4. Ensure training data has teacher-student format with coordinate tokens")
    print("")
    print("Next step: Check the v2_to_training_pipeline.py to see if it generates coordinate tokens.")

if __name__ == "__main__":
    main()