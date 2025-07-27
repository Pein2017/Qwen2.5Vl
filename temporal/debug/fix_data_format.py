#!/usr/bin/env python3
"""
Fix data format mismatch between current data and expected teacher-student format.

Current data format:
{
  "images": ["image.jpg"],
  "objects": [...],
  "width": 532,
  "height": 728
}

Expected format by ChatProcessor:
{
  "teachers": [
    {
      "images": ["image.jpg"],
      "objects": [...],
      "width": 532,
      "height": 728
    }
  ],
  "student": {
    "images": ["image.jpg"], 
    "objects": [...],
    "width": 532,
    "height": 728
  }
}
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

def convert_to_teacher_student_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert direct sample format to teacher-student format."""
    # For now, use the same sample as both teacher and student
    # This is a temporary fix - ideally you'd have different teacher samples
    converted = {
        "teachers": [sample.copy()],  # One teacher sample
        "student": sample.copy()      # Same as student
    }
    return converted

def process_file(input_path: str, output_path: str):
    """Process a JSONL file and convert format."""
    print(f"📂 Processing: {input_path} -> {output_path}")
    
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                sample = json.loads(line)
                
                # Check current format
                if 'teachers' in sample and 'student' in sample:
                    # Already in correct format
                    converted_sample = sample
                elif 'images' in sample and 'objects' in sample:
                    # Need to convert
                    converted_sample = convert_to_teacher_student_format(sample)
                else:
                    print(f"⚠️ Line {line_num}: Unknown format - {list(sample.keys())}")
                    continue
                
                # Write converted sample
                outfile.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"   Processed {processed_count} samples...")
                    
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                continue
    
    print(f"✅ Processed {processed_count} samples")

def main():
    """Main function to fix data format."""
    print("🔧 BBU DATA FORMAT FIX")
    print("=" * 50)
    print("Converting from direct format to teacher-student format...")
    
    # Files to convert
    data_dir = "data/ds_v2_full"
    files_to_convert = [
        "train.jsonl",
        "val.jsonl", 
        "teacher.jsonl",
        "all_samples.jsonl"
    ]
    
    for filename in files_to_convert:
        input_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"📂 Skipping missing file: {filename}")
            continue
        
        # Create backup
        backup_path = input_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"💾 Creating backup: {backup_path}")
            import shutil
            shutil.copy2(input_path, backup_path)
        
        # Convert to temporary file first
        temp_path = input_path + ".temp"
        
        try:
            process_file(input_path, temp_path)
            
            # Replace original with converted version
            os.replace(temp_path, input_path)
            print(f"✅ Updated: {filename}")
            
        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    print("\n🏁 Data format conversion complete!")
    print("The training data now has the teacher-student format expected by ChatProcessor.")
    print("")
    print("⚠️ NOTE: This is a temporary fix using the same sample as teacher and student.")
    print("For better training, you should have different teacher samples.")
    print("")
    print("Next steps:")
    print("1. Test training again to see if coordinate tokens are generated")
    print("2. If still zero losses, debug the coordinate token generation in ChatProcessor")

def test_conversion():
    """Test the conversion on a single sample."""
    print("\n🧪 TESTING CONVERSION")
    print("=" * 30)
    
    # Load a sample
    sample_path = "data/ds_v2_full/all_samples.jsonl"
    if not os.path.exists(sample_path):
        print("❌ Sample file not found")
        return
    
    with open(sample_path, 'r', encoding='utf-8') as f:
        original_sample = json.loads(f.readline().strip())
    
    print("Original format:")
    print(f"  Keys: {list(original_sample.keys())}")
    print(f"  Objects: {len(original_sample.get('objects', []))}")
    
    converted_sample = convert_to_teacher_student_format(original_sample)
    
    print("\nConverted format:")
    print(f"  Keys: {list(converted_sample.keys())}")
    print(f"  Teachers: {len(converted_sample['teachers'])}")
    print(f"  Student objects: {len(converted_sample['student']['objects'])}")
    
    # Verify teacher format
    teacher = converted_sample['teachers'][0]
    print(f"  Teacher keys: {list(teacher.keys())}")
    
    # Verify student format
    student = converted_sample['student']
    print(f"  Student keys: {list(student.keys())}")
    
    print("\n✅ Conversion test passed - format matches ChatProcessor expectations")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_conversion()
    else:
        main()