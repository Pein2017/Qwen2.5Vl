#!/usr/bin/env python3
"""
Fix teacher pool to work with new teacher-student data format.

The teacher pool expects direct samples:
{"images": [...], "objects": [...]}

But our data now has teacher-student format:
{"teachers": [{"images": [...], "objects": [...]}], "student": {...}}

This script extracts the teacher samples from the teacher-student format.
"""

import json
import os

def extract_teacher_samples(teacher_file_path: str, output_file_path: str):
    """Extract teacher samples from teacher-student format."""
    print(f"🔧 Extracting teacher samples from {teacher_file_path}")
    
    teacher_samples = []
    
    with open(teacher_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check if it's teacher-student format
                if 'teachers' in data and isinstance(data['teachers'], list):
                    # Extract all teacher samples
                    for teacher in data['teachers']:
                        if 'images' in teacher and 'objects' in teacher:
                            teacher_samples.append(teacher)
                        else:
                            print(f"⚠️ Line {line_num}: Teacher missing 'images' or 'objects'")
                
                elif 'images' in data and 'objects' in data:
                    # Already in direct format
                    teacher_samples.append(data)
                    
                else:
                    print(f"⚠️ Line {line_num}: Unknown format - {list(data.keys())}")
                    
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
    
    # Write extracted teacher samples
    print(f"📝 Writing {len(teacher_samples)} teacher samples to {output_file_path}")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample in teacher_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Teacher samples extracted successfully")

def main():
    """Main function."""
    print("🔧 BBU TEACHER POOL FORMAT FIX")
    print("=" * 50)
    
    # Input and output paths
    teacher_file = "data/ds_v2_full/teacher.jsonl"
    teacher_pool_file = "data/ds_v2_full/teacher_pool.jsonl"
    
    # Create backup of original teacher file if teacher_pool.jsonl already exists
    if os.path.exists(teacher_pool_file):
        backup_file = teacher_pool_file + ".backup"
        if not os.path.exists(backup_file):
            print(f"💾 Creating backup: {backup_file}")
            import shutil
            shutil.copy2(teacher_pool_file, backup_file)
    
    # Extract teacher samples
    extract_teacher_samples(teacher_file, teacher_pool_file)
    
    # Verify the extracted samples
    print(f"\n🔍 Verifying extracted samples:")
    with open(teacher_pool_file, 'r', encoding='utf-8') as f:
        first_sample = json.loads(f.readline().strip())
        print(f"   Sample keys: {list(first_sample.keys())}")
        print(f"   Images: {len(first_sample.get('images', []))}")
        print(f"   Objects: {len(first_sample.get('objects', []))}")
    
    print(f"\n✅ Teacher pool format fix complete!")
    print(f"The teacher pool now has direct samples that the TeacherPoolManager expects.")

if __name__ == "__main__":
    main()