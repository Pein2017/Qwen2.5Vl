#!/root/miniconda3/envs/ms/bin/python
"""
Debug script to check if square and line geometries are being detected properly
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

import json
from pathlib import Path

def analyze_training_data():
    """Analyze the training data to see what geometry types are present."""
    
    # Load training data
    train_path = "data/ds_v2_full/train.jsonl"
    
    if not Path(train_path).exists():
        print(f"❌ Training data not found at: {train_path}")
        return
        
    print(f"🔍 Analyzing training data: {train_path}")
    
    bbox_count = 0
    square_count = 0
    line_count = 0
    total_objects = 0
    total_samples = 0
    
    with open(train_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line.strip())
                total_samples += 1
                
                # Check teacher objects
                for teacher in sample.get('teachers', []):
                    for obj in teacher.get('objects', []):
                        total_objects += 1
                        if 'bbox_2d' in obj:
                            bbox_count += 1
                        elif 'square' in obj:
                            square_count += 1
                        elif 'line' in obj:
                            line_count += 1
                
                # Check student objects
                student = sample.get('student', sample)
                for obj in student.get('objects', []):
                    total_objects += 1
                    if 'bbox_2d' in obj:
                        bbox_count += 1
                    elif 'square' in obj:
                        square_count += 1
                    elif 'line' in obj:
                        line_count += 1
                        
                # Show first few samples with different geometries
                if line_num <= 5:
                    print(f"\n📝 Sample {line_num}:")
                    print(f"   Teachers: {len(sample.get('teachers', []))}")
                    student = sample.get('student', sample)
                    student_objects = student.get('objects', [])
                    print(f"   Student objects: {len(student_objects)}")
                    
                    for i, obj in enumerate(student_objects[:3]):  # Show first 3 objects
                        geometry_type = None
                        if 'bbox_2d' in obj:
                            geometry_type = f"bbox_2d: {obj['bbox_2d']}"
                        elif 'square' in obj:
                            geometry_type = f"square: {obj['square']}"
                        elif 'line' in obj:
                            geometry_type = f"line: {obj['line']}"
                        else:
                            geometry_type = "unknown"
                        print(f"     Object {i}: {obj.get('desc', 'no desc')} - {geometry_type}")
                        
            except Exception as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                continue
    
    print(f"\n📊 Geometry Analysis Results:")
    print(f"   Total samples: {total_samples}")
    print(f"   Total objects: {total_objects}")
    print(f"   bbox_2d objects: {bbox_count} ({bbox_count/total_objects*100:.1f}%)")
    print(f"   square objects: {square_count} ({square_count/total_objects*100:.1f}%)")
    print(f"   line objects: {line_count} ({line_count/total_objects*100:.1f}%)")
    
    if square_count == 0 and line_count == 0:
        print(f"\n❌ ISSUE FOUND: No square or line geometries in training data!")
        print(f"   This explains why square/line losses are always 0.0")
        print(f"   All objects are bbox_2d, so only bbox losses are computed")
    else:
        print(f"\n✅ Multi-geometry data found - need to check coordinate token generation")

if __name__ == "__main__":
    analyze_training_data()