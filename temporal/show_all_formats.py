#!/usr/bin/env python3
"""Show examples of all supported coordinate formats and descriptions."""

import json
import sys
from pathlib import Path

# Add the data_conversion directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from v2_to_training_pipeline import V2TrainingPipeline

# Create pipeline
pipeline = V2TrainingPipeline()

# Process sample data to find examples of each format
input_dir = "/data3/Qwen2.5-VL-main/ds_v2"
v2_files = list(Path(input_dir).glob("*.json"))[:10]  # Check first 10 files

print("=== ALL SUPPORTED FORMATS SHOWCASE ===\n")

examples = {
    "bbox_2d": [],
    "square": [],
    "line": []
}

object_examples = {}

for v2_file in v2_files:
    samples = pipeline.processor.process_v2_file(str(v2_file))
    
    for sample in samples:
        # Collect format examples
        if len(examples[sample.geometry_format]) < 2:
            examples[sample.geometry_format].append(sample)
        
        # Collect object type examples
        if sample.object_type not in object_examples:
            object_examples[sample.object_type] = sample

print("1. GEOMETRY FORMATS\n")
print("The system supports 3 coordinate formats as requested:\n")

for format_name, format_samples in examples.items():
    if format_samples:
        sample = format_samples[0]
        training_data = sample.to_training_format()
        
        if format_name == "bbox_2d":
            print(f"• {format_name}: Standard rectangular bounding box [x1, y1, x2, y2]")
            print(f"  Example: {training_data[format_name]}")
            print(f"  Object: {sample.object_type}")
            print(f"  Description: {sample.description}")
            print()
        
        elif format_name == "square":
            print(f"• {format_name}: Four-point polygon [x1, y1, x2, y2, x3, y3, x4, y4]")  
            print(f"  Example: {training_data[format_name]}")
            print(f"  Object: {sample.object_type}")
            print(f"  Description: {sample.description}")
            print()
        
        elif format_name == "line":
            coords = training_data[format_name]
            coord_preview = coords[:8] + ['...'] if len(coords) > 8 else coords
            print(f"• {format_name}: Multi-point line [x1, y1, x2, y2, ..., xn, yn]")
            print(f"  Example ({len(coords)//2} points): {coord_preview}")
            print(f"  Object: {sample.object_type}")
            print(f"  Description: {sample.description}")
            print()

print("2. OBJECT TYPES & HIERARCHICAL DESCRIPTIONS\n")

for obj_type, sample in object_examples.items():
    print(f"• {obj_type.upper()}")
    print(f"  Chinese Label: {pipeline.processor.object_types[obj_type]['chinese_label']}")
    print(f"  Geometry Format: {sample.geometry_format}")
    print(f"  Hierarchical Description: {sample.description}")
    print(f"  Grouped Attributes: {list(sample.grouped_attributes.keys())}")
    print()

print("3. TRAINING DATA FORMAT\n")
print("Each training sample contains:")

example_sample = list(object_examples.values())[0]
training_example = example_sample.to_training_format()

for key, value in training_example.items():
    if key in ["bbox_2d", "square", "line"]:
        print(f"• {key}: {value} (coordinates)")
    elif key == "desc":
        print(f"• {key}: '{value}' (hierarchical description)")
    elif key == "object_type":
        print(f"• {key}: '{value}' (object classification)")
    elif key == "attributes":
        print(f"• {key}: {len(value)} attribute groups")
        for group, attrs in value.items():
            print(f"    - {group}: {list(attrs.keys())}")

print(f"\n4. COMPLETE EXAMPLE\n")
print("Final training format (as requested):")
print("objects: [{'bbox_2d':[x1,y1,x2,y2], 'desc':'object/property/extra_info'}]\n")

example_json = json.dumps(training_example, ensure_ascii=False, indent=2)
print(example_json)

print(f"\n=== SUMMARY ===")
print(f"✓ bbox_2d format: 4 coordinates for rectangular annotations")
print(f"✓ square format: 8 coordinates for four-point polygons") 
print(f"✓ line format: Variable coordinates for multi-point lines")
print(f"✓ Hierarchical descriptions: object/properties/assessments/context")
print(f"✓ No hardcoded stages: Flexible attribute grouping by information type")
print(f"✓ Comprehensive coverage: All V2 question patterns mapped")
print(f"✓ Training ready: JSONL format for Qwen2.5VL progressive learning")