#!/usr/bin/env python3
"""
Final Demo: Complete V2 to Training Pipeline

This demonstrates the complete hierarchical data corpus system for Qwen2.5VL
with flexible taxonomy, multiple coordinate formats, and comprehensive coverage.
"""

import json
import sys
from pathlib import Path

# Add the data_conversion directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from v2_to_training_pipeline import V2TrainingPipeline

print("🚀 HIERARCHICAL DATA CORPUS FOR QWEN2.5VL")
print("=" * 60)
print()

# Create pipeline
pipeline = V2TrainingPipeline()

print("✅ SYSTEM CAPABILITIES:")
print("• Flexible taxonomy without hardcoded stages")
print("• Multiple coordinate formats: bbox_2d, square, line") 
print("• Comprehensive V2 question pattern coverage")
print("• Hierarchical description generation")
print("• Progressive learning dataset creation")
print()

# Demo with V2 data
input_dir = "ds_v2"
output_file = "temporal/hierarchical_training_data.jsonl"

print("📊 PROCESSING V2 DATA...")

# Process a subset for demo
v2_files = list(Path(input_dir).glob("*.json"))[:3]
all_samples = []

for v2_file in v2_files:
    samples = pipeline.processor.process_v2_file(str(v2_file))
    all_samples.extend(samples)
    print(f"  {v2_file.name}: {len(samples)} samples")

print(f"\nTotal samples processed: {len(all_samples)}")
print()

# Show format diversity
print("📐 COORDINATE FORMAT EXAMPLES:")
format_examples = {"bbox_2d": None, "square": None, "line": None}

for sample in all_samples:
    if not format_examples[sample.geometry_format]:
        format_examples[sample.geometry_format] = sample

for format_name, sample in format_examples.items():
    if sample:
        coords = sample.coordinates
        if format_name == "bbox_2d":
            print(f"• {format_name}: {coords} (4 coordinates)")
        elif format_name == "square": 
            print(f"• {format_name}: {coords} (8 coordinates)")
        elif format_name == "line":
            print(f"• {format_name}: {coords[:6]}... ({len(coords)} coordinates)")

print()

# Show hierarchical descriptions
print("📝 HIERARCHICAL DESCRIPTION EXAMPLES:")
object_examples = {}
for sample in all_samples:
    if sample.object_type not in object_examples:
        object_examples[sample.object_type] = sample

for obj_type, sample in list(object_examples.items())[:4]:
    print(f"• {obj_type.upper()}: {sample.description}")

print()

# Show training format
print("🎯 FINAL TRAINING FORMAT:")
example = all_samples[0].to_training_format()
print(json.dumps(example, ensure_ascii=False, indent=2))
print()

# Save training data
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in all_samples:
        f.write(json.dumps(sample.to_training_format(), ensure_ascii=False) + '\n')

print(f"💾 SAVED: {len(all_samples)} samples to {output_file}")
print()

# Generate statistics
stats = pipeline.processor.get_statistics(all_samples)
print("📈 DATASET STATISTICS:")
print(f"• Total samples: {stats['total_samples']}")
print(f"• Object types: {stats['object_types']}")
print(f"• Geometry formats: {stats['geometry_formats']}")
print(f"• Attribute groups: {stats['attribute_groups']}")
print()

print("🎉 HIERARCHICAL DATA CORPUS READY FOR QWEN2.5VL TRAINING!")
print("=" * 60)
print()
print("Key achievements:")
print("✓ Flexible taxonomy without hardcoded stages")  
print("✓ Multi-format coordinates (bbox_2d/square/line)")
print("✓ Hierarchical descriptions with slash separation")
print("✓ Comprehensive V2 question pattern coverage")
print("✓ Progressive learning capability")
print("✓ Clean, elegant, minimal codebase")
print()
print("The system is ready for:")
print("• Object identification training")
print("• Property recognition learning")
print("• Complex attribute assessment")
print("• OCR and special case handling")