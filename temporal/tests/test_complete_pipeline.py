#!/usr/bin/env python3
"""Test the complete V2 to training pipeline."""

import json
import sys
from pathlib import Path

# Add the data_conversion directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from v2_to_training_pipeline import V2TrainingPipeline

# Create pipeline
pipeline = V2TrainingPipeline()

# Test with a small subset of V2 data
input_dir = "ds_v2"
output_file = "temporal/test_output.jsonl"

print("Testing complete V2 to training pipeline...")

# Process just a few files for testing
v2_files = list(Path(input_dir).glob("*.json"))[:5]  # First 5 files
print(f"Found {len(v2_files)} files for testing")

if v2_files:
    # Process each file individually for detailed feedback
    all_samples = []
    for v2_file in v2_files:
        print(f"\nProcessing: {v2_file.name}")
        samples = pipeline.processor.process_v2_file(str(v2_file))
        print(f"  - Extracted {len(samples)} samples")
        
        # Show some sample details
        for i, sample in enumerate(samples[:2]):  # Show first 2 from each file
            print(f"  Sample {i+1}:")
            print(f"    Object: {sample.object_type}")
            print(f"    Geometry: {sample.geometry_format} {len(sample.coordinates)} coords")
            print(f"    Description: {sample.description[:100]}...")
            
            # Show training format
            training_data = sample.to_training_format()
            print(f"    Training keys: {list(training_data.keys())}")
        
        all_samples.extend(samples)
    
    # Save test output
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_training_format(), ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(all_samples)} samples to {output_file}")
    
    # Generate statistics
    stats = pipeline.processor.get_statistics(all_samples)
    print(f"\n=== Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Object types: {stats['object_types']}")
    print(f"Geometry formats: {stats['geometry_formats']}")
    print(f"Attribute groups: {stats['attribute_groups']}")
    
    # Validate output
    validation = pipeline.validate_output(output_file)
    print(f"\n=== Validation ===")
    print(f"Total lines: {validation['total_lines']}")
    print(f"Valid JSON lines: {validation['valid_json_lines']}")
    print(f"Geometry distribution: {validation['geometry_format_distribution']}")
    print(f"Description length stats: {validation['description_length_stats']}")
    
else:
    print("No V2 files found for testing!")

print("\nTest completed!")