#!/usr/bin/env python3
"""Debug test to check the processor functionality."""

import json
import sys
from pathlib import Path

# Add the data_conversion directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from flexible_taxonomy_processor import FlexibleTaxonomyProcessor

# Create processor
processor = FlexibleTaxonomyProcessor()

# Simple test data
test_feature = {
    "type": "Feature",
    "geometry": {
        "coordinates": [
            [955.8214285714286, 206.20238095238096],
            [997.4880952380953, 206.20238095238096],
            [997.4880952380953, 254.53571428571445],
            [955.8214285714286, 254.53571428571445],
            [955.8214285714286, 206.20238095238096]
        ],
        "type": "ExtentPolygon"
    },
    "properties": {
        "contentZh": {
            "标签": "螺丝、光纤插头",
            "螺丝/插头是否显示完整": "只显示部分",
            "这个螺丝/插头是什么种类": "BBU安装螺丝",
            "这个螺丝/插头连接是否符合要求": ["符合要求"]
        },
        "content": {
            "connect_point_situation": "connect_point_situation_part",
            "connect_point_type": "install_screw",
            "connect_point_check": ["connect_point_check_true"],
            "label": "connect_point",
            "ex_info": ""
        }
    }
}

print("Testing flexible taxonomy processor...")

# Test object type detection
content = test_feature["properties"]["content"]
content_zh = test_feature["properties"]["contentZh"]

print(f"Content label: {content.get('label')}")
print(f"ContentZh label: {content_zh.get('标签')}")

# Test object type determination
obj_type = processor._determine_object_type(content, content_zh)
print(f"Detected object type: {obj_type}")

# Test full processing
sample = processor.process_v2_feature(test_feature)
print(f"Processed sample: {sample}")

if sample:
    print(f"Object type: {sample.object_type}")
    print(f"Geometry format: {sample.geometry_format}")
    print(f"Coordinates: {sample.coordinates}")
    print(f"Grouped attributes: {sample.grouped_attributes}")
    print(f"Description: {sample.description}")
else:
    print("Sample processing failed!")