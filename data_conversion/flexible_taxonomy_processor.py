#!/usr/bin/env python3
"""
Flexible Taxonomy Processor

A comprehensive system for processing V2 annotations using a flexible attribute taxonomy
without hardcoded stages. Groups information logically and creates hierarchical descriptions.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


@dataclass
class AnnotationSample:
    """Represents a processed annotation sample with hierarchical information."""
    object_type: str
    geometry_format: str  # bbox_2d, square, line
    coordinates: List[float]
    grouped_attributes: Dict[str, Dict[str, str]]  # group -> attribute -> value
    description: str
    original_geometry: Dict = None
    
    def to_training_format(self) -> Dict:
        """Convert to training format with native geometry type."""
        return {
            self.geometry_format: self.coordinates,
            "desc": self.description
        }


class FlexibleTaxonomyProcessor:
    """Process V2 annotations using flexible attribute taxonomy."""
    
    def __init__(self, taxonomy_path: Optional[str] = None):
        if taxonomy_path is None:
            taxonomy_path = Path(__file__).parent / "attribute_taxonomy.json"
        
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.object_types = self.taxonomy["object_types"]
        self.attribute_groups = self.taxonomy["attribute_groups"]
        self.geometry_mapping = self.taxonomy["geometry_format_mapping"]
        
        logger.info(f"Loaded taxonomy with {len(self.object_types)} object types and {len(self.attribute_groups)} attribute groups")
    
    def _load_taxonomy(self, taxonomy_path: str) -> Dict:
        """Load the attribute taxonomy."""
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_v2_feature(self, feature: Dict) -> Optional[AnnotationSample]:
        """
        Process a single V2 feature into structured annotation sample.
        
        Args:
            feature: V2 feature with geometry and properties
            
        Returns:
            AnnotationSample or None if processing fails
        """
        properties = feature.get("properties", {})
        content_zh = properties.get("contentZh", {})
        content = properties.get("content", {})
        geometry = feature.get("geometry", {})
        
        # Determine object type
        object_type = self._determine_object_type(content, content_zh)
        if not object_type:
            logger.warning("Could not determine object type")
            return None
        
        # Process geometry
        geometry_format, coordinates = self._process_geometry(geometry, object_type)
        
        # Group attributes by taxonomy
        grouped_attributes = self._group_attributes(content, content_zh, object_type)
        
        # Create hierarchical description
        description = self._create_description(object_type, grouped_attributes)
        
        return AnnotationSample(
            object_type=object_type,
            geometry_format=geometry_format,
            coordinates=coordinates,
            grouped_attributes=grouped_attributes,
            description=description,
            original_geometry=geometry
        )
    
    def _determine_object_type(self, content: Dict, content_zh: Dict) -> Optional[str]:
        """Determine object type from content fields."""
        # Check content.label first
        content_label = content.get("label", "").strip()
        for obj_type, obj_info in self.object_types.items():
            if content_label == obj_info["content_key"]:
                return obj_type
        
        # Check contentZh.标签 as fallback
        zh_label = content_zh.get("标签", "").strip()
        for obj_type, obj_info in self.object_types.items():
            if zh_label == obj_info["chinese_label"] or zh_label in obj_info.get("aliases", []):
                return obj_type
        
        return None
    
    def _process_geometry(self, geometry: Dict, object_type: str) -> Tuple[str, List[float]]:
        """Process geometry based on object type and geometry structure."""
        from data_conversion.coordinate_manager import CoordinateManager
        
        geometry_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])
        
        # Determine target format based on object type preferences
        obj_info = self.object_types[object_type]
        preferred_formats = obj_info["geometry_types"]
        
        # Extract bbox for all cases
        bbox = CoordinateManager.extract_bbox_from_geometry(geometry)
        
        # Handle geometry types based on their native format
        if geometry_type == "LineString":
            # Extract line coordinates
            line_coords = []
            for coord in coordinates:
                if isinstance(coord, list) and len(coord) >= 2:
                    line_coords.extend([int(round(coord[0])), int(round(coord[1]))])
            return "line", line_coords
        
        elif geometry_type == "Square":
            # Extract square coordinates (first 4 points)
            if coordinates and isinstance(coordinates[0], list):
                points = coordinates[0]
                square_coords = []
                for i, point in enumerate(points[:4]):
                    if isinstance(point, list) and len(point) >= 2:
                        square_coords.extend([int(round(point[0])), int(round(point[1]))])
                
                if len(square_coords) == 8:  # Valid square
                    return "square", square_coords
        
        # ExtentPolygon -> bbox_2d
        return "bbox_2d", bbox
    
    def _group_attributes(self, content: Dict, content_zh: Dict, object_type: str) -> Dict[str, Dict[str, str]]:
        """Group attributes according to taxonomy."""
        grouped = {}
        
        for group_name, group_info in self.attribute_groups.items():
            group_attributes = {}
            
            for attr_name, attr_info in group_info["attributes"].items():
                # Check if attribute applies to this object type
                applies_to = attr_info.get("applies_to", [])
                if applies_to != "all" and object_type not in applies_to:
                    continue
                
                # Extract attribute value
                value = self._extract_attribute_value(content, content_zh, attr_info, object_type)
                if value:
                    group_attributes[attr_name] = value
            
            if group_attributes:
                grouped[group_name] = group_attributes
        
        return grouped
    
    def _extract_attribute_value(self, content: Dict, content_zh: Dict, attr_info: Dict, object_type: str) -> Optional[str]:
        """Extract attribute value from contentZh (Chinese) fields only."""
        # For Chinese mode, extract directly from contentZh using Chinese questions
        chinese_questions = attr_info.get("chinese_questions", [])
        
        # Look for matching Chinese question in contentZh
        for question in chinese_questions:
            if question in content_zh:
                raw_value = content_zh[question]
                
                # Handle array values (like connect_point_check)  
                if isinstance(raw_value, list) and raw_value:
                    raw_value = raw_value[0]
                
                # Return the Chinese value directly
                if raw_value and str(raw_value).strip():
                    return str(raw_value).strip()
        
        # Fallback: check for free text fields in contentZh
        values_mapping = attr_info.get("values")
        if values_mapping == "free_text":
            content_mapping = attr_info.get("content_mapping")
            if content_mapping and content_mapping in content_zh:
                return str(content_zh[content_mapping]).strip()
        
        return None
    
    def _create_description(self, object_type: str, grouped_attributes: Dict[str, Dict[str, str]]) -> str:
        """Create hierarchical description from grouped attributes."""
        description_parts = []
        
        # Start with object type
        obj_info = self.object_types[object_type]
        description_parts.append(obj_info["chinese_label"])
        
        # Add attributes in logical order
        group_order = [
            "physical_properties", 
            "component_classification",
            "functional_assessment", 
            "environmental_context",
            "technical_specifications",
            "textual_content",
            "special_circumstances"
        ]
        
        for group_name in group_order:
            if group_name in grouped_attributes:
                group_attrs = grouped_attributes[group_name]
                
                # Create group description
                attr_parts = []
                for attr_name, attr_value in group_attrs.items():
                    if attr_value and attr_value.strip():
                        attr_parts.append(attr_value.strip())
                
                if attr_parts:
                    group_desc = "，".join(attr_parts)
                    description_parts.append(group_desc)
        
        return "/".join(description_parts)
    
    def process_v2_file(self, file_path: str) -> List[AnnotationSample]:
        """Process entire V2 JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get("markResult", {}).get("features", [])
        samples = []
        
        for feature in features:
            sample = self.process_v2_feature(feature)
            if sample:
                samples.append(sample)
        
        logger.info(f"Processed {len(samples)} samples from {file_path}")
        return samples
    
    def batch_process(self, input_dir: str, output_file: str):
        """Batch process V2 files and save training format."""
        input_path = Path(input_dir)
        all_samples = []
        
        # Process all JSON files
        for json_file in input_path.glob("*.json"):
            samples = self.process_v2_file(str(json_file))
            all_samples.extend(samples)
        
        # Convert to training format and save
        training_data = []
        for sample in all_samples:
            training_data.append(sample.to_training_format())
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(training_data)} samples to {output_file}")
        return len(training_data)
    
    def get_statistics(self, samples: List[AnnotationSample]) -> Dict[str, Any]:
        """Generate statistics about processed samples."""
        stats = {
            "total_samples": len(samples),
            "object_types": {},
            "geometry_formats": {},
            "attribute_groups": {}
        }
        
        for sample in samples:
            # Object type counts
            obj_type = sample.object_type
            stats["object_types"][obj_type] = stats["object_types"].get(obj_type, 0) + 1
            
            # Geometry format counts
            geo_format = sample.geometry_format
            stats["geometry_formats"][geo_format] = stats["geometry_formats"].get(geo_format, 0) + 1
            
            # Attribute group counts
            for group_name in sample.grouped_attributes:
                stats["attribute_groups"][group_name] = stats["attribute_groups"].get(group_name, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Example usage
    processor = FlexibleTaxonomyProcessor()
    
    # Test single file
    test_file = "/data3/Qwen2.5-VL-main/ds_v2/QC-20230216-0000244_377872.json"
    samples = processor.process_v2_file(test_file)
    
    print(f"Processed {len(samples)} samples")
    for sample in samples[:3]:  # Show first 3
        print(f"Object: {sample.object_type}")
        print(f"Format: {sample.geometry_format}")
        print(f"Description: {sample.description}")
        print(f"Attributes: {sample.grouped_attributes}")
        print("---")