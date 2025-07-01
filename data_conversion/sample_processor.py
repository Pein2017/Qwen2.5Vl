#!/usr/bin/env python3
"""
Sample Processor for Individual Sample Processing

Handles processing of individual samples: extracting objects, applying token mapping,
filtering by hierarchy, image processing, and bbox scaling.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image

from core_modules import TokenMapper, ObjectProcessor
from vision_process import smart_resize

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class SampleProcessor:
    """Processes individual samples from JSON data to clean format."""
    
    def __init__(
        self,
        language: str,
        response_types: Set[str],
        label_hierarchy: Dict[str, List[str]],
        token_mapper: Optional[TokenMapper] = None,
        resize_enabled: bool = False,
        output_image_dir: Optional[Path] = None,
        input_dir: Optional[Path] = None
    ):
        self.language = language
        self.response_types = response_types
        self.label_hierarchy = label_hierarchy
        self.token_mapper = token_mapper
        self.resize_enabled = resize_enabled
        self.output_image_dir = Path(output_image_dir) if output_image_dir else None
        self.input_dir = Path(input_dir) if input_dir else None
        
        # Validate required parameters
        if self.resize_enabled and not self.output_image_dir:
            raise ValueError("output_image_dir required when resize_enabled=True")
    
    def _extract_content_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract and normalize content fields based on language."""
        if self.language == "chinese":
            return self._extract_chinese_fields(source_dict)
        else:
            return self._extract_english_fields(source_dict)
    
    def _extract_chinese_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract fields from Chinese contentZh format."""
        content_zh = source_dict.get("contentZh", {})
        if not content_zh:
            return {}
        
        # Extract label entries containing '标签'
        label_values = []
        for key, value in content_zh.items():
            if "标签" in key:
                if isinstance(value, list):
                    label_values.append(", ".join(map(str, value)))
                elif value:
                    label_values.append(str(value))
        
        if not label_values:
            return {}
        
        # Parse first label entry: "object_type/property/extra"
        label_string = label_values[0]
        parts = [p.strip() for p in label_string.split("/")]
        object_type = parts[0] if len(parts) >= 1 else ""
        property_value = parts[1] if len(parts) >= 2 else ""
        existing_extras = parts[2:] if len(parts) >= 3 else []
        
        # Collect additional extra_info from other contentZh entries
        additional_extras = []
        for key, value in content_zh.items():
            if "标签" not in key:
                if isinstance(value, list):
                    additional_extras.extend(str(item) for item in value if item)
                elif value:
                    additional_extras.append(str(value))
        
        extra_info = "/".join(existing_extras + additional_extras)
        
        return {
            "object_type": self._apply_token_mapping(object_type),
            "property": self._apply_token_mapping(property_value),
            "extra_info": self._apply_token_mapping(extra_info)
        }
    
    def _extract_english_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract fields from English format with fallback field names."""
        return {
            "object_type": self._apply_token_mapping(
                source_dict.get("object_type") or source_dict.get("label", "")
            ),
            "property": self._apply_token_mapping(
                source_dict.get("property") or source_dict.get("question", "")
            ),
            "extra_info": self._apply_token_mapping(
                source_dict.get("extra_info") or source_dict.get("question_ex", "")
            )
        }
    
    def _apply_token_mapping(self, token: str) -> str:
        """Apply token mapping if available."""
        if self.token_mapper and token:
            return self.token_mapper.map_token(token)
        return token
    
    def _format_description(self, content_dict: Dict[str, str]) -> str:
        """Format content dict to description string based on response types."""
        parts = []
        for resp_type in self.response_types:
            value = content_dict.get(resp_type, "")
            if value:
                parts.append(value)
        
        if not parts:
            return ""
        
        # Use compact format for Chinese, structured format for English
        if self.language == "chinese":
            result = "/".join(parts)
            # Normalize separators
            return result.replace(", ", "/").replace(",", "/")
        else:
            # English: structured format with explicit labels
            formatted_parts = []
            for resp_type in self.response_types:
                value = content_dict.get(resp_type, "")
                formatted_parts.append(f"{resp_type}:{value if value else 'none'}")
            return ";".join(formatted_parts)
    
    def _is_allowed_object(self, content_dict: Dict[str, str]) -> bool:
        """Check if object passes label hierarchy filtering."""
        obj_type = content_dict.get("object_type", "")
        prop = content_dict.get("property", "")
        
        # Skip if object_type not in hierarchy
        if obj_type not in self.label_hierarchy:
            return False
        
        allowed_props = self.label_hierarchy.get(obj_type, [])
        
        # If no properties allowed, only accept empty property
        if not allowed_props:
            return prop == "" or prop is None
        
        # Check if property is directly allowed
        if prop in allowed_props:
            return True
        
        # Allow variant "obj_type/property" format stored in hierarchy
        combo = f"{obj_type}/{prop}" if prop else obj_type
        return combo in allowed_props
    
    def _extract_objects_from_datalist(self, data_list: List[Dict]) -> List[Dict]:
        """Extract objects from dataList format."""
        objects = []
        
        for item in data_list:
            coords = item.get("coordinates", [])
            if len(coords) < 2:
                logger.warning(f"Invalid coordinates in dataList item: {coords}")
                continue
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            
            properties = item.get("properties", {}) or {}
            content_dict = self._extract_content_fields(properties)
            
            if not content_dict or not self._is_allowed_object(content_dict):
                continue
            
            desc = self._format_description(content_dict)
            if desc:
                objects.append({"bbox_2d": bbox, "desc": desc})
        
        return objects
    
    def _extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """Extract objects from markResult features format."""
        objects = []
        
        for feature in features:
            geometry = feature.get("geometry", {})
            coords = geometry.get("coordinates", [])
            
            if not coords or not isinstance(coords, list) or len(coords) == 0:
                logger.warning(f"Invalid coordinates in markResult feature: {coords}")
                continue
            
            # Handle nested coordinate structures: [[x,y], ...] or [[[x,y], ...]]
            points = coords
            if (points and isinstance(points[0], list) and 
                points[0] and isinstance(points[0][0], list)):
                points = points[0]
            
            if not points or any(len(p) != 2 for p in points):
                logger.warning(f"Invalid points structure: {points}")
                continue
            
            # Extract bounding box from polygon points
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            properties = feature.get("properties", {})
            content_dict = self._extract_content_fields(properties)
            
            if not content_dict or not self._is_allowed_object(content_dict):
                continue
            
            desc = self._format_description(content_dict)
            if desc:
                objects.append({"bbox_2d": bbox, "desc": desc})
        
        return objects
    
    def _process_image_file(self, image_path: Path, width: int, height: int) -> Path:
        """Process image file (copy or resize) and return output path."""
        if not self.output_image_dir or not self.input_dir:
            # If no output directory specified, return original path
            return image_path
        
        # Calculate relative path and output location
        rel_path = image_path.relative_to(self.input_dir)
        output_path = self.output_image_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.resize_enabled:
            # Resize image using smart_resize
            new_height, new_width = smart_resize(height=height, width=width)
            with Image.open(image_path) as img:
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_img.save(output_path)
            logger.debug(f"Resized {image_path.name} to {new_width}x{new_height}")
        else:
            # Simple copy
            if not output_path.exists():
                shutil.copy(image_path, output_path)
            logger.debug(f"Copied {image_path.name}")
        
        return output_path
    
    def _scale_bboxes(self, objects: List[Dict], original_width: int, original_height: int, 
                     new_width: int, new_height: int) -> None:
        """Scale bounding boxes in-place for resized images."""
        for obj in objects:
            bbox = obj["bbox_2d"]
            try:
                scaled_bbox = ObjectProcessor.scale_bbox(
                    bbox, original_width, original_height, new_width, new_height
                )
                obj["bbox_2d"] = scaled_bbox
            except ValueError as e:
                logger.error(f"Error scaling bbox {bbox}: {e}")
                raise
    
    def process_sample(
        self, 
        json_data: Dict, 
        image_path: Path, 
        width: int, 
        height: int,
        output_base_dir: Optional[Path] = None
    ) -> Optional[Dict]:
        """Process a single sample from JSON data and image."""
        
        # Extract objects from JSON data
        objects = []
        
        if "dataList" in json_data:
            objects = self._extract_objects_from_datalist(json_data["dataList"])
        elif "markResult" in json_data and isinstance(json_data.get("markResult", {}).get("features"), list):
            objects = self._extract_objects_from_markresult(json_data["markResult"]["features"])
        else:
            logger.warning(f"No valid annotation format found in sample")
            return None
        
        if not objects:
            logger.debug(f"No valid objects found in sample {image_path.name}")
            return None
        
        # Sort objects by position (top-left to bottom-right)
        objects.sort(key=lambda obj: (obj["bbox_2d"][1], obj["bbox_2d"][0]))
        
        # Process image file
        processed_image_path = self._process_image_file(image_path, width, height)
        
        # Handle image resizing and bbox scaling
        final_width, final_height = width, height
        if self.resize_enabled:
            final_height, final_width = smart_resize(height=height, width=width)
            self._scale_bboxes(objects, width, height, final_width, final_height)
        
        # Build relative image path for JSONL
        if output_base_dir and self.output_image_dir:
            try:
                rel_image_path = str(processed_image_path.relative_to(output_base_dir.parent))
            except ValueError:
                rel_image_path = str(processed_image_path)
        else:
            rel_image_path = str(processed_image_path)
        
        return {
            "images": [rel_image_path],
            "objects": objects,
            "width": final_width,
            "height": final_height
        }