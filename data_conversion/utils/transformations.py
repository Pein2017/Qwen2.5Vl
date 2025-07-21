#!/usr/bin/env python3
"""
Data Transformation Utilities

Handles token mapping, coordinate transformations, and data format conversions.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Union

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)




class CoordinateTransformer:
    """Handles coordinate transformations and scaling operations."""
    
    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor
    
    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor
    
    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer <= 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor
    
    @staticmethod
    def smart_resize(
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        max_ratio: int = 200
    ) -> tuple[int, int]:
        """
        Calculate optimal resize dimensions maintaining aspect ratio.
        
        Conditions:
        1. Both dimensions divisible by 'factor'
        2. Total pixels within [min_pixels, max_pixels]
        3. Aspect ratio maintained as closely as possible
        """
        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"Aspect ratio must be < {max_ratio}, got {max(height, width) / min(height, width)}"
            )
        
        h_bar = max(factor, CoordinateTransformer.round_by_factor(height, factor))
        w_bar = max(factor, CoordinateTransformer.round_by_factor(width, factor))
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, CoordinateTransformer.floor_by_factor(height / beta, factor))
            w_bar = max(factor, CoordinateTransformer.floor_by_factor(width / beta, factor))
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = CoordinateTransformer.ceil_by_factor(height * beta, factor)
            w_bar = CoordinateTransformer.ceil_by_factor(width * beta, factor)
        
        return h_bar, w_bar
    
    @staticmethod
    def scale_bbox(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int
    ) -> List[int]:
        """
        Scale bounding box coordinates from original to new dimensions.
        
        Fail-fast: validates both input and output bboxes.
        Returns clean integer coordinates for VLM training.
        """
        from .validators import DataValidator
        
        # Clean input bbox by rounding to eliminate floating-point errors
        clean_bbox = [round(coord) for coord in bbox]
        
        # Validate original bbox
        DataValidator.validate_bbox(clean_bbox, original_width, original_height)
        
        x_scale = new_width / original_width
        y_scale = new_height / original_height
        
        x_min, y_min, x_max, y_max = clean_bbox
        
        # Scale and round to integer pixel coordinates
        scaled_x1 = int(round(x_min * x_scale))
        scaled_y1 = int(round(y_min * y_scale))
        scaled_x2 = int(round(x_max * x_scale))
        scaled_y2 = int(round(y_max * y_scale))
        
        # Ensure correct ordering
        new_x_min = min(scaled_x1, scaled_x2)
        new_y_min = min(scaled_y1, scaled_y2)
        new_x_max = max(scaled_x1, scaled_x2)
        new_y_max = max(scaled_y1, scaled_y2)
        
        final_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
        
        # Validate scaled bbox
        DataValidator.validate_bbox(final_bbox, new_width, new_height)
        
        return final_bbox
    
    @staticmethod
    def extract_bbox_from_coordinates(coordinates: List[List[float]]) -> List[float]:
        """Extract bounding box from polygon coordinates."""
        if not coordinates or not all(len(p) == 2 for p in coordinates):
            raise ValueError(f"Invalid coordinates structure: {coordinates}")
        
        x_coords = [p[0] for p in coordinates]
        y_coords = [p[1] for p in coordinates]
        
        # Return clean integer coordinates
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return [round(coord) for coord in bbox]
    
    @staticmethod
    def clean_bbox_coordinates(bbox: List[float]) -> List[int]:
        """
        Clean bbox coordinates for VLM training.
        
        Rounds to integers to eliminate floating-point precision errors
        that make JSON unnecessarily complex for model learning.
        """
        return [int(round(coord)) for coord in bbox]
    
    @staticmethod
    def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
        """Normalize bbox coordinates to [0, 1] range."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        x_min, y_min, x_max, y_max = bbox
        
        return [
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height
        ]
    
    @staticmethod
    def denormalize_bbox(normalized_bbox: List[float], width: int, height: int) -> List[float]:
        """Convert normalized bbox back to pixel coordinates."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        x_min, y_min, x_max, y_max = normalized_bbox
        
        return [
            x_min * width,
            y_min * height,
            x_max * width,
            y_max * height
        ]


class FormatConverter:
    """Handles conversion between different data formats."""
    
    @staticmethod
    def format_description(content_dict: Dict[str, str], response_types: List[str], language: str = "chinese") -> str:
        """Format content dictionary to Chinese description string."""
        parts = []
        for resp_type in response_types:
            value = content_dict.get(resp_type, "")
            if value:
                parts.append(value)
        
        if not parts:
            return ""
        
        # Use compact format for Chinese
        result = "/".join(parts)
        return result.replace(", ", "/").replace(",", "/")
    
    @staticmethod
    def parse_description_string(description: str) -> Dict[str, str]:
        """Parse Chinese description string back into components."""
        components = {"object_type": "", "property": "", "extra_info": ""}
        
        if not description:
            return components
        
        # Parse compact format (Chinese)
        parts = description.split("/")
        if len(parts) >= 1:
            components["object_type"] = parts[0].strip()
        if len(parts) >= 2:
            components["property"] = parts[1].strip()
        if len(parts) >= 3:
            components["extra_info"] = "/".join(parts[2:]).strip()
        
        return components
    
    @staticmethod
    def clean_annotation_content(data: Dict) -> Dict:
        """Clean annotation content preserving essential Chinese structure only."""
        cleaned_data = {}
        
        # Preserve essential metadata
        essential_keys = ["info", "tagInfo", "version"]
        for key in essential_keys:
            if key in data:
                cleaned_data[key] = data[key]
        
        # Clean features in markResult - Chinese only
        if "markResult" in data and "features" in data["markResult"]:
            cleaned_features = []
            
            for feature in data["markResult"]["features"]:
                properties = {}
                original_properties = feature.get("properties", {})
                
                # Keep only Chinese content
                properties["contentZh"] = original_properties.get("contentZh", {})
                
                cleaned_features.append({
                    "type": feature.get("type", "Feature"),
                    "geometry": feature.get("geometry", {}),
                    "properties": properties
                })
            
            cleaned_data["markResult"] = {
                "features": cleaned_features,
                "type": data["markResult"].get("type", "FeatureCollection")
            }
            
            # Preserve other markResult fields
            for key in data["markResult"]:
                if key not in ["features", "type"]:
                    cleaned_data["markResult"][key] = data["markResult"][key]
        
        return cleaned_data