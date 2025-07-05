#!/usr/bin/env python3
"""
Data Loader for Raw JSON and Image Files

Handles loading and basic validation of raw JSON annotation files
and their corresponding image files from the input directory.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class SampleLoader:
    """Loads and validates raw JSON/image file pairs."""
    
    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
    
    def find_json_files(self) -> List[Path]:
        """Find all JSON files in the input directory."""
        json_files = list(self.input_dir.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.input_dir}")
        
        logger.info(f"Found {len(json_files)} JSON files")
        return sorted(json_files)
    
    def load_json_data(self, json_path: Path) -> Dict:
        """Load and validate JSON annotation data."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate required structure
            info = data.get("info", {})
            if "height" not in info or "width" not in info:
                raise ValueError(f"Missing height/width in info section: {json_path}")
            
            # Check for annotation data
            has_data_list = "dataList" in data and isinstance(data["dataList"], list)
            has_mark_result = (
                "markResult" in data and 
                isinstance(data.get("markResult", {}).get("features"), list)
            )
            
            if not (has_data_list or has_mark_result):
                raise ValueError(f"No valid annotation data found: {json_path}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            raise
    
    def find_image_file(self, json_path: Path) -> Path:
        """Find corresponding image file for a JSON file."""
        for ext in [".jpeg", ".jpg"]:
            image_path = json_path.with_suffix(ext)
            if image_path.is_file():
                return image_path
        
        raise FileNotFoundError(f"No image file found for {json_path}")
    
    def validate_image_dimensions(self, image_path: Path, expected_width: int, expected_height: int) -> Tuple[int, int]:
        """Validate image dimensions match JSON metadata."""
        try:
            from PIL import ImageOps
            
            with Image.open(image_path) as img:
                # CRITICAL FIX: Apply EXIF orientation transformation
                # This ensures we get dimensions that match the annotation space
                img = ImageOps.exif_transpose(img)
                actual_width, actual_height = img.size
            
            if expected_width != actual_width or expected_height != actual_height:
                raise ValueError(
                    f"Dimension mismatch for {image_path.name}: "
                    f"JSON reports {expected_width}x{expected_height} "
                    f"but image is {actual_width}x{actual_height}"
                )
            
            return actual_width, actual_height
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {e}")
            raise
    
    def load_sample_pair(self, json_path: Path) -> Tuple[Dict, Path, int, int]:
        """Load and validate a JSON/image pair."""
        # Load JSON data
        json_data = self.load_json_data(json_path)
        
        # Find corresponding image
        image_path = self.find_image_file(json_path)
        
        # Extract expected dimensions
        info = json_data["info"]
        expected_width = info["width"]
        expected_height = info["height"]
        
        # Validate image dimensions
        actual_width, actual_height = self.validate_image_dimensions(
            image_path, expected_width, expected_height
        )
        
        logger.debug(f"Loaded sample pair: {json_path.name} -> {image_path.name}")
        
        return json_data, image_path, actual_width, actual_height