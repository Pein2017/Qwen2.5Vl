#!/usr/bin/env python3
"""
File Operations Utilities

Handles all file operations including path management, JSON loading,
image discovery, and directory operations.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class FileOperations:
    """Centralized file operations for the data conversion pipeline."""
    
    @staticmethod
    def find_json_files(directory: Path) -> List[Path]:
        """Find all JSON files in a directory."""
        json_files = list(directory.rglob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")
        
        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        return sorted(json_files)
    
    @staticmethod
    def find_image_file(json_path: Path) -> Path:
        """Find corresponding image file for a JSON file."""
        for ext in [".jpeg", ".jpg"]:
            image_path = json_path.with_suffix(ext)
            if image_path.is_file():
                return image_path
        
        raise FileNotFoundError(f"No image file found for {json_path}")
    
    @staticmethod
    def load_json_data(json_path: Path) -> Dict:
        """Load and validate JSON data structure."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Basic structure validation
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
    
    @staticmethod
    def save_json_data(data: Dict, json_path: Path, indent: Optional[int] = None) -> None:
        """Save JSON data to file."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        logger.debug(f"Saved JSON to {json_path}")
    
    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions with EXIF orientation handling."""
        try:
            from PIL import ImageOps
            
            with Image.open(image_path) as img:
                # Apply EXIF orientation transformation
                img = ImageOps.exif_transpose(img)
                width, height = img.size
            
            return width, height
            
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            raise
    
    @staticmethod
    def validate_image_dimensions(
        image_path: Path, 
        expected_width: int, 
        expected_height: int
    ) -> Tuple[int, int]:
        """Validate image dimensions match expected values."""
        actual_width, actual_height = FileOperations.get_image_dimensions(image_path)
        
        if expected_width != actual_width or expected_height != actual_height:
            raise ValueError(
                f"Dimension mismatch for {image_path.name}: "
                f"Expected {expected_width}x{expected_height} "
                f"but got {actual_width}x{actual_height}"
            )
        
        return actual_width, actual_height
    
    @staticmethod
    def copy_file(src: Path, dst: Path, preserve_structure: bool = True) -> None:
        """Copy file with optional structure preservation."""
        if preserve_structure:
            dst.parent.mkdir(parents=True, exist_ok=True)
        
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.debug(f"Copied {src.name} to {dst}")
    
    @staticmethod
    def write_jsonl(samples: List[Dict], output_path: Path) -> None:
        """Write samples to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Written {len(samples)} samples to {output_path}")
    
    @staticmethod
    def load_token_map(token_map_path: Path) -> Dict[str, str]:
        """Load token mapping from JSON file."""
        if not token_map_path.exists():
            raise FileNotFoundError(f"Token map file not found: {token_map_path}")
        
        with open(token_map_path, "r", encoding="utf-8") as f:
            token_map = json.load(f)
        
        logger.info(f"Loaded {len(token_map)} token mappings from {token_map_path}")
        return token_map
    
    @staticmethod
    def load_label_hierarchy(hierarchy_path: Path) -> Dict[str, List[str]]:
        """Load label hierarchy from JSON file."""
        if not hierarchy_path.exists():
            logger.warning(f"Label hierarchy file not found: {hierarchy_path}")
            return {}
        
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            raw_hierarchy = json.load(f)
        
        # Normalize hierarchy format
        if isinstance(raw_hierarchy, list):
            hierarchy = {entry["object_type"]: entry.get("property", []) for entry in raw_hierarchy}
        elif isinstance(raw_hierarchy, dict):
            if all(isinstance(v, list) for v in raw_hierarchy.values()):
                hierarchy = raw_hierarchy
            else:
                hierarchy = {k: v.get("property", []) for k, v in raw_hierarchy.items()}
        else:
            logger.warning("Invalid hierarchy format, using empty hierarchy")
            hierarchy = {}
        
        logger.info(f"Loaded hierarchy with {len(hierarchy)} object types")
        return hierarchy
    
    @staticmethod
    def clean_directory(directory: Path, keep_patterns: Optional[List[str]] = None) -> None:
        """Clean directory keeping only specified patterns."""
        if not directory.exists():
            return
        
        keep_patterns = keep_patterns or []
        
        for item in directory.iterdir():
            should_keep = any(item.match(pattern) for pattern in keep_patterns)
            if not should_keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                logger.debug(f"Removed {item}")
    
    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Path:
        """Create backup of a file."""
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        return backup_path
    
    @staticmethod
    def calculate_relative_path(file_path: Path, base_dir: Path) -> str:
        """Calculate relative path from base directory."""
        try:
            return str(file_path.relative_to(base_dir))
        except ValueError:
            # If relative path calculation fails, return absolute path
            return str(file_path)