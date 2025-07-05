#!/usr/bin/env python3
"""
Unified Image Processing Module

Consolidates all image processing functionality including EXIF handling,
smart resizing, coordinate scaling, and path management.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from config import DataConversionConfig
from utils.file_ops import FileOperations
from utils.transformations import CoordinateTransformer
from utils.validators import DataValidator

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Unified image processing for the data conversion pipeline."""
    
    def __init__(self, config: DataConversionConfig):
        """Initialize with configuration."""
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.output_image_dir = Path(config.output_image_dir) if config.output_image_dir else None
        
        logger.info(f"ImageProcessor initialized: resize={config.resize_enabled}")
    
    def to_rgb(self, pil_image: Image.Image) -> Image.Image:
        """
        Convert PIL image to RGB with proper EXIF orientation handling.
        
        Applies EXIF orientation transformation to ensure image display
        matches annotation space, then converts to RGB with white background
        for transparency handling.
        """
        # Apply EXIF orientation transformation
        pil_image = ImageOps.exif_transpose(pil_image)
        
        if pil_image.mode == "RGBA":
            white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])
            return white_background
        
        return pil_image.convert("RGB")
    
    def process_image(
        self, 
        image_path: Path, 
        width: int, 
        height: int, 
        output_base_dir: Optional[Path] = None
    ) -> Tuple[Path, int, int]:
        """
        Process a single image: copy or resize with coordinate scaling.
        
        Returns:
            Tuple of (output_image_path, final_width, final_height)
        """
        if not self.output_image_dir:
            # No processing needed, return original
            return image_path, width, height
        
        # Calculate output path
        try:
            rel_path = image_path.relative_to(self.input_dir)
        except ValueError:
            # image_path is not relative to input_dir, it might already be in output_dir
            rel_path = image_path.name
        
        output_path = self.output_image_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if image already exists and has been processed
        if output_path.exists() and output_path != image_path:
            existing_width, existing_height = FileOperations.get_image_dimensions(output_path)
            logger.debug(f"Using existing processed image: {output_path} ({existing_width}x{existing_height})")
            return output_path, existing_width, existing_height
        
        if self.config.resize_enabled:
            # Smart resize
            new_height, new_width = CoordinateTransformer.smart_resize(
                height=height, width=width
            )
            
            with Image.open(image_path) as img:
                # Apply EXIF orientation and convert to RGB
                processed_img = self.to_rgb(img)
                
                # Resize image
                resized_img = processed_img.resize(
                    (new_width, new_height), 
                    Image.Resampling.LANCZOS
                )
                resized_img.save(output_path)
            
            logger.debug(f"Resized {image_path.name}: {width}x{height} → {new_width}x{new_height}")
            return output_path, new_width, new_height
        
        else:
            # Copy with EXIF orientation handling
            if not output_path.exists():
                with Image.open(image_path) as img:
                    processed_img = self.to_rgb(img)
                    processed_img.save(output_path)
            
            logger.debug(f"Copied {image_path.name} with EXIF orientation applied")
            return output_path, width, height
    
    def scale_object_coordinates(
        self, 
        objects: List[Dict], 
        original_width: int, 
        original_height: int, 
        new_width: int, 
        new_height: int
    ) -> None:
        """Scale bounding box coordinates in-place for resized images."""
        if original_width == new_width and original_height == new_height:
            return  # No scaling needed
        
        for obj in objects:
            bbox = obj["bbox_2d"]
            try:
                scaled_bbox = CoordinateTransformer.scale_bbox(
                    bbox, original_width, original_height, new_width, new_height
                )
                obj["bbox_2d"] = scaled_bbox
            except ValueError as e:
                logger.error(f"Error scaling bbox {bbox}: {e}")
                if self.config.fail_fast:
                    raise
    
    def update_output_coordinates(self) -> bool:
        """
        Update coordinates in output JSON files to match resized images.
        
        This fixes coordinate-image mismatches when images are resized
        but JSON coordinates remain at original scale.
        """
        if not self.config.resize_enabled or not self.output_image_dir:
            logger.info("Coordinate update skipped: resize not enabled")
            return True
        
        logger.info("Updating output coordinates to match resized images")
        
        json_files = list(self.output_image_dir.glob("*.json"))
        updated_count = 0
        errors = []
        
        for json_file in json_files:
            try:
                sample_id = json_file.stem
                
                # Find corresponding images
                original_image = self._find_original_image(sample_id)
                output_image = self._find_output_image(sample_id)
                
                if not original_image or not output_image:
                    continue
                
                # Get image dimensions
                original_size = FileOperations.get_image_dimensions(original_image)
                output_size = FileOperations.get_image_dimensions(output_image)
                
                if original_size == output_size:
                    continue  # No resize occurred
                
                # Calculate scale factors
                scale_x = output_size[0] / original_size[0]
                scale_y = output_size[1] / original_size[1]
                
                logger.debug(
                    f"Sample {sample_id}: {original_size} → {output_size} "
                    f"(scale: {scale_x:.4f}, {scale_y:.4f})"
                )
                
                # Load and update JSON
                data = FileOperations.load_json_data(json_file)
                
                # Update image dimensions in info
                if "info" in data:
                    data["info"]["width"] = output_size[0]
                    data["info"]["height"] = output_size[1]
                
                # Check if scaling is needed
                if self._needs_coordinate_scaling(data, output_size):
                    # Backup original file
                    FileOperations.backup_file(json_file)
                    
                    # Scale coordinates
                    features_updated = self._scale_json_coordinates(data, scale_x, scale_y)
                    
                    # Save updated data
                    FileOperations.save_json_data(data, json_file)
                    
                    logger.debug(f"Updated {features_updated} features in {sample_id}")
                    updated_count += 1
            
            except Exception as e:
                error_msg = f"Failed to update coordinates for {json_file}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if self.config.fail_fast:
                    raise
        
        logger.info(f"Updated coordinates for {updated_count} samples")
        if errors:
            logger.warning(f"{len(errors)} coordinate update errors occurred")
        
        return len(errors) == 0
    
    def _find_original_image(self, sample_id: str) -> Optional[Path]:
        """Find original image file for a sample."""
        for ext in ['.jpeg', '.jpg']:
            image_path = self.input_dir / f"{sample_id}{ext}"
            if image_path.exists():
                return image_path
        return None
    
    def _find_output_image(self, sample_id: str) -> Optional[Path]:
        """Find output image file for a sample."""
        for ext in ['.jpeg', '.jpg']:
            image_path = self.output_image_dir / f"{sample_id}{ext}"
            if image_path.exists():
                return image_path
        return None
    
    def _needs_coordinate_scaling(self, data: Dict, output_size: Tuple[int, int]) -> bool:
        """Check if coordinates need scaling based on image bounds."""
        if "markResult" not in data or "features" not in data["markResult"]:
            return False
        
        for feature in data["markResult"]["features"]:
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            
            if coordinates:
                # Check if any coordinate exceeds output image bounds
                x_coords = [pt[0] for pt in coordinates if len(pt) >= 2]
                y_coords = [pt[1] for pt in coordinates if len(pt) >= 2]
                
                if x_coords and y_coords:
                    max_x, max_y = max(x_coords), max(y_coords)
                    if max_x > output_size[0] or max_y > output_size[1]:
                        return True
        
        return False
    
    def _scale_json_coordinates(self, data: Dict, scale_x: float, scale_y: float) -> int:
        """Scale coordinates in JSON data, return number of features updated."""
        features_updated = 0
        
        if "markResult" not in data or "features" not in data["markResult"]:
            return 0
        
        for feature in data["markResult"]["features"]:
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "ExtentPolygon":
                continue
            
            coordinates = geometry.get("coordinates", [])
            if not coordinates:
                continue
            
            # Scale coordinates
            scaled_coordinates = []
            for point in coordinates:
                if len(point) >= 2:
                    scaled_x = point[0] * scale_x
                    scaled_y = point[1] * scale_y
                    scaled_coordinates.append([scaled_x, scaled_y])
            
            if scaled_coordinates:
                geometry["coordinates"] = scaled_coordinates
                features_updated += 1
        
        return features_updated
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files and backups."""
        if self.output_image_dir and self.output_image_dir.exists():
            # Remove backup files
            backup_files = list(self.output_image_dir.glob("*.backup"))
            for backup_file in backup_files:
                backup_file.unlink()
                logger.debug(f"Removed backup file: {backup_file}")
    
    def get_processing_summary(self) -> Dict[str, any]:
        """Get summary of image processing operations."""
        summary = {
            "resize_enabled": self.config.resize_enabled,
            "input_dir": str(self.input_dir),
            "output_image_dir": str(self.output_image_dir) if self.output_image_dir else None,
        }
        
        if self.output_image_dir and self.output_image_dir.exists():
            # Count processed files
            image_files = list(self.output_image_dir.glob("*.{jpeg,jpg}"))
            json_files = list(self.output_image_dir.glob("*.json"))
            
            summary.update({
                "processed_images": len(image_files),
                "processed_jsons": len(json_files),
            })
        
        return summary