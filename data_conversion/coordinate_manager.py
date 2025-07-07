#!/usr/bin/env python3
"""
Centralized Coordinate Management System

This module provides a unified approach to handling all coordinate transformations
including EXIF orientation, dimension rescaling, and smart resize operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class CoordinateManager:
    """
    Centralized coordinate transformation manager.
    
    Handles all coordinate transformations in proper order:
    1. EXIF orientation compensation
    2. Dimension mismatch rescaling 
    3. Smart resize scaling
    """
    
    @staticmethod
    def get_exif_transform_matrix(image_path: Path) -> Tuple[bool, int, int, int, int]:
        """
        Analyze EXIF orientation and return transformation info.
        
        Returns:
            (is_transformed, original_width, original_height, new_width, new_height)
        """
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            
            # Apply EXIF orientation to get transformed dimensions
            transformed_img = ImageOps.exif_transpose(img)
            new_width, new_height = transformed_img.size
            
            is_transformed = (original_width != new_width or original_height != new_height)
            
            return is_transformed, original_width, original_height, new_width, new_height
    
    @staticmethod
    def apply_exif_orientation_to_bbox(
        bbox: List[float], 
        original_width: int, 
        original_height: int, 
        new_width: int, 
        new_height: int,
        exif_orientation: int = None
    ) -> List[float]:
        """
        Transform bbox coordinates to account for EXIF orientation changes.
        
        Args:
            bbox: [x1, y1, x2, y2] in original coordinate system
            original_width, original_height: Image dimensions before EXIF transform
            new_width, new_height: Image dimensions after EXIF transform  
            exif_orientation: EXIF orientation value (optional, can detect from dimensions)
        
        Returns:
            Transformed bbox coordinates
        """
        if original_width == new_width and original_height == new_height:
            return bbox  # No transformation needed
        
        x1, y1, x2, y2 = bbox
        
        # Detect transformation type from dimension changes
        if original_width == new_height and original_height == new_width:
            # 90° or 270° rotation
            if new_width > new_height:
                # Likely 270° rotation (landscape from portrait)
                # Transform: (x,y) -> (y, original_width - x)
                new_x1 = y1
                new_y1 = original_width - x2
                new_x2 = y2  
                new_y2 = original_width - x1
            else:
                # Likely 90° rotation (portrait from landscape)
                # Transform: (x,y) -> (original_height - y, x)
                new_x1 = original_height - y2
                new_y1 = x1
                new_x2 = original_height - y1
                new_y2 = x2
        elif original_width == new_width and original_height == new_height:
            # 180° rotation
            # Transform: (x,y) -> (original_width - x, original_height - y)
            new_x1 = original_width - x2
            new_y1 = original_height - y2
            new_x2 = original_width - x1
            new_y2 = original_height - y1
        else:
            # Complex transformation or no transformation
            logger.warning(
                f"Unexpected EXIF transformation: {original_width}x{original_height} -> {new_width}x{new_height}"
            )
            return bbox
        
        # Ensure coordinates are in correct order
        final_x1 = min(new_x1, new_x2)
        final_y1 = min(new_y1, new_y2)
        final_x2 = max(new_x1, new_x2)
        final_y2 = max(new_y1, new_y2)
        
        # Clamp to image bounds
        final_x1 = max(0, min(final_x1, new_width))
        final_y1 = max(0, min(final_y1, new_height))
        final_x2 = max(0, min(final_x2, new_width))
        final_y2 = max(0, min(final_y2, new_height))
        
        logger.debug(
            f"EXIF bbox transform: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] -> "
            f"[{final_x1:.1f},{final_y1:.1f},{final_x2:.1f},{final_y2:.1f}]"
        )
        
        return [final_x1, final_y1, final_x2, final_y2]
    
    @staticmethod
    def apply_dimension_rescaling(
        bbox: List[float],
        json_width: int,
        json_height: int, 
        actual_width: int,
        actual_height: int
    ) -> List[float]:
        """
        Rescale bbox coordinates when JSON dimensions differ from actual image dimensions.
        
        This typically happens when:
        1. EXIF orientation was applied to image but not to JSON coordinates
        2. Image was preprocessed but JSON coordinates weren't updated
        
        Note: Returns float coordinates to preserve precision for subsequent transformations.
        Final integer conversion happens in smart_resize_scaling.
        """
        if json_width == actual_width and json_height == actual_height:
            return bbox  # No rescaling needed
        
        scale_x = actual_width / json_width
        scale_y = actual_height / json_height
        
        x1, y1, x2, y2 = bbox
        
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y
        
        # Clamp to image bounds (keep as float for precision)
        new_x1 = max(0.0, min(new_x1, float(actual_width)))
        new_y1 = max(0.0, min(new_y1, float(actual_height)))
        new_x2 = max(0.0, min(new_x2, float(actual_width)))
        new_y2 = max(0.0, min(new_y2, float(actual_height)))
        
        logger.debug(
            f"Dimension rescale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({json_width}x{json_height}) -> [{new_x1:.1f},{new_y1:.1f},{new_x2:.1f},{new_y2:.1f}] "
            f"({actual_width}x{actual_height})"
        )
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    @staticmethod  
    def apply_smart_resize_scaling(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int
    ) -> List[int]:
        """
        Scale bbox coordinates for smart resize operation.
        Returns integer coordinates suitable for pixel-based operations.
        """
        if original_width == new_width and original_height == new_height:
            # Convert to integers even if no scaling needed
            x1, y1, x2, y2 = bbox
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        x1, y1, x2, y2 = bbox
        
        # Apply scaling and round to integers
        new_x1 = int(round(x1 * scale_x))
        new_y1 = int(round(y1 * scale_y))
        new_x2 = int(round(x2 * scale_x))
        new_y2 = int(round(y2 * scale_y))
        
        # Clamp to image bounds and ensure valid bbox
        new_x1 = max(0, min(new_x1, new_width - 1))
        new_y1 = max(0, min(new_y1, new_height - 1))
        new_x2 = max(new_x1 + 1, min(new_x2, new_width))  # Ensure x2 > x1
        new_y2 = max(new_y1 + 1, min(new_y2, new_height))  # Ensure y2 > y1
        
        logger.debug(
            f"Smart resize scale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({original_width}x{original_height}) -> [{new_x1},{new_y1},{new_x2},{new_y2}] "
            f"({new_width}x{new_height})"
        )
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    @classmethod
    def transform_bbox_complete(
        cls,
        bbox: List[float],
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
        smart_resize_factor: int = 28
    ) -> Tuple[List[float], int, int]:
        """
        Apply complete bbox transformation pipeline.
        
        Pipeline:
        1. Apply EXIF orientation compensation
        2. Apply dimension mismatch rescaling
        3. Apply smart resize scaling (if enabled)
        
        Returns:
            (transformed_bbox, final_width, final_height)
        """
        # Step 1: Get EXIF transformation info
        is_exif_transformed, orig_w, orig_h, exif_w, exif_h = cls.get_exif_transform_matrix(image_path)
        
        current_bbox = bbox
        current_width, current_height = exif_w, exif_h
        
        # Step 2: Apply EXIF orientation compensation if needed
        if is_exif_transformed:
            current_bbox = cls.apply_exif_orientation_to_bbox(
                current_bbox, orig_w, orig_h, exif_w, exif_h
            )
            logger.debug(f"Applied EXIF orientation: {orig_w}x{orig_h} -> {exif_w}x{exif_h}")
        
        # Step 3: Apply dimension mismatch rescaling if needed
        if json_width != current_width or json_height != current_height:
            current_bbox = cls.apply_dimension_rescaling(
                current_bbox, json_width, json_height, current_width, current_height
            )
            logger.debug(f"Applied dimension rescaling: {json_width}x{json_height} -> {current_width}x{current_height}")
        
        # Step 4: Apply smart resize scaling if enabled
        if enable_smart_resize:
            # Use the proper smart_resize function from vision_process.py that respects MAX_PIXELS
            from data_conversion.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS
            resize_h, resize_w = smart_resize(
                height=current_height, 
                width=current_width, 
                factor=smart_resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
            )
            
            if resize_w != current_width or resize_h != current_height:
                current_bbox = cls.apply_smart_resize_scaling(
                    current_bbox, current_width, current_height, resize_w, resize_h
                )
                current_width, current_height = resize_w, resize_h
                logger.debug(f"Applied smart resize: {current_width}x{current_height} (within MAX_PIXELS={MAX_PIXELS})")
        
        return current_bbox, current_width, current_height
    
    @staticmethod
    def validate_bbox_bounds(bbox: List[float], width: int, height: int) -> bool:
        """
        Validate that bbox coordinates are within image bounds.
        
        Args:
            bbox: [x1, y1, x2, y2] (can be int or float)
            width, height: Image dimensions
        
        Returns:
            True if bbox is valid, False otherwise
        """
        x1, y1, x2, y2 = bbox
        
        # Check coordinate order
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Check bounds (allow small floating point tolerance)
        tolerance = 0.1
        if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
            return False
        
        return True
    
    @classmethod
    def process_sample_coordinates(
        cls,
        sample_data: Dict,
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True
    ) -> Tuple[Dict, int, int]:
        """
        Process all object coordinates in a sample.
        
        Args:
            sample_data: Sample with 'objects' list containing 'bbox_2d' fields
            image_path: Path to the image file
            json_width, json_height: Dimensions from JSON metadata
            enable_smart_resize: Whether to apply smart resize
        
        Returns:
            (updated_sample_data, final_width, final_height)
        """
        if "objects" not in sample_data or not sample_data["objects"]:
            # No objects to process, just get final dimensions
            if enable_smart_resize:
                from data_conversion.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS
                _, _, _, final_w, final_h = cls.get_exif_transform_matrix(image_path)
                resize_h, resize_w = smart_resize(
                    height=final_h, 
                    width=final_w,
                    factor=28,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h = cls.get_exif_transform_matrix(image_path)
                return sample_data, final_w, final_h
        
        # Process first object to get final dimensions
        first_bbox = sample_data["objects"][0]["bbox_2d"]
        _, final_width, final_height = cls.transform_bbox_complete(
            first_bbox, image_path, json_width, json_height, enable_smart_resize
        )
        
        # Process all objects
        updated_objects = []
        for obj in sample_data["objects"]:
            if "bbox_2d" not in obj:
                updated_objects.append(obj)
                continue
            
            transformed_bbox, _, _ = cls.transform_bbox_complete(
                obj["bbox_2d"], image_path, json_width, json_height, enable_smart_resize
            )
            
            # Validate transformed coordinates
            if cls.validate_bbox_bounds(transformed_bbox, final_width, final_height):
                updated_obj = obj.copy()
                updated_obj["bbox_2d"] = transformed_bbox
                updated_objects.append(updated_obj)
            else:
                logger.warning(f"Dropping invalid bbox after transformation: {transformed_bbox}")
        
        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects
        
        return updated_sample, final_width, final_height