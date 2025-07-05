#!/usr/bin/env python3
"""
Data Validation Utilities

Provides comprehensive validation for data structures, bbox coordinates,
and pipeline outputs.
"""

import logging
import sys
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data structures and content."""
    
    @staticmethod
    def validate_bbox(
        bbox: List[float], 
        image_width: Optional[int] = None, 
        image_height: Optional[int] = None
    ) -> bool:
        """
        Validate a single bounding box with enhanced checks.
        
        Args:
            bbox: A list of 4 numbers [x_min, y_min, x_max, y_max]
            image_width: Optional width to check bounds
            image_height: Optional height to check bounds
            
        Raises:
            ValueError: If the bounding box is invalid
        """
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Bbox must be a list of 4 elements, got: {bbox}")
        
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            raise ValueError(f"Bbox coordinates must be numbers, got: {bbox}")
        
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure correct ordering
        if x_min > x_max:
            x_min, x_max = x_max, x_min
            bbox = [x_min, y_min, x_max, y_max]
        if y_min > y_max:
            y_min, y_max = y_max, y_min
            bbox = [x_min, y_min, x_max, y_max]
        
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid bbox: x_min < x_max and y_min < y_max required, got: {bbox}"
            )
        
        if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
            raise ValueError(f"Bbox coordinates cannot be negative, got: {bbox}")
        
        if image_width is not None and image_height is not None:
            if x_max > image_width or y_max > image_height:
                raise ValueError(
                    f"Bbox {bbox} exceeds image dimensions ({image_width}x{image_height})"
                )
        
        return True
    
    @staticmethod
    def validate_sample_structure(sample: Dict[str, Any]) -> bool:
        """Validate basic sample structure for training data."""
        required_fields = ["images", "objects"]
        
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate images field
        images = sample.get("images")
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Field 'images' must be a non-empty list")
        
        # Validate objects field
        objects = sample.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Field 'objects' must be a list")
        
        # Validate each object
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Object {i} must be a dictionary")
            
            if "bbox_2d" not in obj or "desc" not in obj:
                raise ValueError(f"Object {i} missing 'bbox_2d' or 'desc'")
            
            # Validate bbox format
            DataValidator.validate_bbox(obj["bbox_2d"])
            
            # Validate description
            desc = obj["desc"]
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Object {i} 'desc' must be non-empty string")
        
        return True
    
    @staticmethod
    def validate_teacher_student_structure(sample: Dict[str, Any]) -> bool:
        """Validate teacher-student sample structure."""
        if "student" not in sample:
            raise ValueError("Missing 'student' field in teacher-student sample")
        
        # Validate student
        DataValidator.validate_sample_structure(sample["student"])
        
        # Validate teachers if present
        if "teachers" in sample:
            teachers = sample["teachers"]
            if not isinstance(teachers, list):
                raise ValueError("Field 'teachers' must be a list")
            
            for i, teacher in enumerate(teachers):
                try:
                    DataValidator.validate_sample_structure(teacher)
                except ValueError as e:
                    raise ValueError(f"Teacher {i}: {e}")
        
        return True
    
    @staticmethod
    def validate_json_annotation(data: Dict[str, Any]) -> bool:
        """Validate raw JSON annotation structure."""
        # Check required info section
        if "info" not in data:
            raise ValueError("Missing 'info' section")
        
        info = data["info"]
        if "width" not in info or "height" not in info:
            raise ValueError("Missing width/height in info section")
        
        # Check for annotation data
        has_data_list = "dataList" in data and isinstance(data["dataList"], list)
        has_mark_result = (
            "markResult" in data and 
            isinstance(data.get("markResult", {}).get("features"), list)
        )
        
        if not (has_data_list or has_mark_result):
            raise ValueError("No valid annotation data found")
        
        return True
    
    @staticmethod
    def validate_content_dict(content_dict: Dict[str, str], required_fields: Optional[List[str]] = None) -> bool:
        """Validate content dictionary structure."""
        if not isinstance(content_dict, dict):
            raise ValueError("Content must be a dictionary")
        
        required_fields = required_fields or ["object_type"]
        
        for field in required_fields:
            if field not in content_dict:
                raise ValueError(f"Missing required content field: {field}")
        
        # Validate that values are strings
        for key, value in content_dict.items():
            if not isinstance(value, str):
                raise ValueError(f"Content field '{key}' must be a string, got: {type(value)}")
        
        return True


class StructureValidator:
    """Validates pipeline structures and outputs."""
    
    @staticmethod
    def validate_pipeline_output(
        train_samples: List[Dict], 
        val_samples: List[Dict], 
        teacher_samples: List[Dict]
    ) -> bool:
        """Validate complete pipeline output."""
        if not train_samples:
            raise ValueError("Training samples cannot be empty")
        
        # For small datasets, validation samples can be empty
        if not val_samples and len(train_samples) > 1:
            raise ValueError("Validation samples cannot be empty when multiple training samples exist")
        
        # For very small datasets, teacher samples can be empty
        if not teacher_samples and len(train_samples) + len(val_samples) > 2:
            raise ValueError("Teacher samples cannot be empty when sufficient samples exist")
        
        # Validate sample structures
        for i, sample in enumerate(train_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Train sample {i}: {e}")
        
        for i, sample in enumerate(val_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Validation sample {i}: {e}")
        
        for i, sample in enumerate(teacher_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Teacher sample {i}: {e}")
        
        # Check for overlap between sets
        train_images = {sample["images"][0] for sample in train_samples}
        val_images = {sample["images"][0] for sample in val_samples}
        teacher_images = {sample["images"][0] for sample in teacher_samples}
        
        if train_images & val_images:
            raise ValueError("Training and validation sets have overlapping images")
        
        if train_images & teacher_images:
            raise ValueError("Training and teacher sets have overlapping images")
        
        if val_images & teacher_images:
            raise ValueError("Validation and teacher sets have overlapping images")
        
        logger.info(
            f"Pipeline output validation passed: "
            f"{len(train_samples)} train, {len(val_samples)} val, {len(teacher_samples)} teacher"
        )
        
        return True
    
    @staticmethod
    def validate_processing_statistics(stats: Dict[str, int]) -> bool:
        """Validate processing statistics."""
        required_stats = ["processed", "skipped", "total"]
        
        for stat in required_stats:
            if stat not in stats:
                raise ValueError(f"Missing statistic: {stat}")
            
            if not isinstance(stats[stat], int) or stats[stat] < 0:
                raise ValueError(f"Invalid statistic value for {stat}: {stats[stat]}")
        
        if stats["processed"] + stats["skipped"] != stats["total"]:
            raise ValueError("Statistics don't add up: processed + skipped != total")
        
        return True
    
    @staticmethod
    def validate_coordinate_scaling(
        original_bbox: List[float],
        scaled_bbox: List[float],
        scale_x: float,
        scale_y: float,
        tolerance: float = 1.0
    ) -> bool:
        """Validate coordinate scaling accuracy."""
        expected_x1 = original_bbox[0] * scale_x
        expected_y1 = original_bbox[1] * scale_y
        expected_x2 = original_bbox[2] * scale_x
        expected_y2 = original_bbox[3] * scale_y
        
        actual_x1, actual_y1, actual_x2, actual_y2 = scaled_bbox
        
        if (abs(actual_x1 - expected_x1) > tolerance or
            abs(actual_y1 - expected_y1) > tolerance or
            abs(actual_x2 - expected_x2) > tolerance or
            abs(actual_y2 - expected_y2) > tolerance):
            raise ValueError(
                f"Coordinate scaling validation failed: "
                f"expected [{expected_x1:.1f}, {expected_y1:.1f}, {expected_x2:.1f}, {expected_y2:.1f}], "
                f"got {scaled_bbox}"
            )
        
        return True