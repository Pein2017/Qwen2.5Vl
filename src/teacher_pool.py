#!/usr/bin/env python3
"""
Teacher Pool Manager for Multi-Chat Training

This module manages the teacher pool for the teacher-student training approach.
The teacher pool is stored as a JSONL file (``teacher_pool.jsonl``) where each
line is a *clean-format* sample in the form::

    {"images": ["path.jpg"], "objects": [{"bbox_2d": [...], "desc": "..."}, ...]}

The manager loads these samples and can randomly assign teacher examples to
student samples during training.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import config
from src.logger_utils import get_data_logger

logger = get_data_logger()


class TeacherPoolManager:
    """
    Manages the teacher pool for multi-chat training.

    Loads teacher samples from the intermediate JSONL and provides
    random teacher assignment for student samples.
    """

    def __init__(self, teacher_pool_file: str):
        """Initialize the teacher pool manager.

        Args:
            teacher_pool_file: Path to ``teacher_pool.jsonl`` containing teacher samples.
        """
        self.teacher_pool_file = Path(teacher_pool_file)

        # Load teacher samples from teacher_pool.jsonl (already clean format)
        self.teacher_samples = self._load_teacher_samples_from_jsonl()

        # Derive image path list for convenience
        self.teacher_image_paths = [
            sample["images"][0] for sample in self.teacher_samples
        ]

        logger.info("✅ TeacherPoolManager initialized:")
        logger.info(f"   Teacher pool file: {teacher_pool_file}")
        logger.info(f"   Number of teacher images: {len(self.teacher_image_paths)}")
        logger.info(f"   Number of teacher samples: {len(self.teacher_samples)}")

    def _load_teacher_samples_from_jsonl(self) -> List[Dict[str, Any]]:
        """Load teacher samples directly from teacher_pool.jsonl (clean format)."""
        if not self.teacher_pool_file.exists():
            raise FileNotFoundError(
                f"Teacher pool file not found: {self.teacher_pool_file}"
            )

        teacher_samples: List[Dict[str, Any]] = []
        with open(self.teacher_pool_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "images" not in sample or "objects" not in sample:
                    raise ValueError(
                        "Each teacher sample must contain 'images' and 'objects'"
                    )
                teacher_samples.append(sample)

        if not teacher_samples:
            raise ValueError("No teacher samples found in teacher pool file")

        return teacher_samples

    def _convert_to_clean_format(
        self, intermediate_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert intermediate JSONL format to clean semantic format."""
        if "images" not in intermediate_sample:
            raise KeyError("Sample missing required 'images' key")
        if "objects" not in intermediate_sample:
            raise KeyError("Sample missing required 'objects' key")

        images = intermediate_sample["images"]
        objects_data = intermediate_sample["objects"]

        # Extract ref and bbox lists
        if "ref" not in objects_data:
            raise KeyError("Objects data missing required 'ref' key")
        if "bbox" not in objects_data:
            raise KeyError("Objects data missing required 'bbox' key")

        ref_list = objects_data["ref"]
        bbox_list = objects_data["bbox"]

        # Convert to clean format
        clean_objects = []
        for ref_desc, bbox in zip(ref_list, bbox_list):
            # Filter description based on response types (same logic as converter)
            clean_desc = self._filter_description(ref_desc)
            clean_objects.append({"bbox_2d": bbox, "desc": clean_desc})

        return {"images": images, "objects": clean_objects}

    def _filter_description(self, description: str) -> str:
        """Filter description based on response types (simplified version)."""
        # For Chinese descriptions, keep as-is since they're already clean
        if ";" not in description:
            return description

        # For English verbose format, convert to compact
        # This is a simplified version - in practice you'd use ResponseFormatter
        parts = description.split(";")
        clean_parts = []

        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key in ["object_type", "property"] and value != "none":
                    clean_parts.append(value)

        if not clean_parts:
            raise ValueError(
                f"Could not extract meaningful description from: {description}"
            )
        return "/".join(clean_parts)

    def get_random_teacher(self, seed: Optional[int]) -> Dict[str, Any]:
        """
        Get a random teacher sample.

        Args:
            seed: Seed for reproducibility (required for deterministic behavior)

        Returns:
            Random teacher sample in clean format
        """
        if seed is not None:
            random.seed(seed)

        return random.choice(self.teacher_samples)

    def get_multiple_teachers(
        self, num_teachers: int, seed: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Get multiple random teacher samples without replacement.

        Args:
            num_teachers: Number of teachers to sample
            seed: Seed for reproducibility (required for deterministic behavior)

        Returns:
            List of teacher samples
        """
        if seed is not None:
            random.seed(seed)

        if num_teachers >= len(self.teacher_samples):
            return self.teacher_samples.copy()

        return random.sample(self.teacher_samples, num_teachers)

    def create_multi_chat_sample(
        self,
        student_sample: Dict[str, Any],
        num_teachers: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """
        Create a multi-chat sample by adding teacher samples to a student sample.

        Args:
            student_sample: Student sample in clean format
            num_teachers: Number of teacher samples to add (must be > 0)
            seed: Seed for reproducibility (required for deterministic behavior)

        Returns:
            Multi-chat sample with teachers and student structure
        """
        if num_teachers <= 0:
            raise ValueError(f"num_teachers must be > 0, got {num_teachers}")

        # Get teacher samples
        teacher_samples = self.get_multiple_teachers(num_teachers, seed)

        return {"teachers": teacher_samples, "student": student_sample}

    def __len__(self) -> int:
        """Return number of teacher samples."""
        return len(self.teacher_samples)


def create_teacher_pool_manager() -> TeacherPoolManager:
    """
    Create teacher pool manager from global config.

    Returns:
        TeacherPoolManager instance

    Raises:
        AttributeError: If teacher_pool_file not configured
        FileNotFoundError: If required files not found
        ValueError: If configuration is invalid
    """
    # Fail fast if not configured
    if not hasattr(config, "teacher_pool_file"):
        raise AttributeError("teacher_pool_file not configured in config")

    if not config.teacher_pool_file:
        raise ValueError("teacher_pool_file is empty in config")

    # Create manager (let it fail if file invalid)
    return TeacherPoolManager(teacher_pool_file=config.teacher_pool_file)
