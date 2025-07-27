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

from src.logger_utils import get_data_logger


logger = get_data_logger()


class TeacherPoolManager:
    """
    Manages the teacher pool for multi-chat training.

    Loads teacher samples from the intermediate JSONL and provides
    random teacher assignment for student samples.
    """

    def __init__(self, teacher_pool_file: str, data_root: str = None):
        """Initialize the teacher pool manager.

        Args:
            teacher_pool_file: Path to ``teacher_pool.jsonl`` containing teacher samples.
            data_root: Root directory for data files (optional)

        Raises:
            FileNotFoundError: If the teacher pool file doesn't exist
            ValueError: If the teacher pool file is empty or contains invalid data
        """
        # FAIL-FAST: Validate teacher_pool_file
        if not teacher_pool_file:
            raise ValueError("teacher_pool_file cannot be empty")

        self.teacher_pool_file = Path(teacher_pool_file)

        # FAIL-FAST: Validate teacher_pool_file exists
        if not self.teacher_pool_file.exists():
            raise FileNotFoundError(f"Teacher pool file not found: {teacher_pool_file}")
        if not self.teacher_pool_file.is_file():
            raise ValueError(f"Teacher pool path is not a file: {teacher_pool_file}")

        # Validate data_root if provided
        if data_root:
            data_root_path = Path(data_root)
            if not data_root_path.exists():
                raise FileNotFoundError(f"Data root directory not found: {data_root}")
            if not data_root_path.is_dir():
                raise ValueError(f"Data root is not a directory: {data_root}")
            self.data_root = data_root_path
        else:
            self.data_root = None

        # Load teacher samples from teacher_pool.jsonl (already clean format)
        self.teacher_samples = self._load_teacher_samples_from_jsonl()

        # FAIL-FAST: Validate teacher samples were loaded
        if not self.teacher_samples:
            raise ValueError(
                f"No valid teacher samples loaded from {teacher_pool_file}"
            )

        # Derive image path list for convenience
        try:
            self.teacher_image_paths = [
                sample["images"][0] for sample in self.teacher_samples
            ]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid teacher sample structure: {e}")

        logger.info("✅ TeacherPoolManager initialized:")
        logger.info(f"   Teacher pool file: {teacher_pool_file}")
        logger.info(f"   Data root: {data_root}")
        logger.info(f"   Number of teacher images: {len(self.teacher_image_paths)}")
        logger.info(f"   Number of teacher samples: {len(self.teacher_samples)}")

    def _load_teacher_samples_from_jsonl(self) -> List[Dict[str, Any]]:
        """
        Load teacher samples directly from JSONL file using flat sample format.

        Raises:
            FileNotFoundError: If the teacher pool file doesn't exist
            ValueError: If the teacher pool contains invalid samples
            json.JSONDecodeError: If a line contains invalid JSON
        """
        if not self.teacher_pool_file.exists():
            raise FileNotFoundError(
                f"Teacher pool file not found: {self.teacher_pool_file}"
            )

        teacher_samples: List[Dict[str, Any]] = []
        with open(self.teacher_pool_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON in teacher pool at line {line_num}: {e.msg}",
                        e.doc,
                        e.pos,
                    )

                # Validate flat sample format with fail-fast approach
                if not isinstance(sample, dict):
                    raise ValueError(f"Line {line_num}: Sample is not a dictionary")

                if "images" not in sample or "objects" not in sample:
                    raise ValueError(
                        f"Line {line_num}: Missing required fields 'images' or 'objects'"
                    )

                if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
                    raise ValueError(
                        f"Line {line_num}: 'images' field is empty or not a list"
                    )

                if not isinstance(sample["objects"], list):
                    raise ValueError(f"Line {line_num}: 'objects' field is not a list")

                # Add validated sample
                teacher_samples.append(sample)

        if not teacher_samples:
            raise ValueError(f"No valid samples found in {self.teacher_pool_file}")

        logger.info(
            f"Loaded {len(teacher_samples)} teacher samples from {self.teacher_pool_file}"
        )
        return teacher_samples

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

        Raises:
            ValueError: If num_teachers is <= 0
        """
        if num_teachers <= 0:
            raise ValueError(f"num_teachers must be > 0, got {num_teachers}")

        # Get teacher samples
        teacher_samples = self.get_multiple_teachers(num_teachers, seed)

        return {"teachers": teacher_samples, "student": student_sample}

    def __len__(self) -> int:
        """Return number of teacher samples."""
        return len(self.teacher_samples)


def create_teacher_pool_manager(config=None) -> TeacherPoolManager:
    """
    Factory function to create teacher pool manager.

    Args:
        config: Configuration object (explicit config or global config)

    Returns:
        TeacherPoolManager instance

    Raises:
        ValueError: If teacher_pool_file is not specified in config
        Various exceptions from TeacherPoolManager initialization
    """
    # FAIL-FAST: Validate config is available
    if config is None:
        try:
            from src.config import get_config

            config = get_config()
        except RuntimeError as e:
            raise RuntimeError(
                f"No valid configuration provided and global config not initialized: {e}"
            )

    # FAIL-FAST: Validate required config attributes
    if not hasattr(config, "teacher_pool_file"):
        raise ValueError("No teacher_pool_file specified in config")
    if not config.teacher_pool_file:
        raise ValueError("teacher_pool_file in config cannot be empty")

    if not hasattr(config, "data_root"):
        raise ValueError("No data_root specified in config")

    # Create teacher pool manager with explicit error handling
    try:
        teacher_pool_manager = TeacherPoolManager(
            teacher_pool_file=config.teacher_pool_file,
            data_root=config.data_root,
        )
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Failed to initialize TeacherPoolManager: {e}")

    logger.info(
        f"Created teacher pool manager with {len(teacher_pool_manager)} samples"
    )
    return teacher_pool_manager
