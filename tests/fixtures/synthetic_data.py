"""
Synthetic Data Generator for BBU Testing

Generates realistic BBU equipment detection samples with Chinese labels for
comprehensive testing of the training pipeline.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from src.logger_utils import get_logger


logger = get_logger("synthetic_data")


class SyntheticDataGenerator:
    """Professional synthetic data generator for BBU training pipeline testing."""

    # Realistic BBU equipment labels in Chinese
    EQUIPMENT_LABELS = [
        "BBU设备/显示完整，华为/无需安装",
        "标签/5G-BBU-接地线",
        "螺丝、光纤插头/显示完整/BBU安装螺丝/符合要求",
        "电线/无遮挡，捆扎整齐",
        "标签/4G-RRU3-光纤",
        "螺丝、光纤插头/只显示部分/机柜处接地螺丝/符合要求",
        "设备标识/清晰可见/华为BBU3900",
        "光纤连接/状态正常/多模光纤",
        "电源线/规范布线/220V交流电",
        "散热风扇/运行正常/低噪音模式",
        "LED指示灯/绿色常亮/系统正常",
        "网线接口/RJ45/千兆以太网",
    ]

    def __init__(
        self,
        num_samples: int = 20,
        image_size: Tuple[int, int] = (280, 560),
        temp_dir_prefix: str = "bbu_test_data_",
    ):
        """
        Initialize synthetic data generator.

        Args:
            num_samples: Total number of samples to generate
            image_size: Image dimensions (width, height) - must be multiples of 28
            temp_dir_prefix: Prefix for temporary directory name
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.temp_dir = Path(tempfile.mkdtemp(prefix=temp_dir_prefix))
        self.images_dir = self.temp_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # Validate image size (must be multiples of 28 for Qwen2.5-VL)
        if image_size[0] % 28 != 0 or image_size[1] % 28 != 0:
            raise ValueError(
                f"Image size {image_size} must be multiples of 28 for Qwen2.5-VL compatibility"
            )

        logger.info(
            f"🎨 Initialized SyntheticDataGenerator: {num_samples} samples, {image_size} images"
        )
        logger.info(f"📁 Data directory: {self.temp_dir}")

    def generate_synthetic_image(self, filename: str) -> Path:
        """
        Generate a synthetic image with realistic BBU equipment patterns.

        Args:
            filename: Image filename

        Returns:
            Path to generated image
        """
        width, height = self.image_size

        # Create base image with realistic equipment background
        image_array = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)

        # Add equipment-like rectangular regions
        num_equipment_regions = random.randint(2, 5)
        for _ in range(num_equipment_regions):
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, width))
            y2 = random.randint(y1 + 30, min(y1 + 150, height))

            # Equipment colors (metallic grays, blues)
            colors = [
                [60, 70, 80],  # Dark gray
                [100, 110, 120],  # Light gray
                [40, 50, 80],  # Dark blue
                [80, 90, 100],  # Medium gray
            ]
            color_idx = np.random.choice(len(colors))
            color = colors[color_idx]
            image_array[y1:y2, x1:x2] = color

        # Add some noise and texture
        noise = np.random.randint(-10, 10, (height, width, 3))
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

        # Create and save PIL image
        image = Image.fromarray(image_array, "RGB")
        image_path = self.images_dir / filename
        image.save(image_path, "JPEG", quality=90)

        return image_path

    def generate_bbox_2d_coordinates(self) -> List[int]:
        """Generate realistic bounding box coordinates."""
        width, height = self.image_size
        x1 = random.randint(10, width // 3)
        y1 = random.randint(10, height // 3)
        x2 = random.randint(x1 + 30, min(x1 + 200, width - 10))
        y2 = random.randint(y1 + 20, min(y1 + 120, height - 10))
        return [x1, y1, x2, y2]

    def generate_square_coordinates(self) -> List[int]:
        """Generate square/quadrilateral coordinates (8 points)."""
        width, height = self.image_size
        # Generate 4 corner points for a roughly rectangular shape
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)

        # Generate corners with some variation
        half_width = random.randint(30, 80)
        half_height = random.randint(20, 60)

        coords = []
        corners = [
            (center_x - half_width, center_y - half_height),  # Top-left
            (center_x + half_width, center_y - half_height),  # Top-right
            (center_x + half_width, center_y + half_height),  # Bottom-right
            (center_x - half_width, center_y + half_height),  # Bottom-left
        ]

        for x, y in corners:
            # Add some variation to make it more realistic
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            coords.extend([x, y])

        return coords

    def generate_line_coordinates(self) -> List[int]:
        """Generate polyline coordinates (6-12 points)."""
        width, height = self.image_size
        num_points = random.randint(3, 6)  # 3-6 points

        coords = []
        # Start from a random point
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        coords.extend([x, y])

        # Generate subsequent points following a rough path
        for _ in range(num_points - 1):
            # Move in a somewhat consistent direction
            dx = random.randint(-50, 50)
            dy = random.randint(-30, 30)
            x = max(10, min(x + dx, width - 10))
            y = max(10, min(y + dy, height - 10))
            coords.extend([x, y])

        return coords

    def generate_detection_object(
        self, use_multi_geometry: bool = True, force_geometry_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate a single detection object with realistic geometry.

        Args:
            use_multi_geometry: Whether to use multiple geometry types
            force_geometry_type: Force specific geometry type for testing

        Returns:
            Dictionary containing object description and geometry
        """
        obj = {"desc": random.choice(self.EQUIPMENT_LABELS)}

        if force_geometry_type:
            # Force specific geometry type (for testing)
            if force_geometry_type == "bbox_2d":
                obj["bbox_2d"] = self.generate_bbox_2d_coordinates()
            elif force_geometry_type == "square":
                obj["square"] = self.generate_square_coordinates()
            elif force_geometry_type == "line":
                obj["line"] = self.generate_line_coordinates()
        elif use_multi_geometry:
            # Randomly choose ONE geometry type (realistic distribution)
            geometry_choice = random.random()
            if geometry_choice < 0.6:  # 60% bbox_2d (most common)
                obj["bbox_2d"] = self.generate_bbox_2d_coordinates()
            elif geometry_choice < 0.8:  # 20% square
                obj["square"] = self.generate_square_coordinates()
            else:  # 20% line
                obj["line"] = self.generate_line_coordinates()
        else:
            # Default to bbox_2d only
            obj["bbox_2d"] = self.generate_bbox_2d_coordinates()

        return obj

    def generate_flat_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Generate a single flat sample in BBU format.

        Args:
            sample_id: Unique sample identifier

        Returns:
            Dictionary containing the complete sample data
        """
        # Generate synthetic image
        image_filename = f"test_sample_{sample_id:04d}.jpg"
        self.generate_synthetic_image(image_filename)

        # Generate 1-4 objects per sample (realistic distribution)
        num_objects = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.2, 0.05])[0]
        objects = []
        
        # For integration tests, ensure geometry variety by forcing different types
        if self.num_samples >= 12:  # Integration test size
            geometry_types = ["bbox_2d", "square", "line"]
            
            # Ensure each sample gets at least one object with the assigned geometry type
            if num_objects >= 1:
                # Distribute geometry types across samples to ensure all types appear
                primary_type = geometry_types[sample_id % len(geometry_types)]
                objects.append(self.generate_detection_object(force_geometry_type=primary_type))
                
                # Add remaining objects with mixed geometry (50% assigned type, 50% random)
                for i in range(1, num_objects):
                    if random.random() < 0.5:
                        # Use the same type as primary to increase count
                        objects.append(self.generate_detection_object(force_geometry_type=primary_type))
                    else:
                        # Use random type from all available types
                        random_type = random.choice(geometry_types)
                        objects.append(self.generate_detection_object(force_geometry_type=random_type))
        else:
            # Normal generation for smaller test datasets
            for _ in range(num_objects):
                objects.append(self.generate_detection_object())

        return {
            "images": [f"images/{image_filename}"],
            "objects": objects,
            "width": self.image_size[0],
            "height": self.image_size[1],
        }

    def generate_teacher_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Generate a teacher pool sample.
        
        Teacher samples use the same format as flat samples for simplicity.
        The teacher-student pairing logic is handled by the dataset loader.
        """
        return self.generate_flat_sample(sample_id)

    def generate_complete_dataset(self) -> Tuple[Path, Path, Path, Path]:
        """
        Generate complete dataset with train, validation, teacher, and combined files.

        Returns:
            Tuple of (train_path, val_path, teacher_path, all_samples_path)
        """
        logger.info(f"🎨 Generating complete dataset: {self.num_samples} samples")

        # Dataset split: 60% train, 20% val, 20% teacher
        # Ensure minimum counts for small datasets
        if self.num_samples <= 6:
            # For small datasets, ensure at least 1 sample in each split
            train_count = max(1, int(self.num_samples * 0.6))
            val_count = max(1, int(self.num_samples * 0.2))
            teacher_count = max(1, self.num_samples - train_count - val_count)
            # Adjust if total exceeds num_samples
            if train_count + val_count + teacher_count > self.num_samples:
                if self.num_samples >= 3:
                    train_count = self.num_samples - 2
                    val_count = 1
                    teacher_count = 1
                else:
                    # Very small datasets
                    train_count = 1
                    val_count = 1 if self.num_samples > 1 else 0
                    teacher_count = max(1, self.num_samples - train_count - val_count)
        else:
            train_count = int(self.num_samples * 0.6)
            val_count = int(self.num_samples * 0.2)
            teacher_count = self.num_samples - train_count - val_count

        datasets = {
            "train.jsonl": ("flat", train_count),
            "val.jsonl": ("flat", val_count),
            "teacher.jsonl": ("teacher", teacher_count),
        }

        sample_id = 0
        generated_paths = []

        for dataset_name, (sample_type, count) in datasets.items():
            dataset_path = self.temp_dir / dataset_name

            with open(dataset_path, "w", encoding="utf-8") as f:
                for i in range(count):
                    if sample_type == "teacher":
                        sample = self.generate_teacher_sample(sample_id)
                    else:  # flat
                        sample = self.generate_flat_sample(sample_id)

                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    sample_id += 1

            logger.info(f"✅ Generated {dataset_name}: {count} samples")
            generated_paths.append(dataset_path)

        # Generate all_samples.jsonl (train + val combined)
        all_samples_path = self.temp_dir / "all_samples.jsonl"
        with open(all_samples_path, "w", encoding="utf-8") as f_all:
            # Copy train samples
            with open(self.temp_dir / "train.jsonl", "r", encoding="utf-8") as f_train:
                for line in f_train:
                    f_all.write(line)
            # Copy val samples
            with open(self.temp_dir / "val.jsonl", "r", encoding="utf-8") as f_val:
                for line in f_val:
                    f_all.write(line)

        generated_paths.append(all_samples_path)

        logger.info(f"🎯 Dataset generation complete!")
        logger.info(f"   📁 Data root: {self.temp_dir}")
        logger.info(f"   🖼️ Images: {sample_id} synthetic images")
        logger.info(
            f"   📊 Split: {train_count} train, {val_count} val, {teacher_count} teacher"
        )

        return tuple(generated_paths)  # train, val, teacher, all_samples

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated data."""
        return {
            "temp_dir": str(self.temp_dir),
            "num_samples": self.num_samples,
            "image_size": self.image_size,
            "equipment_labels_count": len(self.EQUIPMENT_LABELS),
            "images_generated": len(list(self.images_dir.glob("*.jpg"))),
        }

    def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up test data: {self.temp_dir}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
