#!/usr/bin/env python3
"""
Visualize the raw data exported from 数据堂, the *.json file.

This script loads raw annotation JSON files and visualizes bounding boxes
with labels on the corresponding images. Supports both English and Chinese labels.

Configure the settings below and run the script directly.
"""

# =============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# Image paths to visualize (relative to BASE_DIR)
# Add or remove image paths as needed
IMAGE_PATHS = [
    "ds/QC-20230225-0000414_19823.jpeg",
]

# Optional: Corresponding annotation paths (if not provided, will auto-detect)
# Leave as None to enable auto-detection based on image filenames
# Or specify exact paths like: ["ds/file1.json", "ds/file2.json", ...]
ANNOTATION_PATHS = None

# Output directory for visualizations
OUTPUT_DIR = "raw_visualizations"

# Language preference for labels
# "zh" = Chinese labels (标签 field)
# "en" = English labels (label field)
LANGUAGE = "zh"

# Base directory for resolving relative paths
BASE_DIR = "."

# Maximum number of images to process (-1 for all images in IMAGE_PATHS)
MAX_IMAGES = -1

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image

# Configure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Chinese font for matplotlib
try:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if os.path.exists(font_path):
        font_prop = FontProperties(fname=font_path)
        fontManager.addfont(font_path)
        rcParams["font.sans-serif"] = [font_prop.get_name()]
        rcParams["axes.unicode_minus"] = False
    else:
        logger.warning("Chinese font not found, falling back to default")
except Exception as e:
    logger.warning(f"Failed to configure Chinese font: {e}")


class RawDataVisualizer:
    """Visualizes raw annotation data from 数据堂 format."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.color_palette = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E9",
            "#F8C471",
            "#82E0AA",
            "#F1948A",
            "#85929E",
            "#F4D03F",
            "#AED6F1",
            "#A9DFBF",
            "#F9E79F",
            "#D7BDE2",
            "#A2D9CE",
            "#FADBD8",
            "#D5DBDB",
            "#FCF3CF",
            "#EBDEF0",
            "#D1F2EB",
        ]
        self.label_to_color = {}

    def load_image_safe(
        self, image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Safely load image and return array and dimensions."""
        try:
            # Try different path combinations
            path_candidates = [
                image_path,
                self.base_dir / image_path,
                self.base_dir / "ds" / Path(image_path).name,
                Path(image_path),
            ]

            for candidate in path_candidates:
                abs_path = Path(candidate)
                if abs_path.exists():
                    with Image.open(abs_path) as img:
                        img_rgb = img.convert("RGB")
                        img_array = np.array(img_rgb)
                        return img_array, img_rgb.size  # (width, height)

            logger.error(
                f"Image file not found in any candidate path: {path_candidates}"
            )
            return None, None

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, None

    def load_raw_annotation(self, json_path: str) -> Optional[Dict]:
        """Load raw annotation JSON file."""
        try:
            abs_path = (
                self.base_dir / json_path
                if not Path(json_path).is_absolute()
                else Path(json_path)
            )

            if not abs_path.exists():
                logger.error(f"Annotation file not found: {abs_path}")
                return None

            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Loaded annotation file: {abs_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load annotation {json_path}: {e}")
            return None

    def extract_bboxes_from_raw(
        self, annotation_data: Dict, lang: str = "zh"
    ) -> List[Dict]:
        """
        Extract bounding boxes from raw annotation format.

        Args:
            annotation_data: Raw annotation data (could be single feature or FeatureCollection)
            lang: Language preference ("zh" for Chinese, "en" for English)

        Returns:
            List of bbox dictionaries with format: {"bbox_2d": [x1, y1, x2, y2], "label": str}
        """
        bboxes = []

        # Handle the specific 数据堂 format
        features = []

        # First try markResult.features (main annotation data)
        if "markResult" in annotation_data and isinstance(
            annotation_data["markResult"], dict
        ):
            mark_result = annotation_data["markResult"]
            if mark_result.get("type") == "FeatureCollection":
                features = mark_result.get("features", [])
                logger.info(f"Found {len(features)} features in markResult")

        # Fallback to direct FeatureCollection format
        elif annotation_data.get("type") == "FeatureCollection":
            features = annotation_data.get("features", [])
            logger.info(f"Found {len(features)} features in direct FeatureCollection")

        # Fallback to single feature format
        elif annotation_data.get("geometry"):
            features = [annotation_data]
            logger.info("Found single feature format")

        if not features:
            logger.warning("No features found in annotation data")
            return bboxes

        for feature in features:
            try:
                # Extract geometry
                geometry = feature.get("geometry", {})
                if geometry.get("type") != "ExtentPolygon":
                    continue

                coordinates = geometry.get("coordinates", [])
                if not coordinates or len(coordinates) < 4:
                    continue

                # Convert polygon coordinates to bbox
                # coordinates format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
                x_coords = [point[0] for point in coordinates]
                y_coords = [point[1] for point in coordinates]

                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)

                # Extract label based on language preference
                properties = feature.get("properties", {})
                label = "Unknown"

                if lang == "zh" and "contentZh" in properties:
                    # Chinese label
                    content_zh = properties["contentZh"]
                    if isinstance(content_zh, dict) and "标签" in content_zh:
                        label = content_zh["标签"]
                    elif isinstance(content_zh, str):
                        label = content_zh
                elif "content" in properties:
                    # English label
                    content = properties["content"]
                    if isinstance(content, dict) and "label" in content:
                        label = content["label"]
                    elif isinstance(content, str):
                        label = content

                bboxes.append(
                    {"bbox_2d": [int(x1), int(y1), int(x2), int(y2)], "label": label}
                )

            except Exception as e:
                logger.warning(f"Failed to process feature: {e}")
                continue

        return bboxes

    def get_label_color(self, label: str) -> str:
        """Get consistent color for a label."""
        if label not in self.label_to_color:
            # Assign new color
            color_idx = len(self.label_to_color) % len(self.color_palette)
            self.label_to_color[label] = self.color_palette[color_idx]
        return self.label_to_color[label]

    def draw_bboxes_on_axis(
        self, ax, image_array: np.ndarray, bboxes: List[Dict], title: str
    ):
        """Draw bounding boxes on a matplotlib axis."""
        ax.imshow(image_array)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        for bbox_info in bboxes:
            bbox = bbox_info.get("bbox_2d", [])
            label = bbox_info.get("label", "Unknown")

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Get color for this label
            color = self.get_label_color(label)

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(
                x1,
                y1 - 5,
                label,
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

    def create_legend(self, fig, bboxes: List[Dict]):
        """Create a legend showing label colors and counts."""
        # Count labels
        label_counts = {}
        for bbox_info in bboxes:
            label = bbox_info.get("label", "Unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

        # Create legend elements
        legend_elements = []
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            color = self.get_label_color(label)
            legend_elements.append(
                patches.Patch(color=color, label=f"{label} ({count})")
            )

        # Place legend outside the plot area
        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="center right",
                bbox_to_anchor=(0.98, 0.5),
                fontsize=10,
                framealpha=0.9,
            )

    def visualize_single_image(
        self, image_path: str, annotation_path: str, output_dir: str, lang: str = "zh"
    ) -> bool:
        """
        Visualize a single image with its raw annotations.

        Args:
            image_path: Path to the image file
            annotation_path: Path to the annotation JSON file
            output_dir: Output directory for visualization
            lang: Language preference for labels

        Returns:
            True if successful, False otherwise
        """
        # Load image
        image_array, image_size = self.load_image_safe(image_path)
        if image_array is None:
            return False

        # Load annotation
        annotation_data = self.load_raw_annotation(annotation_path)
        if annotation_data is None:
            return False

        # Extract bboxes
        bboxes = self.extract_bboxes_from_raw(annotation_data, lang)
        if not bboxes:
            logger.warning(f"No bboxes found in {annotation_path}")
            return False

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Draw image with bboxes
        image_name = Path(image_path).name
        self.draw_bboxes_on_axis(
            ax,
            image_array,
            bboxes,
            f"Raw Annotations: {image_name} ({len(bboxes)} objects)",
        )

        # Create legend
        self.create_legend(fig, bboxes)

        # Set overall title
        fig.suptitle(
            f"Raw Data Visualization: {image_name}", fontsize=16, fontweight="bold"
        )

        # Adjust layout to accommodate legend
        plt.subplots_adjust(right=0.75)

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{Path(image_name).stem}_raw_visualization.png"
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Saved visualization: {output_path}")
        return True

    def visualize_batch(
        self,
        image_annotation_pairs: List[Tuple[str, str]],
        output_dir: str,
        lang: str = "zh",
    ) -> int:
        """
        Visualize multiple image-annotation pairs.

        Args:
            image_annotation_pairs: List of (image_path, annotation_path) tuples
            output_dir: Output directory for visualizations
            lang: Language preference for labels

        Returns:
            Number of successful visualizations
        """
        success_count = 0

        for i, (image_path, annotation_path) in enumerate(image_annotation_pairs):
            logger.info(
                f"Processing {i + 1}/{len(image_annotation_pairs)}: {image_path}"
            )

            if self.visualize_single_image(
                image_path, annotation_path, output_dir, lang
            ):
                success_count += 1

        return success_count


def auto_detect_annotation_path(image_path: str, base_dir: str = ".") -> Optional[str]:
    """Auto-detect annotation file path for a given image path."""
    image_path_obj = Path(image_path)
    image_name = image_path_obj.stem  # e.g., "QC-20230216-0000243_120932"
    image_dir = image_path_obj.parent  # e.g., "ds"

    # Look for JSON file with same basename in same directory
    annotation_candidates = [
        f"{image_name}.json",  # Same basename
        image_dir / f"{image_name}.json",  # Same directory
        f"annotations/{image_name}.json",
        f"labels/{image_name}.json",
    ]

    for candidate in annotation_candidates:
        candidate_path = Path(base_dir) / candidate
        if candidate_path.exists():
            return str(candidate)

    return None


def main():
    """Main entry point using configuration from top of file."""
    print("🚀 Raw Data Visualization Tool")
    print(f"📄 Processing {len(IMAGE_PATHS)} images")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Language: {'Chinese' if LANGUAGE == 'zh' else 'English'}")
    print(f"📂 Base directory: {BASE_DIR}")

    # Prepare image-annotation pairs
    image_annotation_pairs = []

    # Determine which images to process
    images_to_process = IMAGE_PATHS[:MAX_IMAGES] if MAX_IMAGES > 0 else IMAGE_PATHS

    if ANNOTATION_PATHS:
        # Use provided annotation paths
        if len(ANNOTATION_PATHS) != len(images_to_process):
            logger.error("Number of images and annotations must match")
            return 1

        image_annotation_pairs = list(zip(images_to_process, ANNOTATION_PATHS))
    else:
        # Auto-detect annotation files
        for image_path in images_to_process:
            annotation_path = auto_detect_annotation_path(image_path, BASE_DIR)

            if annotation_path:
                image_annotation_pairs.append((image_path, annotation_path))
            else:
                logger.warning(f"No annotation file found for {image_path}")

    if not image_annotation_pairs:
        logger.error("No valid image-annotation pairs found")
        return 1

    # Initialize visualizer
    visualizer = RawDataVisualizer(BASE_DIR)

    # Visualize samples
    success_count = visualizer.visualize_batch(
        image_annotation_pairs, OUTPUT_DIR, LANGUAGE
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RAW DATA VISUALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total image-annotation pairs: {len(image_annotation_pairs)}")
    print(f"✅ Successful visualizations: {success_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Language: {'Chinese' if LANGUAGE == 'zh' else 'English'}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
