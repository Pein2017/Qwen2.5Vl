#!/usr/bin/env python3
"""
Visualize the latest JSONL data format with multi-geometry support.

This script loads JSONL annotation files and visualizes different geometry types:
- bbox_2d: Traditional rectangular bounding boxes [x1, y1, x2, y2]
- squar    def get_main_category(self, description: str) -> str:
        Extract main category from description (part before first comma).
        return description.split(",")[0].strip() if description else "Unknown"

    def draw_objects_on_axis(
        self, ax, image_array: np.ndarray, objects: List[Dict], title: str
    ):
        Draw multi-geometry objects on a matplotlib axis.
        ax.imshow(image_array)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # First pass: collect all unique main categories
        for obj in objects:
            description = obj.get("description", "Unknown")
            main_category = self.get_main_category(description)
            self.unique_descriptions.add(main_category)
            if main_category not in self.description_colors:
                color_idx = len(self.description_colors) % len(self.label_color_palette)
                self.description_colors[main_category] = self.label_color_palette[color_idx]ls (四边形) [x1, y1, x2, y2, x3, y3, x4, y4]
- line: Line segments (线段) [x1, y1, x2, y2, x3, y3, ...]

Configure the settings below and run the script directly.
"""

# =============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# JSONL file path to visualize (relative to BASE_DIR)
JSONL_PATH = "temp_invalid.jsonl"

# Output directory for visualizations
OUTPUT_DIR = "invalid_visualizations"

# Base directory for resolving relative paths
BASE_DIR = "."

# Maximum number of samples to process (-1 for all samples)
MAX_SAMPLES = 10

# Show different geometry types with different colors
SHOW_GEOMETRY_LEGEND = True

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

# Configure UTF-8 encoding
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image


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


class MultiGeometryVisualizer:
    """Visualizes JSONL data with multi-geometry support (bbox_2d, square, line)."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)

        # Track unique descriptions and their colors
        self.unique_descriptions = set()
        self.description_colors = {}

        # Geometry-specific styles (for shape outlines only)
        self.geometry_styles = {
            "bbox_2d": {"linewidth": 2, "linestyle": "-"},
            "square": {"linewidth": 2, "linestyle": "--"},
            "line": {"linewidth": 3, "linestyle": "-"},
        }

        # Label color palette for different descriptions
        self.label_color_palette = [
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
            "#FFB6C1",
            "#98FB98",
            "#87CEEB",
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
                self.base_dir / "data" / "ds_v2_full" / image_path,
                self.base_dir
                / "data"
                / "ds_v2_full"
                / "images"
                / Path(image_path).name,
                self.base_dir / "data" / "ds_v2" / image_path,
                self.base_dir / "data" / "ds_v2" / "images" / Path(image_path).name,
                self.base_dir / "images" / Path(image_path).name,
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

    def load_jsonl_file(self, jsonl_path: str) -> List[Dict]:
        """Load JSONL file and return list of samples."""
        try:
            abs_path = (
                self.base_dir / jsonl_path
                if not Path(jsonl_path).is_absolute()
                else Path(jsonl_path)
            )

            if not abs_path.exists():
                logger.error(f"JSONL file not found: {abs_path}")
                return []

            samples = []
            with open(abs_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

            logger.info(f"Loaded {len(samples)} samples from: {abs_path}")
            return samples

        except Exception as e:
            logger.error(f"Failed to load JSONL {jsonl_path}: {e}")
            return []

    def extract_objects_from_sample(self, sample: Dict) -> List[Dict]:
        """
        Extract objects from JSONL sample format.

        Args:
            sample: JSONL sample with 'objects' field containing multi-geometry annotations

        Returns:
            List of object dictionaries with geometry and description
        """
        objects = []

        if "objects" not in sample:
            logger.warning("No 'objects' field found in sample")
            return objects

        for obj in sample["objects"]:
            try:
                # Extract geometry information
                geometry_info = {}
                geometry_type = None

                # Check for different geometry types
                if "bbox_2d" in obj:
                    geometry_type = "bbox_2d"
                    geometry_info["bbox_2d"] = obj["bbox_2d"]
                elif "square" in obj:
                    geometry_type = "square"
                    geometry_info["square"] = obj["square"]
                elif "line" in obj:
                    geometry_type = "line"
                    geometry_info["line"] = obj["line"]
                else:
                    logger.warning(f"Unknown geometry type in object: {obj}")
                    continue

                # Extract description
                description = obj.get("desc", "Unknown")

                # Create object entry
                obj_entry = {
                    "geometry_type": geometry_type,
                    "description": description,
                    **geometry_info,
                }

                objects.append(obj_entry)

            except Exception as e:
                logger.warning(f"Failed to process object: {e}")
                continue

        return objects

    def get_label_color(self, label: str) -> str:
        """Get consistent color for a label."""
        if label not in self.label_to_color:
            # Assign new color
            color_idx = len(self.label_to_color) % len(self.label_color_palette)
            self.label_to_color[label] = self.label_color_palette[color_idx]
        return self.label_to_color[label]

    def draw_objects_on_axis(
        self, ax, image_array: np.ndarray, objects: List[Dict], title: str
    ):
        """Draw multi-geometry objects on a matplotlib axis."""
        ax.imshow(image_array)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # First pass: collect all unique descriptions
        for obj in objects:
            description = obj.get("description", "Unknown")
            self.unique_descriptions.add(description)
            if description not in self.description_colors:
                color_idx = len(self.description_colors) % len(self.label_color_palette)
                self.description_colors[description] = self.label_color_palette[
                    color_idx
                ]

        # Second pass: draw objects with consistent colors
        for obj in objects:
            geometry_type = obj.get("geometry_type", "unknown")
            description = obj.get("description", "Unknown")
            color = self.description_colors[description]

            # Draw based on geometry type with consistent colors
            if geometry_type == "bbox_2d":
                self._draw_bbox_2d(ax, obj, color)
            elif geometry_type == "square":
                self._draw_square(ax, obj, color)
            elif geometry_type == "line":
                self._draw_line(ax, obj, color)

    def _draw_bbox_2d(self, ax, obj: Dict, color: str):
        """Draw rectangular bounding box."""
        bbox = obj.get("bbox_2d", [])
        if len(bbox) != 4:
            return

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Draw rectangle with geometry-specific style
        style = self.geometry_styles["bbox_2d"]
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            edgecolor=color,
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

    def _draw_square(self, ax, obj: Dict, color: str):
        """Draw quadrilateral (四边形)."""
        square = obj.get("square", [])
        if len(square) != 8:
            return

        # Convert to coordinate pairs
        coords = [(square[i], square[i + 1]) for i in range(0, 8, 2)]

        # Create polygon with geometry-specific style
        style = self.geometry_styles["square"]
        polygon = patches.Polygon(
            coords,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            edgecolor=color,
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(polygon)

    def _draw_line(self, ax, obj: Dict, color: str):
        """Draw line segment (线段)."""
        line = obj.get("line", [])
        if len(line) < 4 or len(line) % 2 != 0:
            return

        # Convert to coordinate pairs
        coords = [(line[i], line[i + 1]) for i in range(0, len(line), 2)]

        # Extract x and y coordinates
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        # Draw line with geometry-specific style
        style = self.geometry_styles["line"]
        ax.plot(
            x_coords,
            y_coords,
            color=color,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=0.8,
            marker="o",
            markersize=4,
        )

    def create_legend(
        self, fig, objects: List[Dict], show_geometry_legend: bool = True
    ):
        """Create a legend showing descriptions and geometry types."""
        legend_elements = []

        # Group objects by description and count occurrences
        description_counts = {}
        for obj in objects:
            description = obj.get("description", "Unknown")
            description_counts[description] = description_counts.get(description, 0) + 1

        # Create legend elements for descriptions
        for description in sorted(description_counts.keys()):
            count = description_counts[description]
            color = self.description_colors[description]
            legend_elements.append(
                patches.Patch(
                    facecolor=color,
                    alpha=0.3,
                    edgecolor=color,
                    label=f"{description} ({count})",
                )
            )

        if show_geometry_legend:
            # Add geometry type indicators
            geometry_labels = {
                "bbox_2d": "矩形 (Rectangle)",
                "square": "四边形 (Quadrilateral)",
                "line": "线段 (Line Segment)",
            }

            # Add separator in legend
            legend_elements.append(patches.Patch(color="none", label=""))

            # Add geometry style indicators
            for geometry_type, label in geometry_labels.items():
                style = self.geometry_styles[geometry_type]
                legend_elements.append(
                    patches.Patch(
                        facecolor="none",
                        edgecolor="gray",
                        linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        label=label,
                    )
                )

        # Place legend outside the plot area
        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc="center right",
                bbox_to_anchor=(0.98, 0.5),
                fontsize=10,
                framealpha=0.9,
                title="Objects & Geometry Types",
            )

    def visualize_single_sample(
        self, sample: Dict, output_dir: str, sample_idx: int = 0
    ) -> bool:
        """
        Visualize a single JSONL sample with its multi-geometry annotations.

        Args:
            sample: JSONL sample dictionary
            output_dir: Output directory for visualization
            sample_idx: Sample index for naming

        Returns:
            True if successful, False otherwise
        """
        # Extract image path
        images = sample.get("images", [])
        if not images:
            logger.warning("No images found in sample")
            return False

        image_path = images[0]  # Use first image

        # Load image
        image_array, _ = self.load_image_safe(image_path)
        if image_array is None:
            return False

        # Extract objects
        objects = self.extract_objects_from_sample(sample)
        if not objects:
            logger.warning(f"No objects found in sample")
            return False

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Draw image with objects
        image_name = Path(image_path).name
        self.draw_objects_on_axis(
            ax,
            image_array,
            objects,
            f"Multi-Geometry Annotations: {image_name} ({len(objects)} objects)",
        )

        # Create legend
        self.create_legend(fig, objects, SHOW_GEOMETRY_LEGEND)

        # Set overall title
        fig.suptitle(
            f"JSONL Data Visualization: {image_name}", fontsize=16, fontweight="bold"
        )

        # Adjust layout to accommodate legend
        plt.subplots_adjust(right=0.75)

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_filename = (
            f"sample_{sample_idx:03d}_{Path(image_name).stem}_visualization.png"
        )
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Saved visualization: {output_path}")
        return True

    def visualize_jsonl_batch(
        self, samples: List[Dict], output_dir: str, max_samples: int = -1
    ) -> int:
        """
        Visualize multiple JSONL samples.

        Args:
            samples: List of JSONL sample dictionaries
            output_dir: Output directory for visualizations
            max_samples: Maximum number of samples to process (-1 for all)

        Returns:
            Number of successful visualizations
        """
        success_count = 0

        # Limit samples if specified
        if max_samples > 0:
            samples = samples[:max_samples]

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i + 1}/{len(samples)}")

            if self.visualize_single_sample(sample, output_dir, i):
                success_count += 1

        return success_count


def find_jsonl_files(base_dir: str = ".") -> List[str]:
    """Find JSONL files in the base directory."""
    base_path = Path(base_dir)
    jsonl_files = []

    # Look for JSONL files in common locations
    search_patterns = ["*.jsonl", "data/**/*.jsonl", "**/*.jsonl"]

    for pattern in search_patterns:
        for jsonl_file in base_path.glob(pattern):
            if jsonl_file.is_file():
                jsonl_files.append(str(jsonl_file.relative_to(base_path)))

    return sorted(list(set(jsonl_files)))  # Remove duplicates and sort


def main():
    """Main entry point using configuration from top of file."""
    print("🚀 Multi-Geometry JSONL Visualization Tool")
    print(f"📄 JSONL file: {JSONL_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🔢 Max samples: {'All' if MAX_SAMPLES == -1 else MAX_SAMPLES}")

    # Initialize visualizer
    visualizer = MultiGeometryVisualizer(BASE_DIR)

    # Load JSONL file
    jsonl_file_path = JSONL_PATH
    if not Path(jsonl_file_path).is_absolute():
        # Try to find the JSONL file
        if not (Path(BASE_DIR) / jsonl_file_path).exists():
            print(f"⚠️  JSONL file not found at {jsonl_file_path}")
            print("🔍 Searching for JSONL files...")

            available_jsonl = find_jsonl_files(BASE_DIR)
            if available_jsonl:
                print("📋 Available JSONL files:")
                for i, jsonl_file in enumerate(available_jsonl):
                    print(f"   {i + 1}. {jsonl_file}")

                # Use the first one if JSONL_PATH matches any
                for jsonl_file in available_jsonl:
                    if Path(jsonl_file).name == Path(JSONL_PATH).name:
                        jsonl_file_path = jsonl_file
                        print(f"✅ Using: {jsonl_file_path}")
                        break
                else:
                    # Use the first available one
                    jsonl_file_path = available_jsonl[0]
                    print(f"✅ Using first available: {jsonl_file_path}")
            else:
                logger.error("No JSONL files found")
                return 1

    samples = visualizer.load_jsonl_file(jsonl_file_path)
    if not samples:
        logger.error("No samples loaded from JSONL file")
        return 1

    print(f"📊 Loaded {len(samples)} samples")

    # Visualize samples
    success_count = visualizer.visualize_jsonl_batch(samples, OUTPUT_DIR, MAX_SAMPLES)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"MULTI-GEOMETRY VISUALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(samples)}")
    print(
        f"Processed samples: {min(len(samples), MAX_SAMPLES) if MAX_SAMPLES > 0 else len(samples)}"
    )
    print(f"✅ Successful visualizations: {success_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎨 Geometry legend: {'Enabled' if SHOW_GEOMETRY_LEGEND else 'Disabled'}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
