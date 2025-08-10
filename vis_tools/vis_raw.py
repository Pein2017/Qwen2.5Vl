#!/usr/bin/env python3
"""
Visualize the latest JSONL data format with multi-geometry support.

This script loads JSONL annotation files and visualizes different geometry types:
- bbox_2d: Traditional rectangular bounding boxes [x1, y1, x2, y2]
- quad: Quadrilaterals (四边形) [x1, y1, x2, y2, x3, y3, x4, y4]
- line: Line segments (线段) [x1, y1, x2, y2, x3, y3, ...]

Configure the settings below and run the script directly.
"""

# =============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

# JSONL file path to visualize (relative to BASE_DIR)
JSONL_PATH = "data/ds_v2_bbu_bbu_shield/all_samples.jsonl"

# Output directory for visualizations
OUTPUT_DIR = "vis_raw_data"

# Base directory for resolving relative paths
BASE_DIR = "."

# Maximum number of samples to process (-1 for all samples)
MAX_SAMPLES = 50

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
from typing import Dict, List, Tuple

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
    """Visualizes JSONL data with multi-geometry support (bbox_2d, quad, line)."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)

        # Track unique descriptions and their colors
        self.unique_descriptions = set()
        self.description_colors = {}

        # Geometry-specific styles (for shape outlines only)
        self.geometry_styles = {
            "bbox_2d": {"linewidth": 2, "linestyle": "-"},
            "quad": {"linewidth": 2, "linestyle": "--"},
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

    def get_main_category(self, description: str) -> str:
        """Extract main category from description (part before first comma)."""
        return description.split(",")[0].strip() if description else "Unknown"

    def load_image_safe(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load image and return array and dimensions. Handles path resolution."""
        try:
            # First try direct path
            abs_path = Path(image_path)

            # If not found, try with base_dir
            if not abs_path.exists():
                abs_path = self.base_dir / image_path

            # If still not found, try with data/ds_v2_full/images/ prefix
            if not abs_path.exists():
                # Get just the filename from the path
                image_filename = Path(image_path).name
                # Try with data/ds_v2_full/images/ prefix
                abs_path = (
                    self.base_dir / "data" / "ds_v2_full" / "images" / image_filename
                )

            # If still not found, raise error
            if not abs_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with Image.open(abs_path) as img:
                img_rgb = img.convert("RGB")
                img_array = np.array(img_rgb)
                logger.info(f"Successfully loaded image from: {abs_path}")
                return img_array, img_rgb.size  # (width, height)

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise  # Re-raise the exception

    def load_jsonl_file(self, jsonl_path: str) -> List[Dict]:
        """Load JSONL file and return list of samples. Raises error if file not found."""
        try:
            abs_path = (
                self.base_dir / jsonl_path
                if not Path(jsonl_path).is_absolute()
                else Path(jsonl_path)
            )

            if not abs_path.exists():
                raise FileNotFoundError(f"JSONL file not found: {abs_path}")

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
                        raise  # Re-raise the exception

            logger.info(f"Loaded {len(samples)} samples from: {abs_path}")
            return samples

        except Exception as e:
            logger.error(f"Failed to load JSONL {jsonl_path}: {e}")
            raise  # Re-raise the exception instead of returning empty list

    def extract_objects_from_sample(self, sample: Dict) -> List[Dict]:
        """
        Extract objects from JSONL sample format.

        Args:
            sample: JSONL sample with 'objects' field containing multi-geometry annotations

        Returns:
            List of object dictionaries with geometry and description

        Raises:
            ValueError: If no 'objects' field found or objects can't be processed
        """
        objects = []

        if "objects" not in sample:
            raise ValueError("No 'objects' field found in sample")

        for obj in sample["objects"]:
            try:
                # Extract geometry information
                geometry_info = {}
                geometry_type = None

                # Strictly support only the following geometry types
                if "bbox_2d" in obj:
                    geometry_type = "bbox_2d"
                    geometry_info["bbox_2d"] = obj["bbox_2d"]
                elif "quad" in obj:
                    geometry_type = "quad"
                    geometry_info["quad"] = obj["quad"]
                elif "line" in obj:
                    geometry_type = "line"
                    geometry_info["line"] = obj["line"]
                else:
                    raise ValueError(
                        f"Unsupported geometry type. Expected one of ['bbox_2d','quad','line'], got keys: {list(obj.keys())}"
                    )

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
                logger.error(f"Failed to process object: {e}")
                raise  # Re-raise the exception

        if not objects:
            raise ValueError("No valid objects found in sample")

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
            main_category = self.get_main_category(description)
            self.unique_descriptions.add(main_category)
            if main_category not in self.description_colors:
                color_idx = len(self.description_colors) % len(self.label_color_palette)
                self.description_colors[main_category] = self.label_color_palette[
                    color_idx
                ]

        # Second pass: draw objects with consistent colors
        for obj in objects:
            geometry_type = obj.get("geometry_type", "unknown")
            description = obj.get("description", "Unknown")
            main_category = self.get_main_category(description)
            color = self.description_colors[main_category]

            # Draw based on geometry type with consistent colors
            if geometry_type == "bbox_2d":
                self._draw_bbox_2d(ax, obj, color)
            elif geometry_type == "quad":
                self._draw_quad(ax, obj, color)
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

    def _canonical_quad_ordering(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Apply canonical clockwise ordering starting from top-left vertex.

        **VISUALIZATION-SPECIFIC**: This function ensures proper quadrilateral display
        by using a robust geometric approach that handles incorrectly ordered input data.

        Unlike the data processing pipeline which assumes input data is already reasonable,
        the visualization must handle arbitrary coordinate orders that may cause crossed lines.

        Args:
            points: List of (x, y) vertex coordinates in any order

        Returns:
            Points ordered for proper polygon display: [tl, tr, br, bl] (clockwise)
        """
        if len(points) != 4:
            logger.warning(
                f"Quad must have exactly 4 points, got {len(points)}: {points}"
            )
            return [(int(p[0]), int(p[1])) for p in points]

        # Use robust corner detection based on coordinate extremes
        # This approach is more reliable for visualization than the geometric method
        pts = np.array(points, dtype="float32")

        # Calculate centroid for reference
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])

        # Classify points by their position relative to centroid
        def classify_corner(point):
            x, y = point
            # Determine quadrant relative to centroid
            if x <= cx and y <= cy:
                return (0, -(x + y))  # Top-left: minimize x+y
            elif x >= cx and y <= cy:
                return (1, x - y)  # Top-right: maximize x-y
            elif x >= cx and y >= cy:
                return (2, x + y)  # Bottom-right: maximize x+y
            else:  # x <= cx and y >= cy
                return (3, -x + y)  # Bottom-left: maximize -x+y

        # Sort points by corner classification
        sorted_points = sorted(points, key=classify_corner)

        # Ensure we have exactly one point in each quadrant
        # If not, fall back to simple coordinate-based ordering
        if len(set(classify_corner(p)[0] for p in sorted_points)) != 4:
            logger.debug("Fallback to coordinate-based ordering")
            # Simple fallback: sort by y first (top to bottom), then by x within each row
            sorted_by_y = sorted(points, key=lambda p: p[1])
            top_points = sorted(
                sorted_by_y[:2], key=lambda p: p[0]
            )  # Top row, left to right
            bottom_points = sorted(
                sorted_by_y[2:], key=lambda p: p[0]
            )  # Bottom row, left to right
            sorted_points = [
                top_points[0],
                top_points[1],
                bottom_points[1],
                bottom_points[0],
            ]

        return [(int(p[0]), int(p[1])) for p in sorted_points]

    def _extract_geometry_coordinates(self, geometry: Dict) -> Tuple[str, List[float]]:
        """
        Extract coordinates from legacy geometry format (ds_v1 style).

        Args:
            geometry: Geometry object with 'type' and 'coordinates' fields

        Returns:
            Tuple of (geometry_type, coordinates_list)
        """
        geometry_type_map = {
            "Square": "quad",  # Legacy "Square" type maps to "quad"
            "Quad": "quad",
            "LineString": "line",
            "ExtentPolygon": "quad",  # Treat as quad for visualization
        }

        geom_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])

        if not coordinates:
            logger.warning(f"Empty coordinates in geometry type: {geom_type}")
            return "bbox_2d", [0, 0, 1, 1]

        # Map geometry type to visualization type
        vis_type = geometry_type_map.get(geom_type, "quad")

        if vis_type == "quad":
            # Extract quad coordinates from nested structure
            # Handle format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x1,y1]]]
            if coordinates and isinstance(coordinates[0], list):
                points = (
                    coordinates[0]
                    if isinstance(coordinates[0][0], list)
                    else coordinates
                )
            else:
                points = coordinates

            # Extract first 4 points and flatten to [x1,y1,x2,y2,x3,y3,x4,y4]
            quad_coords = []
            for i, point in enumerate(points[:4]):  # Take first 4 points
                if isinstance(point, list) and len(point) >= 2:
                    quad_coords.extend([float(point[0]), float(point[1])])

            if len(quad_coords) == 8:
                return "quad", quad_coords
            else:
                logger.warning(
                    f"Invalid quad coordinates: expected 8 values, got {len(quad_coords)}"
                )
                return "bbox_2d", [0, 0, 1, 1]

        elif vis_type == "line":
            # Extract line coordinates and flatten
            line_coords = []
            for point in coordinates:
                if isinstance(point, list) and len(point) >= 2:
                    line_coords.extend([float(point[0]), float(point[1])])
            return "line", line_coords

        # Fallback to bbox
        logger.warning(f"Unsupported geometry type: {geom_type}, falling back to bbox")
        return "bbox_2d", [0, 0, 1, 1]

    def _draw_quad(self, ax, obj: Dict, color: str):
        """Draw quadrilateral (四边形) with model-centric coordinate ordering."""
        quad = obj.get("quad", [])
        if len(quad) != 8:
            return

        # Convert to coordinate pairs
        raw_coords = [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]

        # Apply canonical quad ordering to exactly match data conversion pipeline
        # This ensures visualization shows what the model sees (no coordinate reordering)
        ordered_coords = self._canonical_quad_ordering(raw_coords)

        # Create polygon with geometry-specific style using ordered coordinates
        style = self.geometry_styles["quad"]
        polygon = patches.Polygon(
            ordered_coords,
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
            main_category = self.get_main_category(description)
            description_counts[main_category] = (
                description_counts.get(main_category, 0) + 1
            )

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
                "quad": "四边形 (Quadrilateral)",
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
            True if successful

        Raises:
            ValueError: If no images found in sample or other visualization errors
        """
        # Extract image path
        images = sample.get("images", [])
        if not images:
            raise ValueError("No images found in sample")

        image_path = images[0]  # Use first image

        # Load image - will raise FileNotFoundError if image not found
        image_array, _ = self.load_image_safe(image_path)

        # Extract objects - will raise ValueError if no objects found
        objects = self.extract_objects_from_sample(sample)

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

        Raises:
            Exception: If any sample fails to visualize
        """
        success_count = 0

        # Limit samples if specified
        if max_samples > 0:
            samples = samples[:max_samples]

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i + 1}/{len(samples)}")
            # Will raise exception if visualization fails
            self.visualize_single_sample(sample, output_dir, i)
            success_count += 1

        return success_count


def main():
    """Main entry point using configuration from top of file."""
    print("🚀 Multi-Geometry JSONL Visualization Tool")
    print(f"📄 JSONL file: {JSONL_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🔢 Max samples: {'All' if MAX_SAMPLES == -1 else MAX_SAMPLES}")

    # Initialize visualizer
    visualizer = MultiGeometryVisualizer(BASE_DIR)

    # Load JSONL file - will raise error if not found
    jsonl_file_path = JSONL_PATH

    try:
        samples = visualizer.load_jsonl_file(jsonl_file_path)
        print(f"📊 Loaded {len(samples)} samples")

        # Visualize samples - will raise error if visualization fails
        success_count = visualizer.visualize_jsonl_batch(
            samples, OUTPUT_DIR, MAX_SAMPLES
        )

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
        print(
            f"🎨 Geometry legend: {'Enabled' if SHOW_GEOMETRY_LEGEND else 'Disabled'}"
        )
        print(f"{'=' * 60}")

        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
