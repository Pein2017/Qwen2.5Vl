#!/usr/bin/env python3
"""
Visualization script for Qwen2.5-VL model inference results.
Creates side-by-side comparisons of ground truth vs predictions with different colors for different labels.
Supports multiple geometry types: bbox_2d (rectangles), quad (quadrilaterals), and line segments.

Input format: JSONL file where each line contains a JSON object with sample data.
"""

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# =============================================================================
INPUT_FILE = "infer_results/output-814-base-ep50/val/inference/predictions.jsonl"  # JSONL format: one JSON object per line
OUTPUT_DIR = "vis_generation/814-base-ep50-val"
BASE_PATH = "."
# Data root directory for image files
DATA_ROOT_DIR = "data/ds_v2_bbu_bbu_shield/images"
MAX_SAMPLES = None  # Set to a number to limit samples, or None for all
SAMPLE_INDICES: Optional[List[int]] = (
    None  # Set to list of indices [0, 1, 2, 5] or None for all
)
# Show different geometry types with different colors and styles
SHOW_GEOMETRY_LEGEND = True
# Visualization mode: 'split' for side-by-side GT/Pred, 'overlay' for single image overlay
VIEW_MODE = "split"
#
# Examples:
# - Process all samples: SAMPLE_INDICES = None, MAX_SAMPLES = None
# - Process first 10 samples: SAMPLE_INDICES = None, MAX_SAMPLES = 10
# - Process specific samples: SAMPLE_INDICES = [0, 5, 10, 15], MAX_SAMPLES = None
# =============================================================================

shutil.rmtree(matplotlib.get_cachedir())
# Configure logging - reduce verbosity and suppress font warnings
logging.basicConfig(level=logging.ERROR, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress matplotlib font warnings
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# Configure Chinese font for matplotlib
font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
font_prop = FontProperties(fname=font_path)

# Register and set global font
fontManager.addfont(font_path)
rcParams["font.sans-serif"] = [font_prop.get_name()]
rcParams["axes.unicode_minus"] = False  # 解决负号 "-" 显示为方块的问题


def generate_colors(labels: List[str]) -> Dict[str, str]:
    """
    Generate distinct colors for each unique label using a deterministic approach.

    Args:
        labels: List of unique label strings

    Returns:
        Dictionary mapping label to hex color
    """
    # Use a more comprehensive color palette
    base_colors = [
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
        "#FEF9E7",
        "#EAEDED",
        "#E8F8F5",
        "#FDF2E9",
        "#EBF5FB",
        "#E9F7EF",
        "#FEF5E7",
        "#FDEBD0",
        "#EBDEF0",
        "#D0ECE7",
    ]

    color_map = {}
    for i, label in enumerate(sorted(labels)):  # Sort for consistency
        if i < len(base_colors):
            color_map[label] = base_colors[i]
        else:
            # Generate additional colors using hash for consistency
            hash_obj = hashlib.md5(label.encode())
            hash_hex = hash_obj.hexdigest()
            color = f"#{hash_hex[:6]}"
            color_map[label] = color

    return color_map


# Geometry-specific styles for shape outlines (only supported geometries)
GEOMETRY_STYLES = {
    "bbox_2d": {"linewidth": 2, "linestyle": "-"},
    "quad": {"linewidth": 2, "linestyle": "--"},
    "line": {"linewidth": 3, "linestyle": "-"},
}


def _canonical_quad_ordering(
    points: List[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """
    Order four points to a canonical clockwise order starting from top-left.

    This mirrors the ordering logic used in vis_raw.py so that visualization of
    quads is consistent across tools and robust to arbitrary input point order.
    """
    if len(points) != 4:
        return [(int(p[0]), int(p[1])) for p in points]

    pts = np.array(points, dtype="float32")

    # Centroid
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    def classify_corner(point: Tuple[float, float]) -> Tuple[int, float]:
        x, y = point
        if x <= cx and y <= cy:
            return (0, -(x + y))  # top-left: minimize x+y
        elif x >= cx and y <= cy:
            return (1, x - y)  # top-right: maximize x-y
        elif x >= cx and y >= cy:
            return (2, x + y)  # bottom-right: maximize x+y
        else:  # x <= cx and y >= cy
            return (3, -x + y)  # bottom-left: maximize -x+y

    sorted_points = sorted(points, key=classify_corner)

    # Ensure distinct corners; if not, fallback ordering by rows
    if len(set(classify_corner(p)[0] for p in sorted_points)) != 4:
        sorted_by_y = sorted(points, key=lambda p: p[1])
        top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])
        sorted_points = [
            top_points[0],
            top_points[1],
            bottom_points[1],
            bottom_points[0],
        ]

    return [(int(p[0]), int(p[1])) for p in sorted_points]


def parse_bbox_data(bbox_str: str) -> List[Dict[str, Any]]:
    """
    Parse bbox data from JSON string format. If the JSON string is
    truncated (common when model output is cut off), attempt to recover
    by iteratively removing the last object until a valid JSON list is
    obtained.

    Args:
        bbox_str: JSON string or already-parsed Python list.

    Returns:
        List of dictionaries with keys like ``bbox_2d``, ``square``, ``line`` and ``label``.
    """
    # Early exit if data already provided as list
    if isinstance(bbox_str, list):
        return bbox_str

    if not isinstance(bbox_str, str):
        return []

    raw = bbox_str.strip()

    # Fast path: try direct JSON loading first
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON decode failed: {e}")

    # ------------------------------------------------------------------
    # Fallback: extract individual objects using brace matching. This is
    # much faster than repeatedly slicing & re-parsing the whole string.
    # ------------------------------------------------------------------
    objs: List[Dict[str, Any]] = []
    depth = 0
    start_idx = None
    for idx, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                segment = raw[start_idx : idx + 1]
                try:
                    obj = json.loads(segment)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    # Skip malformed individual object
                    pass

    if objs:
        logger.warning(
            "Recovered truncated JSON by collecting %d complete objects", len(objs)
        )
    else:
        logger.warning("Failed to recover truncated bbox JSON string.")

    return objs


def load_image_safe(
    image_path: str, base_path: str = "."
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Safely load image with fallback options.

    Args:
        image_path: Path to image file
        base_path: Base directory to prepend if image_path is relative

    Returns:
        Tuple of (image_array, success_flag)
    """
    # Extract the image filename from the path
    image_filename = os.path.basename(image_path)

    # Try different path combinations including data root dir
    path_candidates = [
        image_path,
        os.path.join(base_path, image_path),
        os.path.join(base_path, DATA_ROOT_DIR, image_filename),
        os.path.join(".", image_path),
        os.path.join(".", DATA_ROOT_DIR, image_filename),
        os.path.join("..", image_path),
        os.path.join("..", DATA_ROOT_DIR, image_filename),
    ]

    for path in path_candidates:
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                return np.array(img), True
            except Exception as e:
                logger.warning(f"Failed to load image from {path}: {e}")
                continue

    logger.error(f"Could not load image from any path: {path_candidates}")
    return None, False


def determine_geometry_type(item: Dict[str, Any]) -> str:
    """
    Determine the geometry type of an item based on available keys.

    Args:
        item: Dictionary containing geometry information

    Returns:
        String identifying the geometry type: 'bbox_2d', 'quad', or 'line'

    Raises:
        ValueError: If geometry type is unknown or unsupported
    """
    if "bbox_2d" in item:
        return "bbox_2d"
    elif "quad" in item:
        return "quad"
    elif "line" in item:
        return "line"
    else:
        raise ValueError(
            f"Unsupported geometry type. Expected one of ['bbox_2d','quad','line'], got keys: {list(item.keys())}"
        )


def generate_label_styles(labels: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Deterministically assign a linestyle and marker for each label so that
    GT and Pred for the same label use identical visual styles.

    Returns:
        Mapping: label -> {"linestyle": str, "marker": str}
    """
    linestyle_options = ["-", "--", ":", "-."]
    marker_options = [
        "o",
        "s",
        "^",
        "v",
        "D",
        "*",
        "x",
        "+",
        "1",
        "2",
        "3",
        "4",
        "P",
        "X",
        "h",
        "H",
    ]

    styles: Dict[str, Dict[str, str]] = {}
    for idx, label in enumerate(sorted(labels)):
        linestyle = linestyle_options[idx % len(linestyle_options)]
        marker = marker_options[idx % len(marker_options)]
        styles[label] = {"linestyle": linestyle, "marker": marker}
    return styles


def draw_bbox_2d(ax, item: Dict[str, Any], color: str, linestyle: str) -> None:
    """
    Draw rectangular bounding box with a consistent style per label.

    Args:
        ax: Matplotlib axis
        item: Dictionary containing bbox_2d information
        color: Color to use for the box
        linestyle: Linestyle chosen per label
    """
    bbox = item.get("bbox_2d", [])
    if len(bbox) != 4:
        return

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    base = GEOMETRY_STYLES["bbox_2d"]
    linewidth = base["linewidth"]

    rect = Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=color,
        facecolor="none",
        alpha=0.9,
    )
    ax.add_patch(rect)

    # Text labels removed - will only show in legend


def draw_quad(ax, item: Dict[str, Any], color: str, linestyle: str) -> None:
    """
    Draw quadrilateral using 'quad' key (8 values: x1,y1,...,x4,y4) with a
    consistent style per label.

    Args:
        ax: Matplotlib axis
        item: Dictionary containing quad information
        color: Color to use for the polygon
        linestyle: Linestyle chosen per label
    """
    quad = item.get("quad", [])
    if len(quad) != 8:
        return

    # Convert to coordinate pairs
    raw_coords = [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]

    # Apply canonical clockwise ordering from top-left to align with vis_raw.py
    ordered_coords = _canonical_quad_ordering(raw_coords)

    base = GEOMETRY_STYLES["quad"]
    linewidth = base["linewidth"]

    polygon = Polygon(
        ordered_coords,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=color,
        facecolor="none",
        alpha=0.9,
    )
    ax.add_patch(polygon)

    # Text labels removed - will only show in legend


def draw_line(
    ax,
    item: Dict[str, Any],
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """
    Draw line segment with a consistent style per label.

    Args:
        ax: Matplotlib axis
        item: Dictionary containing line information
        color: Color to use for the line
        marker: Marker chosen per label
        linestyle: Linestyle chosen per label
    """
    line = item.get("line", [])
    if len(line) < 4 or len(line) % 2 != 0:
        return

    # Convert to coordinate pairs
    coords = [(line[i], line[i + 1]) for i in range(0, len(line), 2)]

    # Extract x and y coordinates
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    base = GEOMETRY_STYLES["line"]
    linewidth = base["linewidth"]

    ax.plot(
        x_coords,
        y_coords,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=0.9,
        marker=marker,
        markersize=4,
    )

    # Text labels removed - will only show in legend


def draw_bboxes(
    ax,
    bbox_data: List[Dict],
    color_map: Dict[str, str],
    style_map: Dict[str, Dict[str, str]],
    title: str,
) -> None:
    """
    Draw bounding boxes and other geometries on the given axis.

    Args:
        ax: Matplotlib axis
        bbox_data: List of geometry dictionaries
        color_map: Mapping from label to color
        style_map: Mapping from label to style (linestyle/marker)
        title: Title for the subplot
    """
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    for item in bbox_data:
        # Prefer 'desc' as the object label; fallback to legacy 'label'
        label = (
            item.get("desc")
            if isinstance(item.get("desc"), str)
            else str(item.get("label", "Unknown"))
        )

        color = color_map.get(label) or "#000000"
        if label in style_map:
            style = style_map[label]
        else:
            style = {"linestyle": "-", "marker": "o"}
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")

        geometry_type = determine_geometry_type(item)

        if geometry_type == "bbox_2d":
            draw_bbox_2d(ax, item, color, linestyle)
        elif geometry_type == "quad":
            draw_quad(ax, item, color, linestyle)
        elif geometry_type == "line":
            draw_line(ax, item, color, marker, linestyle)


def create_legend(
    fig,
    color_map: Dict[str, str],
    bbox_counts: Dict[str, List[int]],
    used_geometry_types: Optional[Set[str]] = None,
    style_map: Optional[Dict[str, Dict[str, str]]] = None,
):
    """
    Create a legend showing label colors and counts with consistent styles.
    Only shows labels that actually appear in the current visualization.

    Args:
        fig: Matplotlib figure
        color_map: Mapping from label to color
        bbox_counts: Mapping from label to [ground_truth_count, prediction_count]
        used_geometry_types: Set of geometry types used in this visualization
        style_map: Mapping from label to style (linestyle/marker)
    """
    legend_elements = []

    active_labels = [
        label for label, counts in bbox_counts.items() if counts[0] > 0 or counts[1] > 0
    ]

    active_labels.sort(key=lambda label: sum(bbox_counts[label]), reverse=True)

    for label in active_labels:
        counts = bbox_counts.get(label, [0, 0])
        gt_count, pred_count = counts[0], counts[1]

        if len(label) > 15:
            short_label = label[:12] + "..."
            legend_label = f"{short_label} ({gt_count}/{pred_count})"
        else:
            legend_label = f"{label} ({gt_count}/{pred_count})"

        linestyle = "-"
        if style_map is not None:
            if label in style_map and "linestyle" in style_map[label]:
                linestyle = style_map[label]["linestyle"]

        legend_elements.append(
            patches.Patch(
                facecolor="none",
                edgecolor=color_map[label],
                linestyle=linestyle,
                label=legend_label,
            )
        )

    if SHOW_GEOMETRY_LEGEND and used_geometry_types and len(used_geometry_types) > 0:
        valid_types = [t for t in used_geometry_types if t != "unknown"]

        if valid_types:
            legend_elements.append(patches.Patch(color="none", label=""))

            geometry_labels = {
                "bbox_2d": "Rectangle",
                "quad": "Quadrilateral",
                "line": "Line",
            }

            for geometry_type in valid_types:
                if geometry_type in geometry_labels:
                    style = GEOMETRY_STYLES[geometry_type]
                    legend_elements.append(
                        patches.Patch(
                            facecolor="none",
                            edgecolor="gray",
                            linewidth=style["linewidth"],
                            linestyle=style["linestyle"],
                            label=geometry_labels[geometry_type],
                        )
                    )

    n_elements = len(legend_elements)

    ncol = 1
    if n_elements > 20:
        ncol = 4
    elif n_elements > 15:
        ncol = 3
    elif n_elements > 8:
        ncol = 2

    legend = fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=10,
        framealpha=0.95,
        ncol=ncol,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.1,
        title="Object Categories (GT/Pred Counts)",
        title_fontsize=11,
    )

    legend._legend_box.align = "left"

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("lightgray")
    frame.set_linewidth(1)


def visualize_sample(
    sample_data: Dict,
    output_dir: str,
    color_map: Dict[str, str],
    style_map: Dict[str, Dict[str, str]],
    base_path: str = ".",
) -> bool:
    """
    Visualize a single sample. Supports two modes:
    - split: side-by-side GT and Pred subimages in one figure
    - overlay: draw GT and Pred overlays on a single image

    Args:
        sample_data: Dictionary containing 'image' (or 'images'), 'ground_truth', 'prediction'
        output_dir: Directory to save the visualization
        color_map: Pre-computed mapping from label to color (global consistency)
        style_map: Pre-computed mapping from label to style (global consistency)
        base_path: Base directory for image paths

    Returns:
        True if successful, False otherwise
    """
    # Extract data
    image_path = sample_data.get("image", "")
    if not image_path:
        images_list = sample_data.get("images", [])
        if isinstance(images_list, list) and len(images_list) > 0:
            image_path = images_list[0]

    ground_truth_raw = sample_data.get("ground_truth", [])
    pred_raw = sample_data.get("prediction", [])

    if not image_path:
        logger.warning(f"No image path found in sample data")
        return False

    image_array, success = load_image_safe(image_path, base_path)
    if not success:
        return False

    ground_truth_data = (
        parse_bbox_data(ground_truth_raw)
        if isinstance(ground_truth_raw, str)
        else ground_truth_raw
    )
    pred_result_data = (
        parse_bbox_data(pred_raw) if isinstance(pred_raw, str) else pred_raw
    )

    if not ground_truth_data and not pred_result_data:
        return False

    bbox_counts = defaultdict(lambda: [0, 0])
    used_geometry_types = set()

    for item in ground_truth_data:
        label = (
            item.get("desc")
            if isinstance(item.get("desc"), str)
            else str(item.get("label", "Unknown"))
        )
        bbox_counts[label][0] += 1
        geometry_type = determine_geometry_type(item)
        used_geometry_types.add(geometry_type)

    for item in pred_result_data:
        label = (
            item.get("desc")
            if isinstance(item.get("desc"), str)
            else str(item.get("label", "Unknown"))
        )
        bbox_counts[label][1] += 1
        geometry_type = determine_geometry_type(item)
        used_geometry_types.add(geometry_type)

    image_name = os.path.basename(image_path)

    if VIEW_MODE == "split":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        plt.subplots_adjust(wspace=0.02)

        ax1.imshow(image_array)
        draw_bboxes(
            ax1,
            ground_truth_data,
            color_map,
            style_map,
            f"Ground Truth ({len(ground_truth_data)} objects)",
        )

        ax2.imshow(image_array)
        draw_bboxes(
            ax2,
            pred_result_data,
            color_map,
            style_map,
            f"Predictions ({len(pred_result_data)} objects)",
        )

        fig.suptitle(
            f"Model Performance Comparison: {image_name}",
            fontsize=14,
            fontweight="bold",
        )
        create_legend(fig, color_map, bbox_counts, used_geometry_types, style_map)
        plt.tight_layout(rect=[0, 0, 0.95, 0.92])

        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(image_name)[0]}_comparison.png"
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return True

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image_array)
    ax.axis("off")

    draw_bboxes(
        ax,
        ground_truth_data,
        color_map,
        style_map,
        f"GT: {len(ground_truth_data)} | Pred: {len(pred_result_data)}",
    )
    draw_bboxes(ax, pred_result_data, color_map, style_map, title="")

    ax.set_title(f"GT vs Pred: {image_name}", fontsize=14, fontweight="bold")
    create_legend(fig, color_map, bbox_counts, used_geometry_types, style_map)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{os.path.splitext(image_name)[0]}_overlay.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return True


def load_inference_results(jsonl_file: str) -> List[Dict]:
    """
    Load inference results from JSONL file.

    Args:
        jsonl_file: Path to JSONL file containing inference results

    Returns:
        List of sample dictionaries
    """
    samples = []
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    sample = json.loads(line)
                    if isinstance(sample, dict):
                        samples.append(sample)
                    else:
                        logger.warning(
                            f"Line {line_num}: Expected dict, got {type(sample)}"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to load {jsonl_file}: {e}")
        return []

    logger.info(f"Successfully loaded {len(samples)} samples from {jsonl_file}")
    return samples


def main():
    # Load inference results
    print(f"Loading inference results from {INPUT_FILE}")
    samples = load_inference_results(INPUT_FILE)

    if not samples:
        print("❌ No samples found in input file")
        return

    print(f"✅ Loaded {len(samples)} samples")
    print(f"📁 Data root directory for images: {DATA_ROOT_DIR}")

    # Determine which samples to process
    if SAMPLE_INDICES and isinstance(SAMPLE_INDICES, list):
        indices = SAMPLE_INDICES
        samples_to_process = [samples[i] for i in indices if 0 <= i < len(samples)]
        print(f"📋 Processing specific samples: {indices}")
    elif MAX_SAMPLES:
        samples_to_process = samples[:MAX_SAMPLES]
        print(f"📋 Processing first {len(samples_to_process)} samples")
    else:
        samples_to_process = samples
        print(f"📋 Processing all {len(samples_to_process)} samples")

    # ------------------------------------------------------------------
    # Build a global color and style map to ensure consistency
    # ------------------------------------------------------------------
    global_labels = set()
    for sample in samples_to_process:
        gt_raw = sample.get("ground_truth", [])
        gt_iter = parse_bbox_data(gt_raw) if isinstance(gt_raw, str) else gt_raw
        for item in gt_iter:
            lbl = (
                item.get("desc")
                if isinstance(item.get("desc"), str)
                else str(item.get("label", "Unknown"))
            )
            global_labels.add(lbl)
        pred_raw = sample.get("prediction", [])
        pred_iter = parse_bbox_data(pred_raw) if isinstance(pred_raw, str) else pred_raw
        for item in pred_iter:
            lbl = (
                item.get("desc")
                if isinstance(item.get("desc"), str)
                else str(item.get("label", "Unknown"))
            )
            global_labels.add(lbl)

    color_map = generate_colors(list(global_labels))
    style_map = generate_label_styles(list(global_labels))

    # Process samples with progress bar
    success_count = 0
    with tqdm(
        total=len(samples_to_process), desc="Generating visualizations", unit="image"
    ) as pbar:
        for sample in samples_to_process:
            if visualize_sample(sample, OUTPUT_DIR, color_map, style_map, BASE_PATH):
                success_count += 1
            pbar.update(1)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input file (JSONL): {INPUT_FILE}")
    print(f"Data root directory: {DATA_ROOT_DIR}")
    print(f"Total samples in file: {len(samples)}")
    print(f"Samples processed: {len(samples_to_process)}")
    print(f"✅ Successful visualizations: {success_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎨 Geometry legend: {'Enabled' if SHOW_GEOMETRY_LEGEND else 'Disabled'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
