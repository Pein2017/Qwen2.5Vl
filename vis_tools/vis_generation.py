#!/usr/bin/env python3
"""
Visualization script for Qwen2.5-VL model inference results.
Creates side-by-side comparisons of ground truth vs predictions with different colors for different labels.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

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


def parse_bbox_data(bbox_str: str) -> List[Dict[str, Any]]:
    """
    Parse bbox data from JSON string format. If the JSON string is
    truncated (common when model output is cut off), attempt to recover
    by iteratively removing the last object until a valid JSON list is
    obtained.

    Args:
        bbox_str: JSON string or already-parsed Python list.

    Returns:
        List of dictionaries with keys like ``bbox_2d`` and ``label``.
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


def load_image_safe(image_path: str, base_path: str = ".") -> Tuple[np.ndarray, bool]:
    """
    Safely load image with fallback options.

    Args:
        image_path: Path to image file
        base_path: Base directory to prepend if image_path is relative

    Returns:
        Tuple of (image_array, success_flag)
    """
    # Try different path combinations
    path_candidates = [
        image_path,
        os.path.join(base_path, image_path),
        os.path.join(".", image_path),
        os.path.join("..", image_path),
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


def draw_bboxes(ax, bbox_data: List[Dict], color_map: Dict[str, str], title: str):
    """
    Draw bounding boxes on the given axis.

    Args:
        ax: Matplotlib axis
        bbox_data: List of bbox dictionaries
        color_map: Mapping from label to color
        title: Title for the subplot
    """
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Draw bounding boxes
    for item in bbox_data:
        bbox = item.get("bbox_2d", [])
        label = item.get("label", "Unknown")

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Get color for this label
        color = color_map.get(label, "#000000")  # Default to black if not found

        # Draw rectangle
        rect = Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add label text with background
        ax.text(
            x1,
            y1 - 5,
            label,
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )


def create_legend(
    fig, color_map: Dict[str, str], bbox_counts: Dict[str, Tuple[int, int]]
):
    """
    Create a legend showing label colors and counts.

    Args:
        fig: Matplotlib figure
        color_map: Mapping from label to color
        bbox_counts: Mapping from label to (ground_truth_count, prediction_count)
    """
    legend_elements = []
    for label in sorted(color_map.keys()):
        gt_count, pred_count = bbox_counts.get(label, (0, 0))
        legend_label = f"{label} (GT: {gt_count}, Pred: {pred_count})"
        legend_elements.append(
            patches.Patch(color=color_map[label], label=legend_label)
        )

    # Place legend outside the plot area
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=10,
        framealpha=0.9,
    )


def visualize_sample(
    sample_data: Dict,
    output_dir: str,
    color_map: Dict[str, str],
    base_path: str = ".",
) -> bool:
    """
    Visualize a single sample with ground truth and prediction side by side.

    Args:
        sample_data: Dictionary containing image, ground_truth, pred_result, etc.
        output_dir: Directory to save the visualization
        color_map: Pre-computed mapping from label to color (global consistency)
        base_path: Base directory for image paths

    Returns:
        True if successful, False otherwise
    """
    # Extract data
    image_path = sample_data.get("image", "")
    ground_truth_str = sample_data.get("ground_truth", "[]")
    pred_result_str = sample_data.get("pred_result", "[]")

    if not image_path:
        return False

    # Load image
    image_array, success = load_image_safe(image_path, base_path)
    if not success:
        return False

    # Parse bbox data
    ground_truth_data = parse_bbox_data(ground_truth_str)
    pred_result_data = parse_bbox_data(pred_result_str)

    if not ground_truth_data and not pred_result_data:
        return False

    # Count bboxes per label
    bbox_counts = defaultdict(lambda: [0, 0])
    for item in ground_truth_data:
        label = item.get("label", "Unknown")
        bbox_counts[label][0] += 1
    for item in pred_result_data:
        label = item.get("label", "Unknown")
        bbox_counts[label][1] += 1

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot ground truth
    ax1.imshow(image_array)
    draw_bboxes(
        ax1,
        ground_truth_data,
        color_map,
        f"Ground Truth ({len(ground_truth_data)} objects)",
    )

    # Plot predictions
    ax2.imshow(image_array)
    draw_bboxes(
        ax2,
        pred_result_data,
        color_map,
        f"Predictions ({len(pred_result_data)} objects)",
    )

    # Add overall title
    image_name = os.path.basename(image_path)
    fig.suptitle(
        f"Model Performance Comparison: {image_name}", fontsize=16, fontweight="bold"
    )

    # Create legend
    create_legend(fig, color_map, bbox_counts)

    # Adjust layout to accommodate legend
    plt.subplots_adjust(right=0.75)

    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{os.path.splitext(image_name)[0]}_comparison.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return True


def load_inference_results(json_file: str) -> List[Dict]:
    """
    Load inference results from JSON file.

    Args:
        json_file: Path to JSON file containing inference results

    Returns:
        List of sample dictionaries
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        else:
            logger.error(f"Expected list format in {json_file}, got {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Failed to load {json_file}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Qwen2.5-VL inference results"
    )
    parser.add_argument(
        "--input", required=True, help="Path to inference results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument("--base_path", default=".", help="Base path for image files")
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of samples to visualize"
    )
    parser.add_argument(
        "--sample_indices",
        help="Comma-separated list of specific sample indices to visualize",
    )

    args = parser.parse_args()

    # Load inference results
    print(f"Loading inference results from {args.input}")
    samples = load_inference_results(args.input)

    if not samples:
        print("❌ No samples found in input file")
        return

    print(f"✅ Loaded {len(samples)} samples")

    # Determine which samples to process
    if args.sample_indices:
        indices = [int(x.strip()) for x in args.sample_indices.split(",")]
        samples_to_process = [samples[i] for i in indices if 0 <= i < len(samples)]
        print(f"📋 Processing specific samples: {indices}")
    elif args.max_samples:
        samples_to_process = samples[: args.max_samples]
        print(f"📋 Processing first {len(samples_to_process)} samples")
    else:
        samples_to_process = samples
        print(f"📋 Processing all {len(samples_to_process)} samples")

    # ------------------------------------------------------------------
    # Build a global color map to ensure color consistency across samples
    # ------------------------------------------------------------------
    global_labels = set()
    for sample in samples_to_process:
        # Collect labels from ground truth and prediction boxes
        for item in parse_bbox_data(sample.get("ground_truth", "[]")):
            global_labels.add(item.get("label", "Unknown"))
        for item in parse_bbox_data(sample.get("pred_result", "[]")):
            global_labels.add(item.get("label", "Unknown"))

    color_map = generate_colors(list(global_labels))

    # Process samples with progress bar
    success_count = 0
    with tqdm(
        total=len(samples_to_process), desc="Generating visualizations", unit="image"
    ) as pbar:
        for sample in samples_to_process:
            if visualize_sample(sample, args.output_dir, color_map, args.base_path):
                success_count += 1
            pbar.update(1)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input file: {args.input}")
    print(f"Total samples in file: {len(samples)}")
    print(f"Samples processed: {len(samples_to_process)}")
    print(f"✅ Successful visualizations: {success_count}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
