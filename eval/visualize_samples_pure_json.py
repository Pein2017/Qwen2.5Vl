#!/usr/bin/env python3
"""
Visualize Qwen-VL pure JSONL responses by drawing bounding boxes and labels.
"""

import argparse
import json
import logging
import os
from itertools import cycle
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# Configure logging
typeLogger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def visualize_sample(
    sample: Dict[str, Any],
    image_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Draw bounding boxes and labels on the sample's image.

    Args:
        sample: A dictionary with keys 'images' and 'objects'.
        image_dir: Optional base directory for relative image paths.
        output_dir: Optional directory to save the visualization.

    Returns:
        Path to the saved visualization image, or None if displayed inline.
    """
    # Extract image path
    images = sample.get("images")
    if not isinstance(images, list) or len(images) == 0:
        logger.error("Missing or empty 'images' field in sample")
        return None
    img_item = images[0]
    if isinstance(img_item, dict):
        path = img_item.get("path")
    else:
        path = img_item  # assume string
    if not path:
        logger.error("Image path is empty")
        return None
    if image_dir and not os.path.isabs(path):
        path = os.path.join(image_dir, path)
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        return None

    # Load image
    image = cv2.imread(path)
    if image is None:
        logger.error(f"Failed to load image: {path}")
        return None
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    # Prepare a cycle of colors for distinct label_texts
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle_iter = cycle(color_list)
    label_color_map = {}
    ax.imshow(image_rgb)

    # Extract objects
    objects = sample.get("objects", {})
    ref_list = objects.get("ref", [])
    bbox_list = objects.get("bbox", [])
    if not (isinstance(ref_list, list) and isinstance(bbox_list, list)):
        logger.error("Invalid 'objects' field; expecting lists for 'ref' and 'bbox'")
        return None

    # Draw each bounding bbox_2d with label, using distinct colors per unique content
    for idx, (ref_str, bbox) in enumerate(zip(ref_list, bbox_list)):
        try:
            content = json.loads(ref_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse object ref JSON: {e}")
            continue
        # Build label text
        pure_label = content.get("label", "")

        # --- Start of new logic for label_text_for_display and category_for_color_and_legend ---
        parts: List[str] = []
        if pure_label:
            parts.append(pure_label)

        raw_q = content.get("question")
        question_str = ""
        if raw_q:
            if isinstance(raw_q, list):
                question_str = " ".join(map(str, raw_q))
            else:
                question_str = str(raw_q)
            if question_str:  # Add non-empty question string to parts
                parts.append(question_str)

        question_ex = content.get("question_ex")
        if question_ex:  # Add non-empty question_ex to parts
            parts.append(str(question_ex))

        # label_text_for_display will be what's shown on the image
        if parts:
            label_text_for_display = " | ".join(parts)
        else:
            label_text_for_display = str(
                idx
            )  # Fallback to index if no label, q, or q_ex

        # category_for_color_and_legend will determine the color and legend entry
        # If there are question details, the category is the full string including them.
        # Otherwise, it's just the pure_label (or index if pure_label is also empty).
        if question_str or question_ex:  # If any question details exist
            category_for_color_and_legend = label_text_for_display
        else:  # Only pure_label (or index)
            category_for_color_and_legend = pure_label if pure_label else str(idx)
        # --- End of new logic ---

        # Choose color for this category_for_color_and_legend
        if category_for_color_and_legend not in label_color_map:
            label_color_map[category_for_color_and_legend] = next(color_cycle_iter)
        color = label_color_map[category_for_color_and_legend]

        # Validate and unpack bbox coordinates
        if not (isinstance(bbox, list) and len(bbox) == 4):
            logger.error(f"Invalid bbox at index {idx}: {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        rect = Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

    # Finalize
    ax.axis("off")
    ax.set_title(os.path.basename(path))
    # Add legend for distinct labels, positioned outside the image
    if label_color_map:
        handles = [
            Patch(color=color, label=label) for label, color in label_color_map.items()
        ]
        labels = list(label_color_map.keys())
        # Shrink image axes to make room for legend on the right
        bbox_2d = ax.get_position()
        ax.set_position([bbox_2d.x0, bbox_2d.y0, bbox_2d.width * 0.75, bbox_2d.height])
        # Place legend outside the image area on the figure
        fig.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize="small",
            framealpha=0.7,
        )
        # Adjust layout to accommodate legend
        plt.tight_layout()

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{base}_vis.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
        return None


def load_samples(
    jsonl_path: str,
    count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load up to 'count' samples from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL input file.
        count: Optional max number of samples to load.

    Returns:
        List of sample dictionaries.
    """
    samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if count is not None and i >= count:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON line at {i}: {e}")
                continue
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pure JSONL Qwen-VL responses"
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--image_dir", help="Base directory for image files if paths are relative"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save visualizations; if omitted, display inline",
    )
    parser.add_argument(
        "--count", type=int, help="Number of samples to process (default: all)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    samples = load_samples(args.input, args.count)
    logger.info(f"Loaded {len(samples)} samples from {args.input}")

    visualization_count = 0
    for idx, sample in enumerate(samples):
        out_path = visualize_sample(sample, args.image_dir, args.output_dir)
        if out_path:
            visualization_count += 1
            logger.info(f"Visualization saved to: {out_path}")

    logger.info(f"Total visualizations: {visualization_count}")


if __name__ == "__main__":
    main()
