#!/usr/bin/env python3
"""
Visualization tool for BBU detection dataset samples.
Reads JSONL files and visualizes images with bounding boxes and labels.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from PIL import Image, ImageDraw, ImageFont

# Configure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Configure Chinese font for matplotlib
try:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if os.path.exists(font_path):
        font_prop = FontProperties(fname=font_path)
        fontManager.addfont(font_path)
        rcParams["font.sans-serif"] = [font_prop.get_name()]
        rcParams["axes.unicode_minus"] = False
    else:
        print("Warning: Chinese font not found, falling back to default")
except Exception as e:
    print(f"Warning: Failed to configure Chinese font: {e}")

# Define paths
PROJECT_ROOT = Path("/data4/Qwen2.5-VL-main")
DEFAULT_JSONL_FILE = PROJECT_ROOT / "data" / "val.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "vis_tools" / "output"

# Color mapping - consistent with vis_scaling_comparison.py
FIXED_LABEL_COLORS = {
    "螺丝连接点": "#FF6B6B",  # Red
    "标签贴纸": "#4ECDC4",  # Teal
    "bbu基带处理单元": "#45B7D1",  # Blue
    "挡风板": "#96CEB4",  # Green
    "线缆": "#FFEAA7",  # Yellow
    "机柜空间": "#DDA0DD",  # Plum
}

# Additional color palette for other labels
COLOR_PALETTE = [
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
    "#F8C471",  # Orange
    "#82E0AA",  # Light Green
    "#F1948A",  # Pink
    "#85929E",  # Gray
    "#F4D03F",  # Bright Yellow
    "#FFA07A",  # Light salmon (fallback)
]


def get_color_for_desc(desc: str) -> str:
    """Get color based on object description."""
    # Check for main object types first
    for key, color in FIXED_LABEL_COLORS.items():
        if key in desc:
            return color

    # For other labels, use palette cycling
    desc_hash = hash(desc) % len(COLOR_PALETTE)
    return COLOR_PALETTE[desc_hash]


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def draw_bbox_with_label(
    draw: ImageDraw.Draw, bbox: List[int], label: str, color: str, font_size: int = 20
) -> None:
    """Draw bounding box with label on image."""
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Try to load Chinese font first, then fallback to default
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size
        )
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

    # Draw label background
    if font:
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
    else:
        text_width = len(label) * 8

    draw.rectangle([x1, y1 - 25, x1 + text_width + 4, y1], fill=color)

    # Draw label text
    draw.text((x1 + 2, y1 - 23), label, fill="white", font=font)


def visualize_sample(sample: Dict, output_dir: Path, show_plot: bool = False) -> None:
    """Visualize a single sample with bounding boxes."""
    image_path = PROJECT_ROOT / sample["images"][0]

    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return

    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Draw bounding boxes
    for obj in sample["objects"]:
        bbox = obj["bbox_2d"]
        desc = obj["desc"]
        color = get_color_for_desc(desc)

        # Truncate long descriptions
        label = desc if len(desc) <= 30 else desc[:27] + "..."
        draw_bbox_with_label(draw, bbox, label, color)

    # Save or show image
    if show_plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Sample: {Path(sample['images'][0]).name}")
        plt.tight_layout()
        plt.show()
    else:
        output_path = output_dir / f"vis_{Path(sample['images'][0]).stem}.png"
        img.save(output_path)
        print(f"Saved visualization: {output_path}")


def create_summary_plot(samples: List[Dict], output_dir: Path) -> None:
    """Create summary statistics plot."""
    # Count objects by type
    object_counts = {}
    for sample in samples:
        for obj in sample["objects"]:
            desc = obj["desc"]
            main_type = desc.split("/")[0] if "/" in desc else desc
            object_counts[main_type] = object_counts.get(main_type, 0) + 1

    # Create bar plot
    plt.figure(figsize=(12, 8))
    types = list(object_counts.keys())
    counts = list(object_counts.values())

    bars = plt.bar(types, counts, color=[get_color_for_desc(t) for t in types])
    plt.xlabel("Object Type")
    plt.ylabel("Count")
    plt.title("Object Type Distribution")
    plt.xticks(rotation=45, ha="right")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "object_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot: {output_dir / 'object_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BBU detection dataset samples"
    )
    parser.add_argument("--jsonl", type=str, help="Path to JSONL file")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to visualize",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plots instead of saving"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Create summary statistics plot"
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find JSONL file if not specified
    if not args.jsonl:
        args.jsonl = DEFAULT_JSONL_FILE

    # Load data
    samples = load_jsonl(args.jsonl)
    print(f"Loaded {len(samples)} samples from {args.jsonl}")

    # Create summary plot
    if args.summary:
        create_summary_plot(samples, output_dir)

    # Visualize samples
    max_samples = min(args.max_samples, len(samples))
    for i, sample in enumerate(samples[:max_samples]):
        print(f"Processing sample {i + 1}/{max_samples}")
        visualize_sample(sample, output_dir, args.show)

    print(f"\nVisualization complete! Check output directory: {output_dir}")


if __name__ == "__main__":
    main()
