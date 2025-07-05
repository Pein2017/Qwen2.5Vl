#!/usr/bin/env python3
"""
Enhanced Visualization Tool for Training vs Raw Cleaned Image Comparison

Compares raw cleaned images (ds_output/) with training rescaled images by:
1. Taking a sample ID as input
2. Loading training data from data/all_samples.jsonl
3. Loading raw cleaned annotations from ds_output/*.json
4. Visualizing both versions side-by-side with consistent annotations
"""

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

# Sample ID to visualize (modify this to change which sample to visualize)
SAMPLE_ID = "QC-20230225-0000414_19823"

# Input paths
TRAINING_DATA_PATH = "data/all_samples.jsonl"  # Training data with rescaled annotations
RAW_CLEANED_DIR = "ds_output"                  # Directory with raw cleaned images and JSON
BASE_DIR = "."                                 # Base directory

# Output settings
OUTPUT_DIR = "scaling_comparisons"             # Where to save visualizations
OUTPUT_FILENAME = f"{SAMPLE_ID}_training_vs_raw.jpeg"  # Output file name

# Visualization settings
FIGURE_SIZE = (20, 10)                        # Figure size (width, height)
DPI = 300                                     # Output resolution
FONT_SIZE_TITLE = 14                          # Title font size
FONT_SIZE_LABEL = 8                           # Label font size

# ============================================================================
# IMPLEMENTATION - No need to modify below this line
# ============================================================================

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


class TrainingVsRawVisualizer:
    """Visualizes training rescaled images vs raw cleaned images comparison."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        # Fixed color palette for consistent visualization
        self.color_palette = [
            "#FF6B6B",  # Red - for 螺丝连接点
            "#4ECDC4",  # Teal - for 标签贴纸  
            "#45B7D1",  # Blue - for bbu基带处理单元
            "#96CEB4",  # Green - for 挡风板
            "#FFEAA7",  # Yellow - for 线缆
            "#DDA0DD",  # Plum - for 机柜空间
            "#98D8C8",  # Mint
            "#F7DC6F",  # Gold
            "#BB8FCE",  # Purple
            "#85C1E9",  # Light Blue
            "#F8C471",  # Orange
            "#82E0AA",  # Light Green
            "#F1948A",  # Pink
            "#85929E",  # Gray
            "#F4D03F",  # Bright Yellow
        ]
        # Pre-assign colors to main object types for consistency
        self.fixed_label_colors = {
            "螺丝连接点": "#FF6B6B",
            "标签贴纸": "#4ECDC4", 
            "bbu基带处理单元": "#45B7D1",
            "挡风板": "#96CEB4",
            "线缆": "#FFEAA7",
            "机柜空间": "#DDA0DD"
        }
        self.label_to_color = {}

    def get_label_color(self, label: str) -> str:
        """Get consistent color for a label with pre-assigned main types."""
        # Check for main object types first
        for main_type, color in self.fixed_label_colors.items():
            if main_type in label:
                return color
        
        # For other labels, assign consistently
        if label not in self.label_to_color:
            used_colors = set(self.fixed_label_colors.values()) | set(self.label_to_color.values())
            available_colors = [c for c in self.color_palette if c not in used_colors]
            if available_colors:
                self.label_to_color[label] = available_colors[0]
            else:
                # Fallback to palette index
                color_idx = len(self.label_to_color) % len(self.color_palette)
                self.label_to_color[label] = self.color_palette[color_idx]
        return self.label_to_color[label]

    def load_training_sample(self, sample_id: str, jsonl_path: str) -> Optional[Dict]:
        """Load training sample data from JSONL file."""
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        # Check if this sample contains our target image
                        images = sample.get("images", [])
                        for image_path in images:
                            if sample_id in image_path:
                                logger.info(f"Found training sample for {sample_id}")
                                return sample
                    except json.JSONDecodeError:
                        continue
            
            logger.error(f"Training sample not found for {sample_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load training data from {jsonl_path}: {e}")
            return None

    def load_raw_cleaned_data(self, sample_id: str, raw_dir: str) -> Optional[Dict]:
        """Load raw cleaned data from ds_output JSON file."""
        try:
            json_path = Path(raw_dir) / f"{sample_id}.json"
            
            if not json_path.exists():
                logger.error(f"Raw cleaned JSON not found: {json_path}")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded raw cleaned data for {sample_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load raw cleaned data: {e}")
            return None

    def extract_objects_from_raw_data(self, raw_data: Dict) -> List[Dict]:
        """Extract objects from raw cleaned data format."""
        objects = []
        
        if "markResult" in raw_data and "features" in raw_data["markResult"]:
            features = raw_data["markResult"]["features"]
            logger.info(f"Processing {len(features)} raw features")
            
            for i, feature in enumerate(features):
                try:
                    # Extract geometry coordinates
                    geometry = feature.get("geometry", {})
                    if geometry.get("type") != "ExtentPolygon":
                        continue
                    
                    coordinates = geometry.get("coordinates", [])
                    if not coordinates or len(coordinates) < 4:
                        continue
                    
                    # Convert polygon to bbox
                    x_coords = [pt[0] for pt in coordinates]
                    y_coords = [pt[1] for pt in coordinates]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    # Extract Chinese label
                    properties = feature.get("properties", {})
                    content_zh = properties.get("contentZh", {})
                    
                    label = "Unknown"
                    # Try different label keys
                    for key in ["标签贴纸", "标签"]:
                        if key in content_zh and content_zh[key]:
                            label = content_zh[key]
                            break
                    
                    objects.append({"bbox_2d": bbox, "desc": label})
                    logger.debug(f"Raw feature {i+1}: {label}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process raw feature {i}: {e}")
                    continue
        
        logger.info(f"Extracted {len(objects)} objects from raw data")
        return objects

    def load_image_safe(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Safely load image and return array and dimensions."""
        try:
            abs_path = (
                self.base_dir / image_path
                if not Path(image_path).is_absolute()
                else Path(image_path)
            )

            if not abs_path.exists():
                logger.error(f"Image file not found: {abs_path}")
                return None, None

            with Image.open(abs_path) as img:
                img_rgb = img.convert("RGB")
                img_array = np.array(img_rgb)
                return img_array, img_rgb.size  # (width, height)

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, None

    def draw_bboxes_on_axis(self, ax, image_array: np.ndarray, objects: List[Dict], title: str):
        """Draw bounding boxes on a matplotlib axis."""
        ax.imshow(image_array)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=10)
        ax.axis("off")

        for i, obj in enumerate(objects):
            bbox = obj.get("bbox_2d", [])
            label = obj.get("desc", "Unknown")

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Get consistent color for this label
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

            # Add numbered label
            label_text = f"{i+1}. {label[:25]}..." if len(label) > 25 else f"{i+1}. {label}"
            ax.text(
                x1,
                y1 - 8,
                label_text,
                fontsize=FONT_SIZE_LABEL,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                verticalalignment="top",
                fontweight="bold"
            )

    def create_comparison_visualization(self, sample_id: str, training_jsonl: str, raw_dir: str, output_dir: str) -> bool:
        """Create training vs raw cleaned comparison visualization."""
        logger.info(f"Creating comparison visualization for {sample_id}")
        
        # Load training data
        training_sample = self.load_training_sample(sample_id, training_jsonl)
        if not training_sample:
            return False
        
        # Load raw cleaned data
        raw_data = self.load_raw_cleaned_data(sample_id, raw_dir)
        if not raw_data:
            return False
        
        # Extract objects
        training_objects = training_sample.get("objects", [])
        raw_objects = self.extract_objects_from_raw_data(raw_data)
        
        if not training_objects and not raw_objects:
            logger.warning("No objects found in either training or raw data")
            return False
        
        # Get image paths and dimensions
        training_images = training_sample.get("images", [])
        if not training_images:
            logger.error("No training images found")
            return False
        
        training_image_path = training_images[0]
        training_width = training_sample.get("width", 0)
        training_height = training_sample.get("height", 0)
        
        # Raw cleaned image path
        raw_image_path = f"{raw_dir}/{sample_id}.jpeg"
        
        # Load images
        raw_image, raw_size = self.load_image_safe(raw_image_path)
        training_image, _ = self.load_image_safe(training_image_path)
        
        if raw_image is None or training_image is None:
            logger.error("Failed to load images")
            return False
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        
        # Draw raw cleaned image with raw annotations
        self.draw_bboxes_on_axis(
            ax1,
            raw_image,
            raw_objects,
            f'Raw Cleaned ({raw_size[0]}×{raw_size[1]})\n{len(raw_objects)} objects with polished annotations'
        )
        
        # Draw training rescaled image with training annotations
        self.draw_bboxes_on_axis(
            ax2,
            training_image,
            training_objects,
            f'Training Rescaled ({training_width}×{training_height})\n{len(training_objects)} objects ready for training'
        )
        
        # Set overall title
        fig.suptitle(f'Training vs Raw Cleaned Comparison: {sample_id}', fontsize=16, fontweight='bold')
        
        # Create legend with all unique labels
        all_labels = set()
        for obj in raw_objects + training_objects:
            all_labels.add(obj.get("desc", "Unknown"))
        
        if all_labels:
            legend_elements = []
            for label in sorted(all_labels):
                color = self.get_label_color(label)
                display_label = label[:30] + "..." if len(label) > 30 else label
                legend_elements.append(
                    patches.Patch(color=color, label=display_label)
                )
            
            # Place legend on the right side
            fig.legend(
                handles=legend_elements,
                loc="center right",
                bbox_to_anchor=(0.98, 0.5),
                fontsize=8,
                framealpha=0.9,
                title="Object Types",
                title_fontsize=10
            )
        
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, hspace=0.3, wspace=0.1)
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, OUTPUT_FILENAME)
        
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', format='jpeg', facecolor='white')
        plt.close()
        
        logger.info(f"✅ Saved comparison visualization: {output_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"TRAINING VS RAW CLEANED COMPARISON")
        print(f"{'='*80}")
        print(f"Sample ID: {sample_id}")
        print(f"")
        print(f"Raw Cleaned:")
        print(f"  Image: {raw_image_path} ({raw_size[0]}×{raw_size[1]})")
        print(f"  Objects: {len(raw_objects)}")
        print(f"  Format: Polished annotations from ds_output")
        print(f"")
        print(f"Training Rescaled:")
        print(f"  Image: {training_image_path} ({training_width}×{training_height})")
        print(f"  Objects: {len(training_objects)}")
        print(f"  Format: Final training data")
        print(f"")
        print(f"Scale factors: x={training_width/raw_size[0]:.4f}, y={training_height/raw_size[1]:.4f}")
        print(f"Output: {output_path}")
        print(f"{'='*80}")
        
        return True


def main():
    """Main execution function."""
    print(f"🎯 Starting visualization for sample: {SAMPLE_ID}")
    print(f"📂 Training data: {TRAINING_DATA_PATH}")
    print(f"📂 Raw cleaned data: {RAW_CLEANED_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("")
    
    # Validate input paths
    if not os.path.exists(TRAINING_DATA_PATH):
        logger.error(f"Training data file not found: {TRAINING_DATA_PATH}")
        return 1
    
    if not os.path.exists(RAW_CLEANED_DIR):
        logger.error(f"Raw cleaned directory not found: {RAW_CLEANED_DIR}")
        return 1
    
    # Initialize visualizer
    visualizer = TrainingVsRawVisualizer(BASE_DIR)
    
    # Create comparison visualization
    success = visualizer.create_comparison_visualization(
        SAMPLE_ID, 
        TRAINING_DATA_PATH, 
        RAW_CLEANED_DIR, 
        OUTPUT_DIR
    )
    
    if success:
        print(f"🎉 Visualization completed successfully!")
        print(f"📁 Check output: {OUTPUT_DIR}/{OUTPUT_FILENAME}")
        return 0
    else:
        print("❌ Visualization failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())