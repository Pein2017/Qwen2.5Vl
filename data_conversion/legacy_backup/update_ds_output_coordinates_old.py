#!/usr/bin/env python3
"""
Update ds_output Coordinates to Match Resized Images

This script fixes the coordinate scaling issue in ds_output by updating
JSON coordinates to match their corresponding resized images.

The issue: 
- ds_output images are resized during processing
- ds_output JSON files still contain original coordinates
- This creates coordinate-image mismatches

The fix:
- Calculate scale factors from original to resized images
- Update all coordinates in ds_output JSON files accordingly
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def update_ds_output_coordinates(
    original_dir: str = "ds",
    ds_output_dir: str = "ds_output"
) -> bool:
    """
    Update coordinates in ds_output JSON files to match resized images.
    
    Args:
        original_dir: Directory with original images (for scale calculation)
        ds_output_dir: Directory with resized images and JSON files to update
    
    Returns:
        True if successful, False otherwise
    """
    
    original_path = Path(original_dir)
    ds_output_path = Path(ds_output_dir)
    
    logger.info(f"Updating ds_output coordinates:")
    logger.info(f"  Original images: {original_path}")
    logger.info(f"  ds_output: {ds_output_path}")
    
    # Find all JSON files in ds_output
    json_files = list(ds_output_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    updated_count = 0
    errors = []
    
    for json_file in json_files:
        try:
            sample_id = json_file.stem
            
            # Find corresponding images
            original_image = None
            ds_output_image = None
            
            # Look for original image
            for ext in ['.jpeg', '.jpg']:
                orig_path = original_path / f"{sample_id}{ext}"
                if orig_path.exists():
                    original_image = orig_path
                    break
            
            # Look for ds_output image
            for ext in ['.jpeg', '.jpg']:
                ds_out_img_path = ds_output_path / f"{sample_id}{ext}"
                if ds_out_img_path.exists():
                    ds_output_image = ds_out_img_path
                    break
            
            if not original_image:
                logger.debug(f"Original image not found for {sample_id}, skipping")
                continue
                
            if not ds_output_image:
                logger.debug(f"ds_output image not found for {sample_id}, skipping")
                continue
            
            # Get image sizes
            with Image.open(original_image) as img:
                original_size = img.size  # (width, height)
            
            with Image.open(ds_output_image) as img:
                ds_output_size = img.size  # (width, height)
            
            # Check if resize occurred
            if original_size == ds_output_size:
                logger.debug(f"No resize detected for {sample_id}")
                continue
            
            # Calculate scale factors
            scale_x = ds_output_size[0] / original_size[0]
            scale_y = ds_output_size[1] / original_size[1]
            
            logger.info(f"Sample {sample_id}: {original_size} → {ds_output_size} (scale: {scale_x:.4f}, {scale_y:.4f})")
            
            # Load and fix JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update image dimensions in info section if present
            if "info" in data:
                if "width" in data["info"]:
                    data["info"]["width"] = ds_output_size[0]
                if "height" in data["info"]:
                    data["info"]["height"] = ds_output_size[1]
            
            # Check if coordinates need scaling
            needs_scaling = False
            if "markResult" in data and "features" in data["markResult"]:
                for feature in data["markResult"]["features"]:
                    geometry = feature.get("geometry", {})
                    coordinates = geometry.get("coordinates", [])
                    
                    if coordinates:
                        # Check if any coordinate exceeds ds_output image bounds
                        x_coords = [pt[0] for pt in coordinates]
                        y_coords = [pt[1] for pt in coordinates]
                        
                        max_x, max_y = max(x_coords), max(y_coords)
                        
                        if max_x > ds_output_size[0] or max_y > ds_output_size[1]:
                            needs_scaling = True
                            break
            
            if not needs_scaling:
                logger.debug(f"Coordinates already correctly scaled for {sample_id}")
                continue
            
            # Backup original file
            backup_path = json_file.with_suffix('.json.backup')
            if not backup_path.exists():  # Only backup if not already backed up
                shutil.copy2(json_file, backup_path)
            
            # Fix coordinates
            features_updated = 0
            
            if "markResult" in data and "features" in data["markResult"]:
                for feature in data["markResult"]["features"]:
                    geometry = feature.get("geometry", {})
                    if geometry.get("type") != "ExtentPolygon":
                        continue
                        
                    coordinates = geometry.get("coordinates", [])
                    if not coordinates:
                        continue
                    
                    # Scale coordinates
                    scaled_coordinates = []
                    for point in coordinates:
                        scaled_x = point[0] * scale_x
                        scaled_y = point[1] * scale_y
                        scaled_coordinates.append([scaled_x, scaled_y])
                    
                    # Update coordinates
                    geometry["coordinates"] = scaled_coordinates
                    features_updated += 1
            
            # Save updated data
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Updated {features_updated} features in {sample_id}")
            updated_count += 1
            
        except Exception as e:
            error_msg = f"Failed to process {json_file}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Summary
    logger.info(f"✅ Successfully updated coordinates for {updated_count} samples")
    if errors:
        logger.warning(f"⚠️ {len(errors)} errors occurred:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return len(errors) == 0

def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update ds_output coordinates to match resized images")
    parser.add_argument("--original_dir", default="ds", help="Directory with original images")
    parser.add_argument("--ds_output_dir", default="ds_output", help="Directory with resized images and JSON files")
    
    args = parser.parse_args()
    
    success = update_ds_output_coordinates(args.original_dir, args.ds_output_dir)
    
    if success:
        logger.info("✅ ds_output coordinate update completed successfully!")
        return 0
    else:
        logger.error("❌ ds_output coordinate update failed - check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())