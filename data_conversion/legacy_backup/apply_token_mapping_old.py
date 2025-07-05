#!/usr/bin/env python3
"""
Apply token mapping to cleaned JSON files.

This script transforms the raw Chinese annotation keys in cleaned JSON files
to standardized terms using the token mapping configuration.

Usage:
    python apply_token_mapping.py <cleaned_json_dir> <token_map_path>
"""

import json
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_token_mapping(token_map_path: str) -> dict:
    """Load token mapping from JSON file."""
    try:
        with open(token_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load token mapping from {token_map_path}: {e}")
        return {}

def apply_token_mapping_to_feature(feature: dict, token_map: dict) -> dict:
    """Apply token mapping to a single feature's contentZh."""
    if 'properties' not in feature or 'contentZh' not in feature['properties']:
        return feature
    
    content_zh = feature['properties']['contentZh']
    updated_content = {}
    
    # Apply token mapping to each key in contentZh
    for key, value in content_zh.items():
        # Map the key if it exists in token_map
        mapped_key = token_map.get(key, key)
        
        # For nested values like "标签/匹配" -> "标签贴纸/匹配"
        if isinstance(value, str) and '/' in value:
            parts = value.split('/')
            if parts[0] in token_map:
                parts[0] = token_map[parts[0]]
                value = '/'.join(parts)
        
        updated_content[mapped_key] = value
    
    # Update the feature
    feature['properties']['contentZh'] = updated_content
    return feature

def apply_token_mapping_to_json(json_path: str, token_map: dict) -> bool:
    """Apply token mapping to a single JSON file."""
    try:
        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Apply token mapping to all features
        if 'markResult' in data and 'features' in data['markResult']:
            for feature in data['markResult']['features']:
                apply_token_mapping_to_feature(feature, token_map)
        
        # Save the updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
        
        return True
    except Exception as e:
        logger.error(f"Failed to process {json_path}: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python apply_token_mapping.py <cleaned_json_dir> <token_map_path>")
        sys.exit(1)
    
    cleaned_json_dir = sys.argv[1]
    token_map_path = sys.argv[2]
    
    # Load token mapping
    token_map = load_token_mapping(token_map_path)
    if not token_map:
        logger.error("No token mapping loaded, exiting")
        sys.exit(1)
    
    logger.info(f"Loaded token mapping: {token_map}")
    
    # Process all JSON files in the directory
    json_files = list(Path(cleaned_json_dir).glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    success_count = 0
    for json_file in json_files:
        if apply_token_mapping_to_json(str(json_file), token_map):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(json_files)} files")
    
    if success_count == len(json_files):
        logger.info("✅ Token mapping applied successfully to all files")
    else:
        logger.error(f"❌ {len(json_files) - success_count} files failed processing")
        sys.exit(1)

if __name__ == "__main__":
    main()