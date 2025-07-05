#!/usr/bin/env python3
"""
Unified Data Processor

Consolidates all data processing functionality including sample processing,
teacher selection, train/val splitting, and output generation.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import DataConversionConfig
from data_splitter import DataSplitter
from image_processor import ImageProcessor
from teacher_selector import TeacherSelector
from utils.file_ops import FileOperations
from utils.transformations import FormatConverter, TokenMapper
from utils.validators import DataValidator, StructureValidator

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class SampleExtractor:
    """Extracts and processes individual samples from raw data."""
    
    def __init__(self, config: DataConversionConfig, token_mapper: Optional[TokenMapper] = None):
        """Initialize with configuration and optional token mapper."""
        self.config = config
        self.token_mapper = token_mapper
        # Load label hierarchy or use default
        if config.hierarchy_path and Path(config.hierarchy_path).exists():
            self.label_hierarchy = FileOperations.load_label_hierarchy(Path(config.hierarchy_path))
        else:
            # Default hierarchy for testing/basic usage
            self.label_hierarchy = {
                "螺丝连接点": ["BBU安装螺丝"],
                "连接状态": ["连接正确", "连接错误"],
                "设备": ["BBU", "天线", "电缆"],
                "标签贴纸": [],  # For token mapping compatibility
            }
        
        logger.info(f"SampleExtractor initialized with {len(self.label_hierarchy)} object types")
    
    def extract_content_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract and normalize content fields based on language."""
        if self.config.language == "chinese":
            return self._extract_chinese_fields(source_dict)
        else:
            return self._extract_english_fields(source_dict)
    
    def _extract_chinese_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract fields from Chinese contentZh format."""
        content_zh = source_dict.get("contentZh", {})
        if not content_zh:
            return {}
        
        # Extract label entries containing '标签' or '标签贴纸' (mapped version)
        label_values = []
        for key, value in content_zh.items():
            if "标签" in key:  # Matches both "标签" and "标签贴纸"
                if isinstance(value, list):
                    label_values.append(", ".join(map(str, value)))
                elif value:
                    label_values.append(str(value))
        
        if not label_values:
            return {}
        
        # Parse first label entry: "object_type/property/extra"
        label_string = label_values[0]
        parts = [p.strip() for p in label_string.split("/")]
        object_type = parts[0] if len(parts) >= 1 else ""
        property_value = parts[1] if len(parts) >= 2 else ""
        existing_extras = parts[2:] if len(parts) >= 3 else []
        
        # Collect additional extra_info from other contentZh entries
        additional_extras = []
        for key, value in content_zh.items():
            if "标签" not in key:
                if isinstance(value, list):
                    additional_extras.extend(str(item) for item in value if item)
                elif value:
                    additional_extras.append(str(value))
        
        extra_info = "/".join(existing_extras + additional_extras)
        
        return {
            "object_type": self._apply_token_mapping(object_type),
            "property": self._apply_token_mapping(property_value),
            "extra_info": self._apply_token_mapping(extra_info)
        }
    
    def _extract_english_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract fields from English format with fallback field names."""
        return {
            "object_type": self._apply_token_mapping(
                source_dict.get("object_type") or source_dict.get("label", "")
            ),
            "property": self._apply_token_mapping(
                source_dict.get("property") or source_dict.get("question", "")
            ),
            "extra_info": self._apply_token_mapping(
                source_dict.get("extra_info") or source_dict.get("question_ex", "")
            )
        }
    
    def _apply_token_mapping(self, token: str) -> str:
        """Apply token mapping if available."""
        if self.token_mapper and token:
            return self.token_mapper.map_token(token)
        return token
    
    def is_allowed_object(self, content_dict: Dict[str, str]) -> bool:
        """Check if object passes label hierarchy filtering."""
        obj_type = content_dict.get("object_type", "")
        prop = content_dict.get("property", "")
        
        # If no hierarchy is loaded, allow all objects
        if not self.label_hierarchy:
            return bool(obj_type)  # At least require an object type
        
        # Skip if object_type not in hierarchy
        if obj_type not in self.label_hierarchy:
            return False
        
        allowed_props = self.label_hierarchy.get(obj_type, [])
        
        # If no properties allowed, only accept empty property
        if not allowed_props:
            return prop == "" or prop is None
        
        # Check if property is directly allowed
        if prop in allowed_props:
            return True
        
        # Allow variant "obj_type/property" format stored in hierarchy
        combo = f"{obj_type}/{prop}" if prop else obj_type
        return combo in allowed_props
    
    def extract_objects_from_datalist(self, data_list: List[Dict]) -> List[Dict]:
        """Extract objects from dataList format."""
        objects = []
        
        for item in data_list:
            coords = item.get("coordinates", [])
            if len(coords) < 2:
                logger.warning(f"Invalid coordinates in dataList item: {coords}")
                continue
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            # Clean bbox coordinates for VLM training
            from utils.transformations import CoordinateTransformer
            bbox = CoordinateTransformer.clean_bbox_coordinates(bbox)
            
            properties = item.get("properties", {}) or {}
            content_dict = self.extract_content_fields(properties)
            
            if not content_dict or not self.is_allowed_object(content_dict):
                continue
            
            desc = FormatConverter.format_description(
                content_dict, self.config.response_types, self.config.language
            )
            if desc:
                objects.append({"bbox_2d": bbox, "desc": desc})
        
        return objects
    
    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """Extract objects from markResult features format."""
        objects = []
        
        for feature in features:
            geometry = feature.get("geometry", {})
            coords = geometry.get("coordinates", [])
            
            if not coords or not isinstance(coords, list) or len(coords) == 0:
                logger.warning(f"Invalid coordinates in markResult feature: {coords}")
                continue
            
            # Handle nested coordinate structures: [[x,y], ...] or [[[x,y], ...]]
            points = coords
            if (points and isinstance(points[0], list) and 
                points[0] and isinstance(points[0][0], list)):
                points = points[0]
            
            if not points or any(len(p) != 2 for p in points):
                logger.warning(f"Invalid points structure: {points}")
                continue
            
            # Extract bounding box from polygon points
            from utils.transformations import CoordinateTransformer
            bbox = CoordinateTransformer.extract_bbox_from_coordinates(points)
            # Already cleaned by extract_bbox_from_coordinates
            
            properties = feature.get("properties", {})
            content_dict = self.extract_content_fields(properties)
            
            if not content_dict or not self.is_allowed_object(content_dict):
                continue
            
            desc = FormatConverter.format_description(
                content_dict, self.config.response_types, self.config.language
            )
            if desc:
                objects.append({"bbox_2d": bbox, "desc": desc})
        
        return objects


class UnifiedProcessor:
    """Main orchestrator for the unified data processing pipeline."""
    
    def __init__(self, config: DataConversionConfig):
        """Initialize with configuration."""
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        
        # Initialize components
        self.token_mapper = self._init_token_mapper()
        self.sample_extractor = SampleExtractor(config, self.token_mapper)
        self.image_processor = ImageProcessor(config)
        self.teacher_selector = TeacherSelector(
            label_hierarchy=self.sample_extractor.label_hierarchy,
            max_teachers=config.max_teachers,
            seed=config.seed
        )
        self.data_splitter = DataSplitter(
            val_ratio=config.val_ratio,
            seed=config.seed
        )
        
        logger.info("UnifiedProcessor initialized successfully")
    
    def _init_token_mapper(self) -> Optional[TokenMapper]:
        """Initialize token mapper if token map file is provided."""
        if not self.config.token_map_path:
            if self.config.language == "english":
                raise ValueError("Token map required for English language mode")
            return None
        
        token_map_path = Path(self.config.token_map_path)
        if not token_map_path.exists():
            raise FileNotFoundError(f"Token map file not found: {token_map_path}")
        
        token_map = FileOperations.load_token_map(token_map_path)
        return TokenMapper(token_map)
    
    def process_single_sample(
        self, 
        json_path: Path
    ) -> Optional[Dict]:
        """Process a single JSON/image pair into a clean sample."""
        try:
            # Load JSON and find corresponding image
            json_data = FileOperations.load_json_data(json_path)
            image_path = FileOperations.find_image_file(json_path)
            
            # Get dimensions from JSON (these should be the processed dimensions)
            info = json_data["info"]
            json_width = info["width"]
            json_height = info["height"]
            
            # Get actual image dimensions
            actual_width, actual_height = FileOperations.get_image_dimensions(image_path)
            
            # Use the actual image dimensions for processing (they should match JSON after cleaning)
            if json_width != actual_width or json_height != actual_height:
                logger.warning(
                    f"Dimension mismatch for {image_path.name}: "
                    f"JSON says {json_width}x{json_height} but image is {actual_width}x{actual_height}. "
                    f"Using actual image dimensions."
                )
            
            # Use actual image dimensions for processing
            original_width, original_height = actual_width, actual_height
            
            # Extract objects from JSON data
            objects = []
            if "dataList" in json_data:
                objects = self.sample_extractor.extract_objects_from_datalist(json_data["dataList"])
            elif "markResult" in json_data and isinstance(json_data.get("markResult", {}).get("features"), list):
                objects = self.sample_extractor.extract_objects_from_markresult(
                    json_data["markResult"]["features"]
                )
            
            if not objects:
                logger.debug(f"No valid objects found in {json_path.name}")
                return None
            
            # Sort objects by position (top-left to bottom-right)
            objects.sort(key=lambda obj: (obj["bbox_2d"][1], obj["bbox_2d"][0]))
            
            # Process image (copy/resize)
            processed_image_path, final_width, final_height = self.image_processor.process_image(
                image_path, original_width, original_height, self.output_dir.parent
            )
            
            # Scale bounding boxes if image was resized
            if self.config.resize_enabled:
                self.image_processor.scale_object_coordinates(
                    objects, original_width, original_height, final_width, final_height
                )
            
            # Build relative image path for JSONL
            rel_image_path = FileOperations.calculate_relative_path(
                processed_image_path, self.output_dir.parent
            )
            
            return {
                "images": [rel_image_path],
                "objects": objects,
                "width": final_width,
                "height": final_height
            }
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            if self.config.fail_fast:
                raise
            return None
    
    def process_all_samples(self) -> List[Dict]:
        """Process all samples in the input directory."""
        logger.info("🚀 Starting sample processing")
        
        # Find all JSON files
        json_files = FileOperations.find_json_files(self.input_dir)
        logger.info(f"📁 Found {len(json_files)} JSON files")
        
        # Process all samples
        all_samples = []
        processed_count = 0
        skipped_count = 0
        
        for json_file in json_files:
            sample = self.process_single_sample(json_file)
            if sample:
                # Validate sample structure
                DataValidator.validate_sample_structure(sample)
                all_samples.append(sample)
                processed_count += 1
            else:
                skipped_count += 1
            
            if processed_count % 100 == 0 and processed_count > 0:
                logger.info(f"Processed {processed_count} samples...")
        
        logger.info(f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped")
        
        if not all_samples:
            raise ValueError("No valid samples were processed")
        
        return all_samples
    
    def split_into_sets(self, all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split samples into train/val/teacher sets."""
        logger.info("📊 Selecting teacher samples...")
        
        # Select teacher samples
        teacher_samples, teacher_indices = self.teacher_selector.select_teachers(all_samples)
        
        # Remove teacher samples from student pool
        teacher_image_paths = {sample["images"][0] for sample in teacher_samples}
        student_samples = [
            sample for sample in all_samples 
            if sample["images"][0] not in teacher_image_paths
        ]
        
        logger.info(f"📚 Teacher pool: {len(teacher_samples)} samples")
        logger.info(f"🎓 Student pool: {len(student_samples)} samples")
        
        # Split student samples into train/val
        logger.info("🔧 Splitting train/validation data...")
        train_samples, val_samples = self.data_splitter.split(student_samples)
        
        return train_samples, val_samples, teacher_samples
    
    def write_outputs(
        self, 
        train_samples: List[Dict], 
        val_samples: List[Dict], 
        teacher_samples: List[Dict]
    ) -> None:
        """Write all output files."""
        logger.info("💾 Writing output files...")
        
        # Write individual JSONL files
        FileOperations.write_jsonl(train_samples, self.output_dir / "train.jsonl")
        FileOperations.write_jsonl(val_samples, self.output_dir / "val.jsonl")
        FileOperations.write_jsonl(teacher_samples, self.output_dir / "teacher.jsonl")
        
        # Write combined file
        all_samples = teacher_samples + train_samples + val_samples
        FileOperations.write_jsonl(all_samples, self.output_dir / "all_samples.jsonl")
        
        # Extract and export unique labels
        self._export_label_vocabulary(all_samples)
        
        logger.info("📊 Output files written successfully")
    
    def _export_label_vocabulary(self, all_samples: List[Dict]) -> None:
        """Extract and export unique labels from all samples."""
        unique_labels = set()
        object_types = set()
        properties = set()
        full_descriptions = set()
        
        # Extract labels from all samples
        for sample in all_samples:
            for obj in sample.get("objects", []):
                desc = obj.get("desc", "")
                if desc:
                    full_descriptions.add(desc)
                    
                    # Parse description to extract components
                    from utils.transformations import FormatConverter
                    components = FormatConverter.parse_description_string(desc)
                    
                    obj_type = components.get("object_type", "").strip()
                    prop = components.get("property", "").strip()
                    extra = components.get("extra_info", "").strip()
                    
                    if obj_type:
                        object_types.add(obj_type)
                        unique_labels.add(obj_type)
                    
                    if prop:
                        properties.add(prop)
                        unique_labels.add(prop)
                    
                    if extra:
                        unique_labels.add(extra)
        
        # Create comprehensive label vocabulary
        label_vocabulary = {
            "metadata": {
                "total_samples": len(all_samples),
                "total_objects": sum(len(sample.get("objects", [])) for sample in all_samples),
                "language": self.config.language,
                "extraction_date": self._get_current_timestamp(),
                "description": "Complete vocabulary of labels extracted from the dataset for training prompt enhancement"
            },
            "statistics": {
                "unique_labels_count": len(unique_labels),
                "object_types_count": len(object_types),
                "properties_count": len(properties),
                "full_descriptions_count": len(full_descriptions)
            },
            "vocabulary": {
                "all_unique_labels": sorted(list(unique_labels)),
                "object_types": sorted(list(object_types)),
                "properties": sorted(list(properties)),
                "full_descriptions": sorted(list(full_descriptions))
            },
            "usage_notes": {
                "training_prompts": "Use 'all_unique_labels' for comprehensive label-aware training",
                "object_detection": "Use 'object_types' for class-specific detection tasks",
                "attribute_prediction": "Use 'properties' for attribute/property prediction",
                "full_context": "Use 'full_descriptions' for complete description generation"
            }
        }
        
        # Export to JSON file
        output_path = self.output_dir / "label_vocabulary.json"
        FileOperations.save_json_data(label_vocabulary, output_path, indent=2)
        
        logger.info(f"📋 Label vocabulary exported to {output_path}")
        logger.info(f"   📊 {len(unique_labels)} unique labels")
        logger.info(f"   🔖 {len(object_types)} object types")
        logger.info(f"   🏷️  {len(properties)} properties")
        logger.info(f"   📝 {len(full_descriptions)} complete descriptions")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def process(self) -> Dict[str, int]:
        """
        Execute the complete unified processing pipeline.
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info("🚀 Starting unified data processing pipeline")
        
        # Step 1: Process all samples
        all_samples = self.process_all_samples()
        
        # Step 2: Split into train/val/teacher sets
        train_samples, val_samples, teacher_samples = self.split_into_sets(all_samples)
        
        # Step 3: Validate output structure
        StructureValidator.validate_pipeline_output(
            train_samples, val_samples, teacher_samples
        )
        
        # Step 4: Write output files
        self.write_outputs(train_samples, val_samples, teacher_samples)
        
        # Step 5: Update output image coordinates if needed
        if self.config.resize_enabled:
            logger.info("🔧 Updating output coordinates...")
            self.image_processor.update_output_coordinates()
        
        # Report token mapping issues if applicable
        if self.token_mapper:
            missing_tokens = self.token_mapper.get_missing_tokens()
            if missing_tokens:
                logger.warning(f"Found {len(missing_tokens)} missing tokens: {sorted(missing_tokens)}")
            else:
                logger.info("✅ All tokens successfully mapped")
        
        # Final summary
        result = {
            "train": len(train_samples),
            "val": len(val_samples),
            "teacher": len(teacher_samples),
            "total_processed": len(all_samples)
        }
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {result['train']} samples → train.jsonl")
        logger.info(f"   Validation: {result['val']} samples → val.jsonl")
        logger.info(f"   Teacher: {result['teacher']} samples → teacher.jsonl")
        logger.info(f"   Combined: {result['total_processed']} samples → all_samples.jsonl")
        
        return result