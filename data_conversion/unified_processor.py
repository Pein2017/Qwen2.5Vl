#!/usr/bin/env python3
"""
Streamlined Unified Data Processor

Consolidates all data processing functionality with simplified architecture.
Merged SampleExtractor directly into UnifiedProcessor to eliminate redundancy.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from coordinate_manager import CoordinateManager
from data_splitter import DataSplitter  
from hierarchical_processor_compat import HierarchicalProcessor
from image_processor import ImageProcessor
from teacher_selector import TeacherSelector

from config import DataConversionConfig
from utils.file_ops import FileOperations
from utils.transformations import FormatConverter
from utils.validators import DataValidator, StructureValidator


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class UnifiedProcessor:
    """Streamlined orchestrator for the unified data processing pipeline."""

    def __init__(self, config: DataConversionConfig):
        """Initialize with configuration."""
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = config.get_dataset_output_dir()

        # Initialize components - no token mapping needed for Chinese-only

        # Load label hierarchy or use default
        if config.hierarchy_path and Path(config.hierarchy_path).exists():
            self.label_hierarchy = FileOperations.load_label_hierarchy(
                Path(config.hierarchy_path)
            )
        else:
            # Default hierarchy matching actual v2 data structure
            self.label_hierarchy = {
                "螺丝、光纤插头": ["BBU安装螺丝", "BBU端光纤插头"],
                "标签": [],  # Labels can have any content
                "BBU设备": ["华为"],  # BBU equipment brand
                "光纤": [],  # Fiber optics
                "电线": [],  # Electrical wires
                "挡风板": ["华为"],  # BBU shields
            }

        # Initialize hierarchical processor for v2 data support (Chinese only)
        self.hierarchical_processor = HierarchicalProcessor(
            response_types=set(config.response_types),
            label_hierarchy=self.label_hierarchy,
        )

        self.image_processor = ImageProcessor(config)
        self.teacher_selector = TeacherSelector(
            label_hierarchy=self.label_hierarchy,
            max_teachers=config.max_teachers,
            seed=config.seed,
        )
        self.data_splitter = DataSplitter(val_ratio=config.val_ratio, seed=config.seed)

        logger.info("UnifiedProcessor initialized successfully (Chinese-only mode)")

    def extract_content_fields(self, source_dict: Dict) -> Dict[str, str]:
        """Extract and normalize content fields from Chinese contentZh format."""
        return self._extract_chinese_fields(source_dict)

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
            "object_type": object_type,
            "property": property_value,
            "extra_info": extra_info,
        }


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
                content_dict, self.config.response_types, "chinese"
            )
            if desc:
                objects.append({"bbox_2d": bbox, "desc": desc})

        return objects

    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """Extract objects from markResult features with native geometry types."""
        # Use hierarchical processor for V2 data support
        objects = self.hierarchical_processor.extract_objects_from_markresult(features)
        return objects

    def process_single_sample(self, json_path: Path) -> Optional[Dict]:
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
            actual_width, actual_height = FileOperations.get_image_dimensions(
                image_path
            )

            # Detect dimension mismatch (likely due to EXIF orientation)
            if json_width != actual_width or json_height != actual_height:
                logger.info(
                    f"Dimension mismatch for {image_path.name}: "
                    f"JSON says {json_width}x{json_height} but image is {actual_width}x{actual_height}. "
                    f"Will apply coordinate rescaling."
                )

            # Keep JSON dimensions for coordinate transformation pipeline
            original_width, original_height = json_width, json_height

            # Extract objects from JSON data
            objects = []
            if "dataList" in json_data:
                objects = self.extract_objects_from_datalist(json_data["dataList"])
            elif "markResult" in json_data and isinstance(
                json_data.get("markResult", {}).get("features"), list
            ):
                objects = self.extract_objects_from_markresult(
                    json_data["markResult"]["features"]
                )

            if not objects:
                logger.debug(f"No valid objects found in {json_path.name}")
                return None

            # Apply unified coordinate transformation pipeline
            sample_data = {"objects": objects}
            processed_sample, final_width, final_height = (
                self._process_sample_coordinates_unified(
                    sample_data, image_path, json_width, json_height, self.config.resize
                )
            )
            objects = processed_sample["objects"]

            # Sort objects by position using first coordinate pair
            def get_sort_key(obj):
                if "bbox_2d" in obj:
                    return (obj["bbox_2d"][1], obj["bbox_2d"][0])  # y, x
                elif "square" in obj:
                    return (obj["square"][1], obj["square"][0])  # y, x of first point
                elif "line" in obj:
                    return (obj["line"][1], obj["line"][0])  # y, x of first point
                return (0, 0)  # fallback
            
            objects.sort(key=get_sort_key)

            # Process image (copy/resize) to match coordinate transformations
            processed_image_path, _, _ = self.image_processor.process_image(
                image_path, json_width, json_height
            )

            # Build relative image path for JSONL (relative to dataset directory)
            rel_image_path = self.image_processor.get_relative_image_path(
                processed_image_path
            )

            return {
                "images": [rel_image_path],
                "objects": objects,
                "width": final_width,
                "height": final_height,
            }

        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            if self.config.fail_fast:
                raise
            return None

    def _process_sample_coordinates_unified(
        self,
        sample_data: Dict,
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
    ) -> Tuple[Dict, int, int]:
        """
        Process sample coordinates using unified geometry transformation.

        This replaces the old bbox-only processing with geometry-aware processing
        that works with both simple bbox and complex geometries.
        """
        if "objects" not in sample_data or not sample_data["objects"]:
            # No objects to process, just get final dimensions
            if enable_smart_resize:
                from data_conversion.vision_process import (
                    MAX_PIXELS,
                    MIN_PIXELS,
                    smart_resize,
                )

                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                resize_h, resize_w = smart_resize(
                    height=final_h,
                    width=final_w,
                    factor=28,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                return sample_data, final_w, final_h

        # Process first object to get final dimensions
        first_obj = sample_data["objects"][0]

        # Get any coordinate for dimension calculation
        if "bbox_2d" in first_obj:
            geometry_input = first_obj["bbox_2d"]
        elif "square" in first_obj:
            # Create bbox from square for dimension calculation
            square = first_obj["square"]
            x_coords = [square[i] for i in range(0, len(square), 2)]
            y_coords = [square[i] for i in range(1, len(square), 2)]
            geometry_input = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        elif "line" in first_obj:
            # Create bbox from line for dimension calculation
            line = first_obj["line"]
            x_coords = [line[i] for i in range(0, len(line), 2)]
            y_coords = [line[i] for i in range(1, len(line), 2)]
            geometry_input = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        else:
            raise ValueError(f"No supported geometry type in first object: {first_obj}")

        # Use unified geometry transformation for dimension calculation
        final_bbox, final_geometry, final_width, final_height = (
            CoordinateManager.transform_geometry_complete(
                geometry_input, image_path, json_width, json_height, enable_smart_resize
            )
        )

        # Process all objects with their native geometry
        updated_objects = []
        for obj in sample_data["objects"]:
            # Transform coordinates based on geometry type
            updated_obj = obj.copy()
            
            if "bbox_2d" in obj:
                # Transform bbox coordinates
                transformed_coords = self._transform_coordinates(
                    obj["bbox_2d"], image_path, json_width, json_height, enable_smart_resize
                )
                updated_obj["bbox_2d"] = [int(round(c)) for c in transformed_coords]
            
            elif "square" in obj:
                # Transform square coordinates (8 coordinates: x1,y1,x2,y2,x3,y3,x4,y4)
                coords = obj["square"]
                transformed_coords = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = [coords[i], coords[i+1]]
                        transformed_point = self._transform_coordinates(
                            point, image_path, json_width, json_height, enable_smart_resize, is_point=True
                        )
                        transformed_coords.extend([int(round(transformed_point[0])), int(round(transformed_point[1]))])
                updated_obj["square"] = transformed_coords
            
            elif "line" in obj:
                # Transform line coordinates (sequence of x,y pairs)
                coords = obj["line"]
                transformed_coords = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = [coords[i], coords[i+1]]
                        transformed_point = self._transform_coordinates(
                            point, image_path, json_width, json_height, enable_smart_resize, is_point=True
                        )
                        transformed_coords.extend([int(round(transformed_point[0])), int(round(transformed_point[1]))])
                updated_obj["line"] = transformed_coords
            
            updated_objects.append(updated_obj)

        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects

        return updated_sample, final_width, final_height

    def _transform_coordinates(self, coords, image_path, json_width, json_height, enable_smart_resize, is_point=False):
        """Simple coordinate transformation for any coordinate format."""
        # Use CoordinateManager for the transformation
        if is_point:
            # For single points, create a minimal bbox and extract the transformed point
            bbox = [coords[0], coords[1], coords[0], coords[1]]
            transformed_bbox, _, _, _ = CoordinateManager.transform_geometry_complete(
                bbox, image_path, json_width, json_height, enable_smart_resize
            )
            return [transformed_bbox[0], transformed_bbox[1]]  # Return just x, y
        else:
            # For bbox format
            transformed_bbox, _, _, _ = CoordinateManager.transform_geometry_complete(
                coords, image_path, json_width, json_height, enable_smart_resize
            )
            return transformed_bbox

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

        logger.info(
            f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped"
        )

        if not all_samples:
            raise ValueError("No valid samples were processed")

        return all_samples

    def split_into_sets(
        self, all_samples: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split samples into train/val/teacher sets."""
        logger.info("📊 Selecting teacher samples...")

        # Select teacher samples
        teacher_samples, teacher_indices = self.teacher_selector.select_teachers(
            all_samples
        )

        # Remove teacher samples from student pool
        teacher_image_paths = {sample["images"][0] for sample in teacher_samples}
        student_samples = [
            sample
            for sample in all_samples
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
        teacher_samples: List[Dict],
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
                "total_objects": sum(
                    len(sample.get("objects", [])) for sample in all_samples
                ),
                "language": "chinese",
                "extraction_date": self._get_current_timestamp(),
                "description": "Complete vocabulary of labels extracted from the dataset for training prompt enhancement",
            },
            "statistics": {
                "unique_labels_count": len(unique_labels),
                "object_types_count": len(object_types),
                "properties_count": len(properties),
                "full_descriptions_count": len(full_descriptions),
            },
            "vocabulary": {
                "all_unique_labels": sorted(list(unique_labels)),
                "object_types": sorted(list(object_types)),
                "properties": sorted(list(properties)),
                "full_descriptions": sorted(list(full_descriptions)),
            },
            "usage_notes": {
                "training_prompts": "Use 'all_unique_labels' for comprehensive label-aware training",
                "object_detection": "Use 'object_types' for class-specific detection tasks",
                "attribute_prediction": "Use 'properties' for attribute/property prediction",
                "full_context": "Use 'full_descriptions' for complete description generation",
            },
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

        # Chinese-only mode - no token mapping needed

        # Final summary
        result = {
            "train": len(train_samples),
            "val": len(val_samples),
            "teacher": len(teacher_samples),
            "total_processed": len(all_samples),
        }

        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {result['train']} samples → train.jsonl")
        logger.info(f"   Validation: {result['val']} samples → val.jsonl")
        logger.info(f"   Teacher: {result['teacher']} samples → teacher.jsonl")
        logger.info(
            f"   Combined: {result['total_processed']} samples → all_samples.jsonl"
        )

        return result
