#!/usr/bin/env python3
"""
Streamlined Unified Data Processor

Consolidates all data processing functionality with simplified architecture.
Merged SampleExtractor directly into UnifiedProcessor to eliminate redundancy.
"""

import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# TeacherSelector now integrated as nested class
from data_conversion.config import DataConversionConfig
from data_conversion.coordinate_manager import (
    CoordinateManager,
    DataValidator,
    FormatConverter,
    StructureValidator,
)
from data_conversion.data_splitter import DataSplitter
from data_conversion.flexible_taxonomy_processor import HierarchicalProcessor
from data_conversion.utils.file_ops import FileOperations
from data_conversion.validation_manager import ValidationManager
from data_conversion.vision_process import ImageProcessor


# Configure UTF-8 encoding for stdout/stderr if supported
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

try:
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
except (AttributeError, TypeError):
    pass

logger = logging.getLogger(__name__)


class UnifiedProcessor:
    """Streamlined orchestrator for the unified data processing pipeline."""

    def __init__(self, config: DataConversionConfig):
        """Initialize processor with configuration."""
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
            object_types=set(config.object_types),
            label_hierarchy=self.label_hierarchy,
        )

        self.image_processor = ImageProcessor(config)
        self.teacher_selector = TeacherSelector(
            label_hierarchy=self.label_hierarchy,
            allowed_object_types=set(config.object_types),
            max_teachers=config.max_teachers,
            seed=config.seed,
        )
        self.data_splitter = DataSplitter(val_ratio=config.val_ratio, seed=config.seed)

        # Initialize validation manager with strict validation
        self.validation_manager = ValidationManager(
            validation_mode="strict",
            min_object_size=10,
            max_coordinate_value=50000,
            require_non_empty_description=True,
            check_coordinate_bounds=True,
        )

        # Track invalid objects and samples for reporting
        self.invalid_objects = []
        self.invalid_samples = []

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

    def _sanitize_description(self, desc: str) -> str:
        """Remove auxiliary occlusion tokens (contains '遮挡') per level/token.
        - Split by '/' into levels, by ',' within levels.
        - Drop any token containing '遮挡'.
        - Rejoin; drop empty levels.
        """
        if not desc or not isinstance(desc, str):
            return desc
        levels = [lvl.strip() for lvl in desc.split("/")]
        kept_levels = []
        for lvl in levels:
            if not lvl:
                continue
            tokens = [t.strip() for t in lvl.split(",")]
            kept_tokens = [t for t in tokens if t and ("遮挡" not in t)]
            if kept_tokens:
                kept_levels.append(",".join(kept_tokens))
        return "/".join(kept_levels)

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
            # Clean bbox coordinates for VLM training - ensure proper ordering and bounds
            bbox = [max(0, coord) for coord in bbox]  # Remove any negative coordinates
            # Ensure min <= max for both x and y
            x_min, y_min, x_max, y_max = bbox
            bbox = [
                min(x_min, x_max),
                min(y_min, y_max),
                max(x_min, x_max),
                max(y_min, y_max),
            ]

            properties = item.get("properties", {}) or {}
            content_dict = self.extract_content_fields(properties)

            if not content_dict or not self.is_allowed_object(content_dict):
                continue

            desc = FormatConverter.format_description(
                content_dict, self.config.response_types, "chinese"
            )
            if desc:
                if getattr(self.config, "remove_occlusion_tokens", False):
                    desc = self._sanitize_description(desc)
                if desc:
                    objects.append({"bbox_2d": bbox, "desc": desc})

        return objects

    def extract_objects_from_markresult(self, features: List[Dict]) -> List[Dict]:
        """Extract objects from markResult features with native geometry types."""
        # Use hierarchical processor for V2 data support
        objects = self.hierarchical_processor.extract_objects_from_markresult(features)
        # Apply sanitizer if configured
        if getattr(self.config, "remove_occlusion_tokens", False):
            for obj in objects:
                d = obj.get("desc", "")
                if d:
                    obj["desc"] = self._sanitize_description(d)
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
            _, _ = json_width, json_height

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
                logger.warning(f"No valid objects found in {json_path.name}")
                return None

            # Apply unified coordinate transformation pipeline
            sample_data = {"objects": objects}
            processed_sample, final_width, final_height = (
                self._process_sample_coordinates_unified(
                    sample_data, image_path, json_width, json_height, self.config.resize
                )
            )
            objects = processed_sample["objects"]

            # Process objects (validation step has been removed)
            objects = self._filter_valid_objects(
                objects, final_width, final_height, str(image_path.name)
            )

            if not objects:
                # Track invalid sample for reporting
                invalid_sample = {
                    "sample_id": str(json_path.name),
                    "reason": "no_valid_objects",
                    "original_object_count": len(sample_data.get("objects", [])),
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                }
                self.invalid_samples.append(invalid_sample)

                # Skip sample if no objects
                logger.warning(
                    f"No valid objects for {json_path.name}, skipping sample"
                )
                return None

            # Sort objects by position using first coordinate pair
            def get_sort_key(obj):
                if "bbox_2d" in obj:
                    return (obj["bbox_2d"][1], obj["bbox_2d"][0])  # y, x
                elif "quad" in obj:
                    return (obj["quad"][1], obj["quad"][0])  # y, x of first point
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

    def _filter_valid_objects(
        self, objects: List[Dict], img_w: int, img_h: int, image_id: str
    ) -> List[Dict]:
        """Filter objects using comprehensive validation with reporting."""
        if not objects:
            return objects

        # Re-sanitize descriptions right before validation/output, in case any slipped through
        if getattr(self.config, "remove_occlusion_tokens", False):
            for obj in objects:
                d = obj.get("desc", "")
                if d:
                    obj["desc"] = self._sanitize_description(d)

        # Use ValidationManager to filter objects
        valid_objects, invalid_objects = self.validation_manager.filter_valid_objects(
            objects, img_w, img_h, image_id
        )

        # Track invalid objects for reporting
        self.invalid_objects.extend(invalid_objects)

        # Log validation results
        if invalid_objects:
            logger.warning(
                f"Filtered out {len(invalid_objects)} invalid objects from {image_id}: "
                f"{len(valid_objects)} valid objects remaining"
            )
            logger.debug(
                f"Invalid objects details: {[obj.get('_validation_errors', []) for obj in invalid_objects]}"
            )

        return valid_objects

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
        elif "quad" in first_obj:
            # Create bbox from quad for dimension calculation
            quad = first_obj["quad"]
            x_coords = [quad[i] for i in range(0, len(quad), 2)]
            y_coords = [quad[i] for i in range(1, len(quad), 2)]
            geometry_input = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ]
        elif "line" in first_obj:
            # Create bbox from line for dimension calculation
            line = first_obj["line"]
            x_coords = [line[i] for i in range(0, len(line), 2)]
            y_coords = [line[i] for i in range(1, len(line), 2)]
            geometry_input = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            ]
        else:
            raise ValueError(f"No supported geometry type in first object: {first_obj}")

        # Use unified geometry transformation for dimension calculation
        _, _, final_width, final_height = CoordinateManager.transform_geometry_complete(
            geometry_input, image_path, json_width, json_height, enable_smart_resize
        )

        # Process all objects with their native geometry
        updated_objects = []
        for obj in sample_data["objects"]:
            # Transform coordinates based on geometry type
            updated_obj = obj.copy()

            if "bbox_2d" in obj:
                # Transform bbox coordinates
                transformed_coords = self._transform_coordinates(
                    obj["bbox_2d"],
                    image_path,
                    json_width,
                    json_height,
                    enable_smart_resize,
                )
                updated_obj["bbox_2d"] = [int(round(c)) for c in transformed_coords]

            elif "quad" in obj:
                # Transform quad coordinates (8 coordinates: x1,y1,x2,y2,x3,y3,x4,y4)
                coords = obj["quad"]
                transformed_coords = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = [coords[i], coords[i + 1]]
                        transformed_point = self._transform_coordinates(
                            point,
                            image_path,
                            json_width,
                            json_height,
                            enable_smart_resize,
                            is_point=True,
                        )
                        transformed_coords.extend(
                            [
                                int(round(transformed_point[0])),
                                int(round(transformed_point[1])),
                            ]
                        )
                updated_obj["quad"] = transformed_coords

            elif "line" in obj:
                # Transform line coordinates (sequence of x,y pairs)
                coords = obj["line"]
                transformed_coords = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point = [coords[i], coords[i + 1]]
                        transformed_point = self._transform_coordinates(
                            point,
                            image_path,
                            json_width,
                            json_height,
                            enable_smart_resize,
                            is_point=True,
                        )
                        transformed_coords.extend(
                            [
                                int(round(transformed_point[0])),
                                int(round(transformed_point[1])),
                            ]
                        )
                updated_obj["line"] = transformed_coords

            # Apply coordinate normalization after transformation
            normalized_obj = CoordinateManager.normalize_object_coordinates(
                updated_obj, final_width, final_height
            )
            updated_objects.append(normalized_obj)

        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects

        return updated_sample, final_width, final_height

    def _transform_coordinates(
        self,
        coords,
        image_path,
        json_width,
        json_height,
        enable_smart_resize,
        is_point=False,
    ):
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
        """Split samples into training, validation, and teacher sets.

        All samples remain in flat format without any teacher-student nesting.

        Args:
            all_samples: List of all processed samples

        Returns:
            Tuple of (train_samples, val_samples, teacher_samples)
        """
        logger.info(f"📊 Splitting {len(all_samples)} samples...")

        # Create data splitter
        data_splitter = DataSplitter(
            val_ratio=self.config.val_ratio, seed=self.config.seed
        )

        # Select teacher samples first (before train/val split)
        teacher_selector = self.teacher_selector
        teacher_samples, teacher_indices, teacher_stats = (
            teacher_selector.select_teachers(all_samples)
        )

        # Remove teacher samples from the pool
        remaining_samples = [
            s for i, s in enumerate(all_samples) if i not in teacher_indices
        ]

        # Split remaining samples into train and validation sets
        train_samples, val_samples = data_splitter.split(remaining_samples)

        logger.info(
            f"✅ Split complete: {len(train_samples)} train, {len(val_samples)} val, {len(teacher_samples)} teacher samples"
        )

        # Store teacher stats for later export
        self._teacher_pool_stats = teacher_stats

        return train_samples, val_samples, teacher_samples

    def write_outputs(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        teacher_samples: List[Dict],
    ) -> None:
        """Write output files in flat format only.

        All files are written in the same flat format: {'images': [...], 'objects': [...]}
        No nested teacher-student structure is used.
        """
        logger.info("📾 Writing output files...")

        # Write all files in flat format (no teacher-student nesting)
        FileOperations.write_jsonl(train_samples, self.output_dir / "train.jsonl")
        FileOperations.write_jsonl(val_samples, self.output_dir / "val.jsonl")
        FileOperations.write_jsonl(
            teacher_samples, self.output_dir / "teacher_pool.jsonl"
        )

        # Write all samples in flat format
        all_direct_samples = teacher_samples + train_samples + val_samples
        FileOperations.write_jsonl(
            all_direct_samples, self.output_dir / "all_samples.jsonl"
        )

        # Extract and export unique labels from original samples
        self._export_label_vocabulary(all_direct_samples)

        # Write teacher pool stats if available
        stats = getattr(self, "_teacher_pool_stats", None)
        if isinstance(stats, dict) and stats:
            stats_path = self.output_dir / "teacher_pool_stats.json"
            FileOperations.save_json_data(stats, stats_path, indent=2)
            logger.info(f"📈 Teacher pool stats written to {stats_path}")

        # Export validation reports
        self._export_validation_reports()

        logger.info("📊 Output files written successfully")
        logger.info(
            f"   📚 All samples (flat format): all_samples.jsonl ({len(all_direct_samples)} samples)"
        )
        logger.info(
            f"   🎓 Training files (flat format): train.jsonl ({len(train_samples)} samples), val.jsonl ({len(val_samples)} samples)"
        )
        logger.info(
            f"   📚 Teacher pool (flat format): teacher_pool.jsonl ({len(teacher_samples)} samples)"
        )

    def _convert_to_teacher_student_format(
        self, student_samples: List[Dict], teacher_pool: List[Dict]
    ) -> List[Dict]:
        """DEPRECATED: No longer needed as we use flat format only.

        This method is kept for backward compatibility but now simply returns
        the student samples without any transformation to teacher-student format.

        Args:
            student_samples: List of samples to use as students
            teacher_pool: Pool of teacher samples (unused)

        Returns:
            List of samples in flat format (unchanged)
        """
        # Simply return the samples without any transformation
        # This ensures all files use the flat format
        return student_samples.copy()

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

    def _export_validation_reports(self) -> None:
        """Export comprehensive validation reports including invalid samples and objects."""
        logger.info("📋 Exporting validation reports...")

        # Export ValidationManager reports
        validation_files = self.validation_manager.export_validation_reports(
            self.output_dir
        )

        # Export invalid objects with detailed error information
        if self.invalid_objects:
            invalid_objects_path = self.output_dir / "invalid_objects.jsonl"
            FileOperations.write_jsonl(self.invalid_objects, invalid_objects_path)
            logger.info(
                f"📋 Exported {len(self.invalid_objects)} invalid objects to {invalid_objects_path}"
            )

        # Export invalid samples summary
        if self.invalid_samples:
            invalid_samples_path = self.output_dir / "invalid_samples.jsonl"
            FileOperations.write_jsonl(self.invalid_samples, invalid_samples_path)
            logger.info(
                f"📋 Exported {len(self.invalid_samples)} invalid samples to {invalid_samples_path}"
            )

        # Generate validation summary statistics
        validation_summary = {
            "total_invalid_objects": len(self.invalid_objects),
            "total_invalid_samples": len(self.invalid_samples),
            "validation_manager_stats": self.validation_manager.generate_validation_summary(),
            "invalid_sample_reasons": {},
            "timestamp": self._get_current_timestamp(),
        }

        # Count reasons for invalid samples
        for sample in self.invalid_samples:
            reason = sample.get("reason", "unknown")
            validation_summary["invalid_sample_reasons"][reason] = (
                validation_summary["invalid_sample_reasons"].get(reason, 0) + 1
            )

        # Export validation summary
        summary_path = self.output_dir / "validation_report.json"
        FileOperations.save_json_data(validation_summary, summary_path, indent=2)

        logger.info("✅ Validation reports exported successfully")
        logger.info(f"   📊 {len(self.invalid_objects)} invalid objects")
        logger.info(f"   📊 {len(self.invalid_samples)} invalid samples")
        logger.info(f"   📋 Detailed reports: {list(validation_files.keys())}")

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

        # Validation steps have been removed

        # Final summary with validation statistics
        result = {
            "train": len(train_samples),
            "val": len(val_samples),
            "teacher": len(teacher_samples),
            "total_processed": len(all_samples),
            "total_invalid_objects": len(self.invalid_objects),
            "total_invalid_samples": len(self.invalid_samples),
            "validation_success_rate": self.validation_manager.valid_samples
            / max(1, self.validation_manager.total_samples_processed),
        }

        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {result['train']} samples → train.jsonl")
        logger.info(f"   Validation: {result['val']} samples → val.jsonl")
        logger.info(f"   Teacher: {result['teacher']} samples → teacher_pool.jsonl")
        logger.info(
            f"   Combined: {result['total_processed']} samples → all_samples.jsonl"
        )
        logger.info("🔍 Validation Summary:")
        logger.info(f"   Invalid objects filtered: {result['total_invalid_objects']}")
        logger.info(f"   Invalid samples skipped: {result['total_invalid_samples']}")
        logger.info(
            f"   Validation success rate: {result['validation_success_rate']:.2%}"
        )

        return result


class TeacherSelector:
    """Deterministic teacher pool builder using fixed vocabulary coverage.

    Implements a greedy set-cover over a fixed universe of canonical tokens
    derived from attribute taxonomy/mapping. Falls back to a simple free
    vocabulary if the taxonomy/mapping files are unavailable.
    """

    FREE_TOP_K_PER_TYPE: int = 50

    def __init__(
        self,
        label_hierarchy: Dict[str, List[str]],
        allowed_object_types: Set[str],
        max_teachers: int = 10,
        seed: int = 42,
    ):
        self.label_hierarchy = label_hierarchy
        self.allowed_object_types = allowed_object_types
        self.max_teachers = max_teachers
        self.seed = seed

        # Chinese → English object type mapping (static fallback)
        self.cn2en_types: Dict[str, str] = {
            "BBU设备": "bbu",
            "挡风板": "bbu_shield",
            "螺丝、光纤插头": "connect_point",
            "标签": "label",
            "光纤": "fiber",
            "电线": "wire",
        }

        # Load fixed universe from taxonomy/mapping if available
        (
            self.mode,
            self.universe_units,
            self.unit_to_objtypes,
            self.objtype_cn_by_en,
        ) = self._build_universe()

        logger.info(
            f"Initialized TeacherSelector mode={self.mode}, units={len(self.universe_units)}, max_teachers={max_teachers}"
        )

    def _build_universe(
        self,
    ) -> Tuple[str, Set[str], Dict[str, Set[str]], Dict[str, str]]:
        """Build coverage universe from fixed taxonomy/mapping; fallback to free mode.

        Returns:
            (mode, universe_units, unit_to_objtypes, objtype_cn_by_en)
        """
        base_dir = Path(__file__).parent
        taxonomy_path = base_dir / "attribute_taxonomy.json"
        mapping_path = base_dir / "hierarchical_attribute_mapping.json"

        if taxonomy_path.exists() and mapping_path.exists():
            try:
                taxonomy = FileOperations.load_json_data(taxonomy_path)
                mapping = FileOperations.load_json_data(mapping_path)
                units: Set[str] = set()
                unit_to_objtypes: Dict[str, Set[str]] = defaultdict(set)
                objtype_cn_by_en: Dict[str, str] = {}

                # From mapping: object types and attributes
                obj_types_map = mapping.get("object_types", {})
                for en_type, spec in obj_types_map.items():
                    cn_label = spec.get("chinese_label", "").strip()
                    if cn_label:
                        units.add(cn_label)
                        unit_to_objtypes[cn_label].add(en_type)
                        objtype_cn_by_en[en_type] = cn_label

                    attributes = spec.get("attributes", [])
                    for attr in attributes:
                        if attr.get("is_free_text"):
                            continue  # exclude free text
                        values = attr.get("values")
                        if isinstance(values, dict):
                            kv_pairs = list(values.items())
                        elif isinstance(values, list):
                            kv_pairs = [(v, v) for v in values]
                        else:
                            kv_pairs = []

                        for k, v in kv_pairs:
                            for token in self._split_tokens(k) | self._split_tokens(v):
                                units.add(token)
                                unit_to_objtypes[token].add(en_type)

                # Also incorporate explicit values from taxonomy attribute_groups
                attr_groups = taxonomy.get("attribute_groups", {})
                for group in attr_groups.values():
                    attrs = group.get("attributes", {})
                    for attr in attrs.values():
                        vals = attr.get("values")
                        if vals == "free_text":
                            continue
                        if isinstance(vals, dict):
                            candidates = list(vals.keys()) + list(vals.values())
                        elif isinstance(vals, list):
                            candidates = vals
                        else:
                            candidates = []
                        for c in candidates:
                            for token in self._split_tokens(c):
                                units.add(token)
                                # Heuristic association: if applies_to present, map tokens to those types
                                applies = attr.get("applies_to")
                                if isinstance(applies, list):
                                    for en_type in applies:
                                        unit_to_objtypes[token].add(en_type)

                # Filter units by allowed object types if mappings are exclusive
                if self.allowed_object_types:
                    filtered_units: Set[str] = set()
                    for token in units:
                        types = unit_to_objtypes.get(token)
                        if not types:
                            filtered_units.add(token)
                        elif set(types) & self.allowed_object_types:
                            filtered_units.add(token)
                    units = filtered_units

                return "fixed", units, unit_to_objtypes, objtype_cn_by_en
            except Exception as e:
                logger.warning(
                    f"Failed loading fixed taxonomy/mapping: {e}; falling back to free mode"
                )

        # Fallback: universe will be derived later from dataset (free mode)
        return "free", set(), defaultdict(set), {}

    @staticmethod
    def _split_tokens(text: str) -> Set[str]:
        if not isinstance(text, str) or not text.strip():
            return set()
        parts: List[str] = []
        for level in text.split("/"):
            parts.extend([p.strip() for p in level.split(",")])
        return {p for p in parts if p}

    @staticmethod
    def _geometry_types(sample: Dict) -> Set[str]:
        g: Set[str] = set()
        for obj in sample.get("objects", []):
            if "bbox_2d" in obj:
                g.add("bbox_2d")
            if "quad" in obj:
                g.add("quad")
            if "line" in obj:
                g.add("line")
        return g

    def _extract_units(self, sample: Dict) -> Set[str]:
        tokens: Set[str] = set()
        for obj in sample.get("objects", []):
            desc = obj.get("desc", "")
            if not desc:
                continue
            tokens |= self._split_tokens(desc)
        if self.mode == "fixed":
            return tokens & self.universe_units
        return tokens

    def _detect_brand(self, sample_tokens: Set[str]) -> str:
        for brand in ("华为", "中兴", "爱立信"):
            if brand in sample_tokens:
                return brand
        return "unknown"

    def _build_free_universe(self, samples: List[Dict]) -> Set[str]:
        # Build per-type token DF and select top-K per allowed type
        df_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in samples:
            seen_in_sample_by_type: Dict[str, Set[str]] = defaultdict(set)
            for obj in s.get("objects", []):
                desc = obj.get("desc", "")
                if not desc:
                    continue
                parts = [p.strip() for p in desc.split("/") if p.strip()]
                obj_type_cn = parts[0] if parts else ""
                en_type = self.cn2en_types.get(obj_type_cn, "unknown")
                if en_type != "unknown" and en_type not in self.allowed_object_types:
                    continue
                tokens = self._split_tokens(desc)
                seen_in_sample_by_type[en_type] |= tokens
            for en_type, toks in seen_in_sample_by_type.items():
                for t in toks:
                    df_by_type[en_type][t] += 1

        universe: Set[str] = set()
        for en_type, counter in df_by_type.items():
            if en_type == "unknown":
                continue
            # Select top-K tokens
            top = sorted(counter.items(), key=lambda kv: -kv[1])[
                : self.FREE_TOP_K_PER_TYPE
            ]
            for token, _ in top:
                universe.add(token)
                self.unit_to_objtypes[token].add(en_type)
        return universe

    def select_teachers(
        self, samples: List[Dict]
    ) -> Tuple[List[Dict], List[int], Dict]:
        """Deterministic greedy selection based on fixed rules.

        Returns:
            teacher_samples, selected_indices, stats_dict
        """
        if not samples:
            logger.warning("No samples provided for teacher selection")
            return [], [], {}

        if len(samples) <= self.max_teachers:
            logger.info(
                f"Using all {len(samples)} samples as teachers (below max_teachers)"
            )
            # Minimal stats
            stats = {
                "mode": self.mode,
                "pool_size": len(samples),
                "max_teachers": self.max_teachers,
            }
            return samples, list(range(len(samples))), stats

        # Build universe if in free mode
        if self.mode == "free":
            self.universe_units = self._build_free_universe(samples)

        # Precompute per-sample metadata
        per_sample_units: List[Set[str]] = []
        per_sample_tokens: List[Set[str]] = []
        per_sample_geometry: List[Set[str]] = []
        per_sample_brand: List[str] = []
        object_counts: List[int] = []
        for s in samples:
            tokens = (
                self._extract_units(s)
                if self.mode == "fixed"
                else self._extract_units(s)
            )
            per_sample_tokens.append(tokens)
            per_sample_units.append(
                tokens if self.mode == "fixed" else (tokens & self.universe_units)
            )
            per_sample_geometry.append(self._geometry_types(s))
            per_sample_brand.append(self._detect_brand(tokens))
            object_counts.append(len(s.get("objects", [])))

        median_objects = statistics.median(object_counts) if object_counts else 0
        selected: List[int] = []
        units_remaining: Set[str] = set(self.universe_units)
        selected_geometries: Set[str] = set()
        brand_counts: Dict[str, int] = defaultdict(int)

        # Helper to produce sort key per candidate (recomputed each round)
        def candidate_key(idx: int) -> Tuple:
            units_hit = len(per_sample_units[idx] & units_remaining)
            geom = per_sample_geometry[idx]
            has_line = 1 if "line" in geom else 0
            # Only activate line preference if fiber/wire-related units remain
            fiber_wire_remaining = any(
                (
                    self.unit_to_objtypes.get(u)
                    and (self.unit_to_objtypes[u] & {"fiber", "wire"})
                )
                for u in units_remaining
            )
            line_pref = has_line if fiber_wire_remaining else 0
            brand = per_sample_brand[idx]
            brand_balance = brand_counts[brand]
            geom_novelty = len(geom - selected_geometries)
            obj_delta = abs(object_counts[idx] - median_objects)
            # Lexicographic fallback by first image path
            img_path = ""
            imgs = samples[idx].get("images") or []
            if imgs:
                img_path = str(imgs[0])
            # Sort by descending units_hit, then line_pref, ascending brand_balance, descending geom_novelty, ascending obj_delta, lexicographic path
            return (
                -units_hit,
                -line_pref,
                brand_balance,
                -geom_novelty,
                obj_delta,
                img_path,
            )

        # Phase B: Greedy cover until cap or universe exhausted
        candidate_indices = list(range(len(samples)))
        while units_remaining and len(selected) < self.max_teachers:
            # Filter candidates that contribute anything
            contributing = [
                i for i in candidate_indices if (per_sample_units[i] & units_remaining)
            ]
            if not contributing:
                break
            best = min(contributing, key=candidate_key)
            selected.append(best)
            units_remaining -= per_sample_units[best]
            selected_geometries |= per_sample_geometry[best]
            brand_counts[per_sample_brand[best]] += 1
            candidate_indices.remove(best)

        # Phase C: Cap and fill for diversity if capacity remains
        if len(selected) < self.max_teachers and candidate_indices:

            def fill_key(idx: int) -> Tuple:
                geom = per_sample_geometry[idx]
                geom_novelty = len(geom - selected_geometries)
                brand = per_sample_brand[idx]
                brand_balance = brand_counts[brand]
                pool_units_seen = self.universe_units - units_remaining
                pool_novelty = len(per_sample_units[idx] - pool_units_seen)
                obj_delta = abs(object_counts[idx] - median_objects)
                img_path = ""
                imgs = samples[idx].get("images") or []
                if imgs:
                    img_path = str(imgs[0])
                # Prefer higher geom_novelty, lower brand_balance, higher pool_novelty, lower obj_delta
                return (
                    -geom_novelty,
                    brand_balance,
                    -pool_novelty,
                    obj_delta,
                    img_path,
                )

            remaining_sorted = sorted(candidate_indices, key=fill_key)
            for idx in remaining_sorted:
                if len(selected) >= self.max_teachers:
                    break
                selected.append(idx)
                selected_geometries |= per_sample_geometry[idx]
                brand_counts[per_sample_brand[idx]] += 1

        selected = sorted(set(selected))
        teacher_samples = [samples[i] for i in selected]

        # Stats
        covered_units = sorted(list(self.universe_units - units_remaining))
        stats: Dict = {
            "mode": self.mode,
            "pool_size": len(selected),
            "max_teachers": self.max_teachers,
            "universe_size": len(self.universe_units),
            "covered_units_count": len(covered_units),
            "coverage_ratio": float(len(covered_units) / len(self.universe_units))
            if self.universe_units
            else 0.0,
            "uncovered_units": sorted(list(units_remaining)) if units_remaining else [],
            "brand_distribution": dict(brand_counts),
            "geometry_presence": sorted(list(selected_geometries)),
        }
        pool_object_counts = [object_counts[i] for i in selected]
        if pool_object_counts:
            sorted_counts = sorted(pool_object_counts)
            p95_idx = max(
                0, min(len(sorted_counts) - 1, int(0.95 * (len(sorted_counts) - 1)))
            )
            stats["object_count_summary"] = {
                "min": sorted_counts[0],
                "median": statistics.median(sorted_counts),
                "p95": sorted_counts[p95_idx],
            }

        logger.info(
            f"Teacher pool built: size={len(selected)} coverage={stats.get('covered_units_count')}/{stats.get('universe_size')}"
        )

        return teacher_samples, selected, stats


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    from config import setup_logging, validate_config

    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")

    # Required arguments
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with JSON/image files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--language",
        choices=["chinese", "english"],
        required=True,
        help="Language mode",
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset name for organized output (auto-detected from input_dir if not provided)",
    )

    # Processing arguments - REQUIRED
    parser.add_argument(
        "--object_types",
        nargs="+",
        required=True,
        help="Object types to include (e.g., bbu label fiber connect_point)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="Validation split ratio (e.g., 0.1)",
    )
    parser.add_argument(
        "--max_teachers",
        type=int,
        required=True,
        help="Maximum teacher samples (e.g., 10)",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed (e.g., 42)"
    )

    # Processing options - OPTIONAL
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Advanced processing options
    parser.add_argument(
        "--geometry_diversity_weight",
        type=float,
        default=4.0,
        help="Weight for geometry diversity in teacher selection",
    )

    # New filtering flag
    parser.add_argument(
        "--strip_occlusion",
        action="store_true",
        help="Remove tokens containing '遮挡' from descriptions",
    )

    args = parser.parse_args()

    # Create configuration from arguments
    config = DataConversionConfig.from_args(args)

    # Setup logging
    setup_logging(config)

    # Validate configuration
    validate_config(config)

    # Create and run unified processor
    processor = UnifiedProcessor(config)
    result = processor.process()

    # Print result for compatibility
    print(f"\n✅ Processing complete: {result}")


if __name__ == "__main__":
    main()
