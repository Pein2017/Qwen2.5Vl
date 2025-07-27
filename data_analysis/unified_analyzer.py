#!/usr/bin/env python3
"""
Simplified Dataset Analyzer for Qwen2.5-VL

Basic dataset analysis tool focused on statistics and sample extraction.
Removed complex validation logic that was based on incorrect data structure assumptions.

Features:
- Dataset statistics and analysis  
- Representative sample extraction for prompt engineering
- Export capabilities for analysis results
"""

import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UnifiedDatasetAnalyzer:
    """Simplified analyzer for dataset statistics and sample extraction."""

    def __init__(self, data_path: str):
        """Initialize with dataset path."""
        self.data_path = Path(data_path)
        self.samples = []
        self.analysis_results = {}

    def load_data(self) -> None:
        """Load dataset from JSONL file."""
        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.samples.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        logger.info(f"Loaded {len(self.samples)} samples")

    def _extract_descriptions(self, sample: Dict[str, Any]) -> List[str]:
        """Extract object descriptions from a sample."""
        descriptions = []
        
        # Handle the actual data format used by the pipeline
        if "objects" in sample and isinstance(sample["objects"], list):
            # New format: objects is a list of objects with desc field
            for obj in sample["objects"]:
                if isinstance(obj, dict) and "desc" in obj:
                    descriptions.append(obj["desc"])
        
        return descriptions

    def analyze_dataset(self) -> Dict[str, Any]:
        """Perform basic dataset analysis."""
        logger.info("Analyzing dataset...")

        # Basic statistics
        total_samples = len(self.samples)
        all_descriptions = []
        objects_per_sample = []

        # Process each sample
        for sample in self.samples:
            descriptions = self._extract_descriptions(sample)
            objects_per_sample.append(len(descriptions))
            all_descriptions.extend(descriptions)

        # Calculate statistics
        self.analysis_results = {
            "dataset_overview": {
                "total_samples": total_samples,
                "total_objects": len(all_descriptions),
                "avg_objects_per_sample": statistics.mean(objects_per_sample)
                if objects_per_sample
                else 0,
                "median_objects_per_sample": statistics.median(objects_per_sample)
                if objects_per_sample
                else 0,
                "min_objects_per_sample": min(objects_per_sample)
                if objects_per_sample
                else 0,
                "max_objects_per_sample": max(objects_per_sample)
                if objects_per_sample
                else 0,
                "unique_descriptions": len(set(all_descriptions)),
            },
            "description_distribution": {
                "most_common": Counter(all_descriptions).most_common(10),
                "all_counts": dict(Counter(all_descriptions)),
            },
            "complexity_analysis": self._analyze_complexity(objects_per_sample),
        }

        logger.info("Dataset analysis complete")
        return self.analysis_results

    def _analyze_complexity(self, objects_per_sample: List[int]) -> Dict[str, Any]:
        """Analyze sample complexity based on object count."""
        if not objects_per_sample:
            return {}

        return {
            "sparse_samples": len([x for x in objects_per_sample if x <= 3]),
            "medium_samples": len([x for x in objects_per_sample if 4 <= x <= 10]),
            "dense_samples": len([x for x in objects_per_sample if x > 10]),
            "complexity_distribution": dict(Counter(objects_per_sample)),
        }

    def extract_representative_samples(
        self, num_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """Extract representative samples for prompt engineering."""
        logger.info(f"Extracting {num_samples} representative samples...")

        # Calculate complexity scores for each sample
        scored_samples = []
        for i, sample in enumerate(self.samples):
            objects = sample.get("objects", {})
            ref_items = objects.get("ref", [])

            # Complexity score based on number of objects
            num_objects = len(ref_items)
            complexity_score = num_objects * 2.5

            # Rarity score based on unique object types
            object_types = []
            for ref in ref_items:
                components = self._parse_ref(ref)
                object_types.append(components["object_type"])

            # Calculate rarity based on frequency in dataset
            rarity_score = 0
            if (
                hasattr(self, "analysis_results")
                and "object_type_distribution" in self.analysis_results
            ):
                all_counts = self.analysis_results["object_type_distribution"][
                    "all_counts"
                ]
                total_objects = sum(all_counts.values())
                for obj_type in set(object_types):
                    frequency = all_counts.get(obj_type, 1)
                    rarity_score += (total_objects / frequency) * 0.1

            scored_samples.append(
                {
                    **sample,
                    "_index": i,
                    "_complexity_score": complexity_score,
                    "_rarity_score": rarity_score,
                    "_num_objects": num_objects,
                    "_object_types": object_types,
                }
            )

        # Sort and categorize samples
        scored_samples.sort(key=lambda x: x["_complexity_score"])

        # Select diverse samples
        selected = []

        # 1. Sparse sample (fewest objects)
        sparse_samples = [s for s in scored_samples if s["_num_objects"] <= 3]
        if sparse_samples:
            selected.append({**sparse_samples[0], "_category": "sparse"})

        # 2. Dense sample (most objects)
        dense_samples = [s for s in scored_samples if s["_num_objects"] >= 15]
        if dense_samples:
            selected.append({**dense_samples[-1], "_category": "dense"})

        # 3. Medium complexity sample
        medium_samples = [s for s in scored_samples if 6 <= s["_num_objects"] <= 12]
        if medium_samples:
            mid_idx = len(medium_samples) // 2
            selected.append({**medium_samples[mid_idx], "_category": "medium"})

        # 4. High rarity sample
        high_rarity = sorted(
            scored_samples, key=lambda x: x["_rarity_score"], reverse=True
        )
        if high_rarity:
            selected.append({**high_rarity[0], "_category": "rare"})

        # 5. Diverse sample (good mix of object types)
        diverse_samples = sorted(
            scored_samples, key=lambda x: len(set(x["_object_types"])), reverse=True
        )
        if diverse_samples:
            selected.append({**diverse_samples[0], "_category": "diverse"})

        # Remove duplicates and limit to requested number
        unique_selected = []
        seen_indices = set()
        for sample in selected:
            if sample["_index"] not in seen_indices:
                unique_selected.append(sample)
                seen_indices.add(sample["_index"])
                if len(unique_selected) >= num_samples:
                    break

        logger.info(f"Selected {len(unique_selected)} representative samples")
        return unique_selected

    def inspect_sample(self, sample_index: int) -> Dict[str, Any]:
        """Inspect a specific sample in detail."""
        if sample_index >= len(self.samples):
            raise ValueError(
                f"Sample index {sample_index} out of range (0-{len(self.samples) - 1})"
            )

        sample = self.samples[sample_index]
        objects = sample.get("objects", {})
        ref_items = objects.get("ref", [])
        bbox_items = objects.get("bbox", [])

        # Parse object details using unified parser
        object_details = []
        for i, ref in enumerate(ref_items):
            bbox = bbox_items[i] if i < len(bbox_items) else None

            components = self._parse_ref(ref)

            object_details.append(
                {
                    "index": i,
                    "bbox": bbox,
                    "object_type": components.get("object_type", ""),
                    "property": components.get("property", ""),
                    "extra_info": components.get("extra_info", ""),
                    "raw_ref": ref,
                }
            )

        return {
            "sample_index": sample_index,
            "image_path": sample.get("images", [""])[0],
            "num_objects": len(ref_items),
            "image_dimensions": {
                "width": sample.get("width", 0),
                "height": sample.get("height", 0),
            },
            "objects": object_details,
            "raw_sample": sample,
        }

    # Object validation functionality
    def validate_objects(
        self, raw_objects: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Validate all objects in the dataset or provided raw objects list."""
        logger.info(
            f"Validating objects from {self.data_path if raw_objects is None else 'provided list'}"
        )

        # Use provided raw objects or load from samples
        objects_to_validate = raw_objects if raw_objects is not None else self.samples

        if not objects_to_validate and not raw_objects:
            # Load data if not already loaded and no raw objects provided
            if not self.samples:
                self.load_data()
                objects_to_validate = self.samples

        # Reset validation results
        self.validation_results = {
            "total_objects": 0,
            "valid_objects": 0,
            "invalid_objects": 0,
            "validation_errors": [],
            "error_categories": {},
        }

        # Process each object
        for line_num, obj in enumerate(objects_to_validate, 1):
            is_valid, errors = self._validate_object(obj, line_num)

            self.validation_results["total_objects"] += 1
            if is_valid:
                self.validation_results["valid_objects"] += 1
            else:
                self.validation_results["invalid_objects"] += 1
                for error in errors:
                    self.validation_results["validation_errors"].append(error)

                    # Track error categories
                    category = error["error_category"]
                    if category not in self.validation_results["error_categories"]:
                        self.validation_results["error_categories"][category] = 0
                    self.validation_results["error_categories"][category] += 1

        logger.info(
            f"Validation complete: {self.validation_results['valid_objects']} valid, "
            f"{self.validation_results['invalid_objects']} invalid objects"
        )
        return self.validation_results

    def _validate_object(
        self, obj: Dict[str, Any], line_num: int
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate a single object and return validation results."""
        errors = []

        # Required fields validation
        required_fields = ["image", "desc", "geometry"]
        for field in required_fields:
            if field not in obj:
                errors.append(
                    {
                        "line": line_num,
                        "object": obj,
                        "error_category": "missing_field",
                        "error_message": f"Missing required field: {field}",
                        "severity": "critical",
                    }
                )

        # If missing critical fields, return early
        if len(errors) > 0:
            return False, errors

        # Validate geometry
        geometry_errors = self._validate_geometry(obj["geometry"], line_num, obj)
        errors.extend(geometry_errors)

        # Validate description
        desc_errors = self._validate_description(obj["desc"], line_num, obj)
        errors.extend(desc_errors)

        # Validate image reference
        image_errors = self._validate_image(obj["image"], line_num, obj)
        errors.extend(image_errors)

        return len(errors) == 0, errors

    def _validate_geometry(
        self, geometry: List[int], line_num: int, obj: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate object geometry (bounding box coordinates)."""
        errors = []

        # Check if geometry has exactly 4 values
        if len(geometry) != 4:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_geometry_format",
                    "error_message": f"Geometry must have exactly 4 values (x1, y1, x2, y2), got {len(geometry)}",
                    "severity": "critical",
                }
            )
            return errors

        # Check if all values are integers
        if not all(isinstance(coord, int) for coord in geometry):
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_geometry_type",
                    "error_message": "All geometry coordinates must be integers",
                    "severity": "critical",
                }
            )

        # Check for negative coordinates
        if any(coord < 0 for coord in geometry):
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "negative_coordinates",
                    "error_message": f"Negative coordinates detected: {geometry}",
                    "severity": "critical",
                }
            )

        # Check if x2 > x1 and y2 > y1
        x1, y1, x2, y2 = geometry
        if x2 <= x1:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_bbox_width",
                    "error_message": f"Invalid bounding box width: x2 ({x2}) must be greater than x1 ({x1})",
                    "severity": "high",
                }
            )

        if y2 <= y1:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_bbox_height",
                    "error_message": f"Invalid bounding box height: y2 ({y2}) must be greater than y1 ({y1})",
                    "severity": "high",
                }
            )

        # Check for zero-area bounding boxes
        if x1 == x2 or y1 == y2:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "zero_area_bbox",
                    "error_message": f"Zero-area bounding box detected: {geometry}",
                    "severity": "high",
                }
            )

        # Check for unreasonably large coordinates (potential errors)
        if any(coord > 10000 for coord in geometry):
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "suspicious_large_coordinates",
                    "error_message": f"Suspiciously large coordinates detected: {geometry}",
                    "severity": "medium",
                }
            )

        return errors

    def _validate_description(
        self, desc: str, line_num: int, obj: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate object description."""
        errors = []

        # Check if description is empty
        if not desc.strip():
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "empty_description",
                    "error_message": "Object description is empty",
                    "severity": "high",
                }
            )
            return errors

        # Check description format (should be object_type/property/extra_info)
        parts = [p.strip() for p in desc.split("/") if p.strip()]

        if len(parts) < 1:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_description_format",
                    "error_message": "Description must have at least an object type",
                    "severity": "high",
                }
            )

        # Validate object type
        if len(parts) >= 1 and len(parts[0]) < 2:
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_object_type",
                    "error_message": f"Object type too short: '{parts[0]}'",
                    "severity": "medium",
                }
            )

        return errors

    def _validate_image(
        self, image: str, line_num: int, obj: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate image reference."""
        errors = []

        # Check if image reference is empty
        if not image.strip():
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "empty_image_reference",
                    "error_message": "Image reference is empty",
                    "severity": "critical",
                }
            )
            return errors

        # Check image filename format
        if not image.endswith((".jpg", ".jpeg", ".png")):
            errors.append(
                {
                    "line": line_num,
                    "object": obj,
                    "error_category": "invalid_image_format",
                    "error_message": f"Image filename has invalid extension: {image}",
                    "severity": "medium",
                }
            )

        return errors

    def export_analysis(self, output_path: str) -> None:
        """Export analysis results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis exported to {output_path}")

    def export_validation_results(self, output_path: str) -> None:
        """Export validation results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Validation results exported to {output_path}")

    def export_invalid_objects(self, output_path: str) -> None:
        """Export invalid objects to JSONL file with error details."""
        with open(output_path, "w", encoding="utf-8") as f:
            for error in self.validation_results["validation_errors"]:
                error_info = {
                    "object": error["object"],
                    "error_category": error["error_category"],
                    "error_message": error["error_message"],
                    "severity": error["severity"],
                    "line": error["line"],
                }
                f.write(json.dumps(error_info, ensure_ascii=False) + "\n")
        logger.info(f"Invalid objects exported to {output_path}")

    def export_invalid_samples(self, output_path: str) -> None:
        """Export invalid objects in the same format as all_samples for visualization."""
        # Group errors by object
        invalid_objects = {}
        for error in self.validation_results["validation_errors"]:
            obj = error["object"]
            obj_id = json.dumps(obj)  # Use JSON string as dict key

            if obj_id not in invalid_objects:
                invalid_objects[obj_id] = {"object": obj, "errors": []}

            invalid_objects[obj_id]["errors"].append(
                {
                    "category": error["error_category"],
                    "message": error["error_message"],
                    "severity": error["severity"],
                }
            )

        # Convert to visualization-friendly format
        with open(output_path, "w", encoding="utf-8") as f:
            for obj_data in invalid_objects.values():
                obj = obj_data["object"]
                errors = obj_data["errors"]

                # Create a sample in the same format as all_samples
                sample = {
                    "images": [obj.get("image", "")],
                    "width": 1000,  # Default values
                    "height": 1000,
                    "objects": {
                        "ref": [obj.get("desc", "")],
                        "bbox": [obj.get("geometry", [0, 0, 0, 0])],
                    },
                    "validation_errors": errors,
                }

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Invalid samples exported to {output_path}")

    def export_samples(self, samples: List[Dict[str, Any]], output_path: str) -> None:
        """Export samples to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Samples exported to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        if not self.analysis_results:
            logger.warning(
                "No analysis results available. Run analyze_dataset() first."
            )
            return

        overview = self.analysis_results["dataset_overview"]
        print("\n=== DATASET SUMMARY ===")
        print(f"Total samples: {overview['total_samples']}")
        print(f"Total objects: {overview['total_objects']}")
        print(f"Average objects per sample: {overview['avg_objects_per_sample']:.2f}")
        print(f"Median objects per sample: {overview['median_objects_per_sample']}")
        print(
            f"Object count range: {overview['min_objects_per_sample']}-{overview['max_objects_per_sample']}"
        )
        print(f"Unique object types: {overview['unique_object_types']}")
        print(f"Unique properties: {overview['unique_properties']}")
        print(f"Unique extra infos: {overview['unique_extra_infos']}")

        print("\n=== TOP OBJECT TYPES ===")
        for obj_type, count in self.analysis_results["object_type_distribution"][
            "most_common"
        ][:5]:
            print(f"  {obj_type}: {count}")

        print("\n=== TOP PROPERTIES ===")
        for prop, count in self.analysis_results["property_distribution"][
            "most_common_properties"
        ][:5]:
            if prop:  # Skip empty properties
                print(f"  {prop}: {count}")

        if "complexity_analysis" in self.analysis_results:
            complexity = self.analysis_results["complexity_analysis"]
            print("\n=== COMPLEXITY DISTRIBUTION ===")
            print(f"Sparse samples (≤3 objects): {complexity.get('sparse_samples', 0)}")
            print(
                f"Medium samples (4-10 objects): {complexity.get('medium_samples', 0)}"
            )
            print(f"Dense samples (>10 objects): {complexity.get('dense_samples', 0)}")

    def print_validation_summary(self) -> None:
        """Print a summary of the validation results."""
        if not self.validation_results:
            logger.warning(
                "No validation results available. Run validate_objects() first."
            )
            return

        print("\n=== VALIDATION SUMMARY ===")
        print(f"Total objects: {self.validation_results['total_objects']}")
        print(f"Valid objects: {self.validation_results['valid_objects']}")
        print(f"Invalid objects: {self.validation_results['invalid_objects']}")

        if self.validation_results["error_categories"]:
            print("\n=== ERROR CATEGORIES ===")
            for category, count in sorted(
                self.validation_results["error_categories"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {category}: {count}")

        if self.validation_results["validation_errors"]:
            print("\n=== SAMPLE ERRORS ===")
            for error in self.validation_results["validation_errors"][:5]:
                print(f"  Line {error['line']}: {error['error_message']}")
                print(f"    Object: {error['object']}")

            if len(self.validation_results["validation_errors"]) > 5:
                print(
                    f"  ... and {len(self.validation_results['validation_errors']) - 5} more errors"
                )

    def attempt_fix_invalid_objects(self) -> Dict[str, Any]:
        """Attempt to fix invalid objects when possible and return fix statistics."""
        logger.info("Attempting to fix invalid objects...")

        if (
            not self.validation_results
            or not self.validation_results["validation_errors"]
        ):
            logger.warning(
                "No validation results available. Run validate_objects() first."
            )
            return {"fixed_objects": 0, "unfixable_objects": 0, "fixes_by_category": {}}

        # Group errors by object line number
        errors_by_line = {}
        for error in self.validation_results["validation_errors"]:
            line = error["line"]
            if line not in errors_by_line:
                errors_by_line[line] = []
            errors_by_line[line].append(error)

        # Statistics
        fix_stats = {
            "fixed_objects": 0,
            "unfixable_objects": 0,
            "fixes_by_category": {},
            "fixed_objects_data": [],
        }

        # Process each object with errors
        for line, errors in errors_by_line.items():
            # Get the object from the first error (all errors for same line have same object)
            obj = errors[0]["object"].copy()

            # Check if we can fix this object
            fixed, fixed_obj, applied_fixes = self._fix_object(obj, errors)

            if fixed:
                fix_stats["fixed_objects"] += 1
                fix_stats["fixed_objects_data"].append(
                    {
                        "original": obj,
                        "fixed": fixed_obj,
                        "applied_fixes": applied_fixes,
                    }
                )

                # Track fixes by category
                for fix in applied_fixes:
                    category = fix["category"]
                    if category not in fix_stats["fixes_by_category"]:
                        fix_stats["fixes_by_category"][category] = 0
                    fix_stats["fixes_by_category"][category] += 1
            else:
                fix_stats["unfixable_objects"] += 1

        logger.info(
            f"Fix attempt complete: {fix_stats['fixed_objects']} fixed, "
            f"{fix_stats['unfixable_objects']} unfixable"
        )
        return fix_stats

    def _fix_object(
        self, obj: Dict[str, Any], errors: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Attempt to fix a single object based on its errors.

        Returns:
            Tuple containing:
            - Boolean indicating if the fix was successful
            - The fixed object (or original if unfixable)
            - List of applied fixes with descriptions
        """
        fixed_obj = obj.copy()
        applied_fixes = []

        # Check if we have geometry errors that can be fixed
        geometry_errors = [
            e
            for e in errors
            if e["error_category"].startswith("invalid_")
            or e["error_category"] == "zero_area_bbox"
        ]

        if geometry_errors and "geometry" in fixed_obj:
            # Get the current geometry
            x1, y1, x2, y2 = fixed_obj["geometry"]
            original_geometry = [x1, y1, x2, y2]

            # Fix common geometry issues

            # 1. Swap coordinates if x2 < x1 or y2 < y1
            if x2 < x1:
                x1, x2 = x2, x1
                applied_fixes.append(
                    {
                        "category": "swapped_x_coordinates",
                        "description": f"Swapped x coordinates: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            if y2 < y1:
                y1, y2 = y2, y1
                applied_fixes.append(
                    {
                        "category": "swapped_y_coordinates",
                        "description": f"Swapped y coordinates: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            # 2. Fix zero-area bounding boxes
            if x1 == x2:
                # Add 1 pixel width
                x2 = x1 + 1
                applied_fixes.append(
                    {
                        "category": "fixed_zero_width",
                        "description": f"Added 1px width: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            if y1 == y2:
                # Add 1 pixel height
                y2 = y1 + 1
                applied_fixes.append(
                    {
                        "category": "fixed_zero_height",
                        "description": f"Added 1px height: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            # 3. Fix negative coordinates
            if x1 < 0:
                x1 = 0
                applied_fixes.append(
                    {
                        "category": "fixed_negative_x1",
                        "description": f"Set negative x1 to 0: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            if y1 < 0:
                y1 = 0
                applied_fixes.append(
                    {
                        "category": "fixed_negative_y1",
                        "description": f"Set negative y1 to 0: [{original_geometry}] -> [{x1}, {y1}, {x2}, {y2}]",
                    }
                )

            # Update the geometry if any fixes were applied
            if applied_fixes:
                fixed_obj["geometry"] = [x1, y1, x2, y2]

        # Return the fixed object and list of applied fixes
        return bool(applied_fixes), fixed_obj, applied_fixes

    def export_fixed_objects(self, fix_stats: Dict[str, Any], output_path: str) -> None:
        """Export fixed objects to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for fix_data in fix_stats["fixed_objects_data"]:
                f.write(json.dumps(fix_data["fixed"], ensure_ascii=False) + "\n")
        logger.info(f"Fixed objects exported to {output_path}")

    def print_fix_summary(self, fix_stats: Dict[str, Any]) -> None:
        """Print a summary of the fix attempts."""
        print("\n=== FIX ATTEMPT SUMMARY ===")
        print(f"Fixed objects: {fix_stats['fixed_objects']}")
        print(f"Unfixable objects: {fix_stats['unfixable_objects']}")

        if fix_stats["fixes_by_category"]:
            print("\n=== FIXES BY CATEGORY ===")
            for category, count in sorted(
                fix_stats["fixes_by_category"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {category}: {count}")

        if fix_stats["fixed_objects_data"]:
            print("\n=== SAMPLE FIXES ===")
            for i, fix_data in enumerate(fix_stats["fixed_objects_data"][:3]):
                print(f"\nFix #{i + 1}:")
                print(f"  Original: {fix_data['original']['geometry']}")
                print(f"  Fixed: {fix_data['fixed']['geometry']}")
                for fix in fix_data["applied_fixes"]:
                    print(f"  - {fix['description']}")

            if len(fix_stats["fixed_objects_data"]) > 3:
                print(
                    f"\n  ... and {len(fix_stats['fixed_objects_data']) - 3} more fixes"
                )


# Predefined configuration (no CLI)
DATA_PATH = "data_conversion/qwen_combined.jsonl"
ANALYSIS_OUTPUT_PATH = "analysis_results.json"


def main():
    """Run object validation and export the results."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Unified Dataset Analyzer for Qwen2.5-VL"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_conversion/qwen_combined.jsonl",
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_analysis",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--attempt_fix", action="store_true", help="Attempt to fix invalid objects"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run analyzer
    analyzer = UnifiedDatasetAnalyzer(args.data_path)
    analyzer.load_data()

    # Run validation on the dataset
    analyzer.validate_objects()

    # Export validation results directly to output directory
    validation_output = output_dir / "validation_results.json"
    invalid_objects_output = output_dir / "invalid_objects.jsonl"
    invalid_samples_output = output_dir / "invalid_samples.jsonl"

    analyzer.export_validation_results(str(validation_output))
    analyzer.export_invalid_objects(str(invalid_objects_output))
    analyzer.export_invalid_samples(str(invalid_samples_output))

    # Attempt to fix invalid objects if requested
    if args.attempt_fix and analyzer.validation_results["invalid_objects"] > 0:
        fix_stats = analyzer.attempt_fix_invalid_objects()
        if fix_stats["fixed_objects"] > 0:
            fixed_objects_output = output_dir / "fixed_objects.jsonl"
            analyzer.export_fixed_objects(fix_stats, str(fixed_objects_output))
            analyzer.print_fix_summary(fix_stats)

    # Print summary
    analyzer.print_validation_summary()


if __name__ == "__main__":
    main()
