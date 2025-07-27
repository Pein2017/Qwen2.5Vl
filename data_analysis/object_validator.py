#!/usr/bin/env python3
"""
Object Validator Tool for Qwen2.5-VL Telecommunications Dataset

This script validates objects in the dataset and provides detailed error reports.
It extends the UnifiedDatasetAnalyzer to specifically handle validation tasks.

Features:
- Validates object annotations (geometry, descriptions, etc.)
- Provides detailed error reports with specific failure reasons
- Categorizes validation errors by type
- Exports validation results for further analysis
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Ensure project root is on module search path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from data_analysis.unified_analyzer import UnifiedDatasetAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ObjectValidator(UnifiedDatasetAnalyzer):
    """Validator for telecommunications quality inspection dataset objects."""

    def __init__(self, data_path: str):
        """Initialize with dataset path."""
        super().__init__(data_path)
        self.validation_results = {
            "total_objects": 0,
            "valid_objects": 0,
            "invalid_objects": 0,
            "validation_errors": [],
            "error_categories": {},
        }

    def validate_objects(self) -> Dict[str, Any]:
        """Validate all objects in the dataset."""
        logger.info(f"Validating objects from {self.data_path}")

        # Load data if not already loaded
        if not self.samples:
            self.load_data()

        # Reset validation results
        self.validation_results = {
            "total_objects": 0,
            "valid_objects": 0,
            "invalid_objects": 0,
            "validation_errors": [],
            "error_categories": {},
        }

        # Process each object in the dataset
        for line_num, obj in enumerate(self.samples, 1):
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

            if len(self.validation_results["validation_errors"]) > 5:
                print(
                    f"  ... and {len(self.validation_results['validation_errors']) - 5} more errors"
                )


def main():
    """Run object validation on the invalid objects file."""
    validator = ObjectValidator("data/ds_v2_full/invalid_objects.jsonl")
    validator.validate_objects()
    validator.print_validation_summary()
    validator.export_validation_results("data_analysis/validation_results.json")
    validator.export_invalid_objects("data_analysis/invalid_objects_with_details.jsonl")


if __name__ == "__main__":
    main()
