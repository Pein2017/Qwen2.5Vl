#!/usr/bin/env python3
"""
V2 to Training Data Pipeline

Complete pipeline for converting V2 annotation data to hierarchical training format
supporting bbox_2d, square, and line geometries with flexible taxonomy.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


try:
    from .flexible_taxonomy_processor import AnnotationSample, FlexibleTaxonomyProcessor
except ImportError:
    from flexible_taxonomy_processor import AnnotationSample, FlexibleTaxonomyProcessor

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V2TrainingPipeline:
    """Complete pipeline for V2 data conversion."""

    def __init__(self, taxonomy_path: Optional[str] = None):
        self.processor = FlexibleTaxonomyProcessor(taxonomy_path)
        self.stats = {
            "total_files_processed": 0,
            "total_samples_created": 0,
            "object_type_counts": {},
            "geometry_format_counts": {},
            "attribute_group_counts": {},
            "failed_processing": 0,
        }

    def process_directory(
        self,
        input_dir: str,
        output_file: str,
        filter_object_types: Optional[List[str]] = None,
        min_description_length: int = 10,
    ) -> Dict:
        """
        Process entire directory of V2 JSON files.

        Args:
            input_dir: Directory containing V2 JSON files
            output_file: Output JSONL file for training data
            filter_object_types: Only include specified object types
            min_description_length: Minimum description length to include

        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        all_samples = []
        json_files = list(input_path.glob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            try:
                samples = self.processor.process_v2_file(str(json_file))

                # Apply filters
                filtered_samples = self._apply_filters(
                    samples, filter_object_types, min_description_length
                )

                all_samples.extend(filtered_samples)
                self.stats["total_files_processed"] += 1

                logger.info(
                    f"Processed {json_file.name}: {len(samples)} -> {len(filtered_samples)} samples"
                )

            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                self.stats["failed_processing"] += 1

        # Update statistics
        self._update_statistics(all_samples)

        # Save training data
        self._save_training_data(all_samples, output_file)

        # Generate summary report
        report = self._generate_report()

        logger.info(
            f"Pipeline completed: {len(all_samples)} samples saved to {output_file}"
        )
        return report

    def _apply_filters(
        self,
        samples: List[AnnotationSample],
        filter_object_types: Optional[List[str]],
        min_description_length: int,
    ) -> List[AnnotationSample]:
        """Apply filtering criteria to samples."""
        filtered = []

        for sample in samples:
            # Filter by object type
            if filter_object_types and sample.object_type not in filter_object_types:
                continue

            # Filter by description length
            if len(sample.description) < min_description_length:
                continue

            # Filter out empty coordinate lists
            if not sample.coordinates or all(c == 0 for c in sample.coordinates):
                continue

            filtered.append(sample)

        return filtered

    def _update_statistics(self, samples: List[AnnotationSample]):
        """Update processing statistics."""
        self.stats["total_samples_created"] = len(samples)

        for sample in samples:
            # Object type counts
            obj_type = sample.object_type
            self.stats["object_type_counts"][obj_type] = (
                self.stats["object_type_counts"].get(obj_type, 0) + 1
            )

            # Geometry format counts
            geo_format = sample.geometry_format
            self.stats["geometry_format_counts"][geo_format] = (
                self.stats["geometry_format_counts"].get(geo_format, 0) + 1
            )

            # Attribute group counts
            for group_name in sample.grouped_attributes:
                self.stats["attribute_group_counts"][group_name] = (
                    self.stats["attribute_group_counts"].get(group_name, 0) + 1
                )

    def _save_training_data(self, samples: List[AnnotationSample], output_file: str):
        """Save samples in training format."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                training_data = sample.to_training_format()
                f.write(json.dumps(training_data, ensure_ascii=False) + "\n")

    def _generate_report(self) -> Dict:
        """Generate comprehensive processing report."""
        report = {
            "summary": {
                "total_files_processed": self.stats["total_files_processed"],
                "total_samples_created": self.stats["total_samples_created"],
                "failed_processing": self.stats["failed_processing"],
                "success_rate": f"{(self.stats['total_files_processed'] / (self.stats['total_files_processed'] + self.stats['failed_processing'])) * 100:.1f}%"
                if self.stats["total_files_processed"] + self.stats["failed_processing"]
                > 0
                else "N/A",
            },
            "object_types": dict(
                sorted(
                    self.stats["object_type_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
            "geometry_formats": dict(
                sorted(
                    self.stats["geometry_format_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
            "attribute_groups": dict(
                sorted(
                    self.stats["attribute_group_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }
        return report

    def create_curriculum_datasets(
        self,
        input_dir: str,
        output_dir: str,
        curriculum_stages: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Create multiple datasets for curriculum learning.

        Args:
            input_dir: Directory with V2 JSON files
            output_dir: Directory to save curriculum datasets
            curriculum_stages: Dictionary mapping stage names to object types
        """
        if curriculum_stages is None:
            curriculum_stages = {
                "basic_objects": ["connect_point", "label"],
                "equipment": ["bbu", "bbu_shield"],
                "infrastructure": ["fiber", "wire"],
                "complete": [
                    "connect_point",
                    "label",
                    "bbu",
                    "bbu_shield",
                    "fiber",
                    "wire",
                ],
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for stage_name, object_types in curriculum_stages.items():
            stage_file = output_path / f"{stage_name}_dataset.jsonl"

            logger.info(f"Creating curriculum stage: {stage_name}")

            report = self.process_directory(
                input_dir, str(stage_file), filter_object_types=object_types
            )

            # Save stage report
            report_file = output_path / f"{stage_name}_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Stage {stage_name}: {report['summary']['total_samples_created']} samples"
            )

    def validate_output(self, output_file: str) -> Dict:
        """Validate generated training data."""
        validation_results = {
            "total_lines": 0,
            "valid_json_lines": 0,
            "geometry_format_distribution": {},
            "description_length_stats": {"min": float("inf"), "max": 0, "avg": 0},
            "coordinate_validation": {"valid": 0, "invalid": 0},
        }

        descriptions = []

        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    validation_results["total_lines"] += 1

                    try:
                        data = json.loads(line.strip())
                        validation_results["valid_json_lines"] += 1

                        # Check required fields
                        desc = data.get("desc", "")
                        if desc:
                            descriptions.append(desc)

                        # Geometry format distribution
                        for geo_type in ["bbox_2d", "square", "line"]:
                            if geo_type in data:
                                validation_results["geometry_format_distribution"][
                                    geo_type
                                ] = (
                                    validation_results[
                                        "geometry_format_distribution"
                                    ].get(geo_type, 0)
                                    + 1
                                )

                                # Coordinate validation
                                coords = data[geo_type]
                                if self._validate_coordinates(coords, geo_type):
                                    validation_results["coordinate_validation"][
                                        "valid"
                                    ] += 1
                                else:
                                    validation_results["coordinate_validation"][
                                        "invalid"
                                    ] += 1

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {line_num}")

        except FileNotFoundError:
            logger.error(f"Output file not found: {output_file}")
            return validation_results

        # Description statistics
        if descriptions:
            desc_lengths = [len(desc) for desc in descriptions]
            validation_results["description_length_stats"] = {
                "min": min(desc_lengths),
                "max": max(desc_lengths),
                "avg": sum(desc_lengths) / len(desc_lengths),
            }

        return validation_results

    def _validate_coordinates(self, coords: List[float], geo_type: str) -> bool:
        """Validate coordinate format."""
        if not isinstance(coords, list):
            return False

        if geo_type == "bbox_2d":
            return len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)
        elif geo_type == "square":
            return len(coords) == 8 and all(isinstance(c, (int, float)) for c in coords)
        elif geo_type == "line":
            return (
                len(coords) >= 4
                and len(coords) % 2 == 0
                and all(isinstance(c, (int, float)) for c in coords)
            )

        return False


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert V2 annotations to training data"
    )
    parser.add_argument("input_dir", help="Directory containing V2 JSON files")
    parser.add_argument("output_file", help="Output JSONL file")
    parser.add_argument("--filter-types", nargs="+", help="Filter by object types")
    parser.add_argument(
        "--min-desc-length", type=int, default=10, help="Minimum description length"
    )
    parser.add_argument(
        "--create-curriculum", action="store_true", help="Create curriculum datasets"
    )
    parser.add_argument("--curriculum-dir", help="Directory for curriculum datasets")
    parser.add_argument(
        "--validate", action="store_true", help="Validate output after processing"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = V2TrainingPipeline()

    if args.create_curriculum:
        curriculum_dir = args.curriculum_dir or str(
            Path(args.output_file).parent / "curriculum"
        )
        pipeline.create_curriculum_datasets(args.input_dir, curriculum_dir)
    else:
        # Process single dataset
        report = pipeline.process_directory(
            args.input_dir,
            args.output_file,
            filter_object_types=args.filter_types,
            min_description_length=args.min_desc_length,
        )

        # Print report
        print("\n=== Processing Report ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        # Validation
        if args.validate:
            print("\n=== Validation Results ===")
            validation = pipeline.validate_output(args.output_file)
            print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
