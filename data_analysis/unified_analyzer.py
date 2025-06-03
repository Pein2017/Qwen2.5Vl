#!/usr/bin/env python3
"""
Unified Data Analysis Tool for Qwen2.5-VL Telecommunications Dataset

This script consolidates dataset analysis, sample inspection, and prompt sample extraction
into a single comprehensive tool for better maintainability and reduced file count.

Features:
- Dataset statistics and analysis
- Sample inspection and visualization
- Representative sample extraction for prompt engineering
- Export capabilities for different formats
- Uses core modules for consistent processing
"""

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for core_modules import
sys.path.append(str(Path(__file__).parent.parent / "data_conversion"))

from data_conversion.core_modules import ResponseFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UnifiedDatasetAnalyzer:
    """Unified analyzer for telecommunications quality inspection dataset."""

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

    def analyze_dataset(self) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis."""
        logger.info("Analyzing dataset...")

        # Basic statistics
        total_samples = len(self.samples)
        all_objects = []
        object_types = []
        properties = []
        extra_infos = []
        objects_per_sample = []

        # Process each sample
        for sample in self.samples:
            objects = sample.get("objects", {})
            ref_items = objects.get("ref", [])

            objects_per_sample.append(len(ref_items))
            all_objects.extend(ref_items)

            # Parse object descriptions using ResponseFormatter
            for ref in ref_items:
                if isinstance(ref, str):
                    components = ResponseFormatter.parse_description_string(ref)
                    object_types.append(components.get("object_type", ""))
                    properties.append(components.get("property", ""))
                    extra_infos.append(components.get("extra_info", ""))
                elif isinstance(ref, dict):
                    # Handle dict format with standardized field names
                    object_types.append(ref.get("object_type", ""))
                    properties.append(ref.get("property", ""))
                    extra_infos.append(ref.get("extra_info", ""))

        # Calculate statistics
        self.analysis_results = {
            "dataset_overview": {
                "total_samples": total_samples,
                "total_objects": len(all_objects),
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
                "unique_object_types": len(set(object_types)),
                "unique_properties": len(set(properties)),
                "unique_extra_infos": len(set(extra_infos)),
            },
            "object_type_distribution": {
                "most_common": Counter(object_types).most_common(10),
                "rarest": Counter(object_types).most_common()[-10:],
                "all_counts": dict(Counter(object_types)),
            },
            "property_distribution": {
                "most_common_properties": Counter(properties).most_common(10),
                "most_common_extra_infos": Counter(extra_infos).most_common(10),
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
                if isinstance(ref, str):
                    components = ResponseFormatter.parse_description_string(ref)
                    object_types.append(components.get("object_type", ""))
                elif isinstance(ref, dict):
                    object_types.append(ref.get("object_type", ""))

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

        # Parse object details using ResponseFormatter
        object_details = []
        for i, ref in enumerate(ref_items):
            bbox = bbox_items[i] if i < len(bbox_items) else None

            if isinstance(ref, str):
                components = ResponseFormatter.parse_description_string(ref)
            elif isinstance(ref, dict):
                components = {
                    "object_type": ref.get("object_type", ""),
                    "property": ref.get("property", ""),
                    "extra_info": ref.get("extra_info", ""),
                }
            else:
                components = {"object_type": str(ref), "property": "", "extra_info": ""}

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

    def export_analysis(self, output_path: str) -> None:
        """Export analysis results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis exported to {output_path}")

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


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Unified Data Analysis Tool for Qwen2.5-VL Dataset"
    )
    parser.add_argument("data_path", help="Path to the JSONL dataset file")
    parser.add_argument(
        "--analyze", action="store_true", help="Perform dataset analysis"
    )
    parser.add_argument(
        "--extract-samples",
        type=int,
        default=5,
        help="Extract N representative samples (default: 5)",
    )
    parser.add_argument("--inspect", type=int, help="Inspect specific sample by index")
    parser.add_argument(
        "--export-analysis", help="Export analysis results to JSON file"
    )
    parser.add_argument(
        "--export-samples", help="Export representative samples to JSON file"
    )
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = UnifiedDatasetAnalyzer(args.data_path)
    analyzer.load_data()

    # Perform analysis if requested
    if args.analyze or args.summary or args.export_analysis:
        analyzer.analyze_dataset()

    # Print summary
    if args.summary:
        analyzer.print_summary()

    # Extract representative samples
    if args.extract_samples > 0:
        samples = analyzer.extract_representative_samples(args.extract_samples)
        print("\n=== REPRESENTATIVE SAMPLES ===")
        for i, sample in enumerate(samples, 1):
            print(
                f"{i}. {sample.get('_category', 'unknown').upper()}: {sample.get('images', [''])[0]}"
            )
            print(
                f"   Objects: {sample.get('_num_objects', 0)}, Complexity: {sample.get('_complexity_score', 0):.1f}"
            )

        if args.export_samples:
            analyzer.export_samples(samples, args.export_samples)

    # Inspect specific sample
    if args.inspect is not None:
        try:
            inspection = analyzer.inspect_sample(args.inspect)
            print(f"\n=== SAMPLE INSPECTION (Index {args.inspect}) ===")
            print(f"Image: {inspection['image_path']}")
            print(f"Objects: {inspection['num_objects']}")
            print(
                f"Dimensions: {inspection['image_dimensions']['width']}x{inspection['image_dimensions']['height']}"
            )
            print("\nObject Details:")
            for obj in inspection["objects"]:
                print(f"  {obj['index']}: {obj['object_type']}")
                print(f"    Property: {obj['property']}")
                print(f"    Extra Info: {obj['extra_info']}")
                print(f"    BBox: {obj['bbox']}")
        except ValueError as e:
            logger.error(f"Inspection error: {e}")

    # Export analysis
    if args.export_analysis:
        analyzer.export_analysis(args.export_analysis)


if __name__ == "__main__":
    main()
