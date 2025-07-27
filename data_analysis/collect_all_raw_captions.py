#!/usr/bin/env python3
"""
Script to collect all Chinese annotations from raw JSON files.

This script traverses a directory containing JSON annotation files and extracts
all Chinese text content from the 'contentZh' fields in the annotations.

Usage:
    python collect_all_raw_captions.py <input_folder> [--output <output_file>]

Example:
    python collect_all_raw_captions.py ds_v2 --output chinese_annotations.json
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_chinese_content(data: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
    """
    Extract all Chinese content from a JSON annotation file.

    Args:
        data: Parsed JSON data
        file_path: Path to the source file

    Returns:
        List of extracted Chinese annotations with metadata
    """
    annotations = []

    # Check if this is a valid annotation file
    if 'markResult' not in data or 'features' not in data['markResult']:
        logger.warning(f"Invalid annotation format in {file_path}")
        return annotations

    # Extract file metadata
    file_info = {
        'file_path': file_path,
        'image_info': data.get('info', {}),
        'tag_info': data.get('tagInfo', {}),
        'version': data.get('version', '')
    }

    # Process each feature/annotation
    features = data['markResult']['features']
    for i, feature in enumerate(features):
        if 'properties' not in feature:
            continue

        properties = feature['properties']

        # Extract Chinese content
        content_zh = properties.get('contentZh', {})
        if not content_zh:
            continue

        # Extract English content for comparison
        content_en = properties.get('content', {})

        # Extract geometry information
        geometry = feature.get('geometry', {})

        annotation = {
            'annotation_id': i,
            'geometry_type': geometry.get('type', ''),
            'label_en': content_en.get('label', ''),
            'chinese_content': content_zh,
            'file_info': file_info
        }

        annotations.append(annotation)

    return annotations


def collect_all_annotations(input_folder: str) -> List[Dict[str, Any]]:
    """
    Collect all Chinese annotations from JSON files in the input folder.

    Args:
        input_folder: Path to folder containing JSON files

    Returns:
        List of all collected annotations
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    all_annotations = []
    json_files = list(input_path.glob("*.json"))

    logger.info(f"Found {len(json_files)} JSON files in {input_folder}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = extract_chinese_content(data, str(json_file))
            all_annotations.extend(annotations)

            if annotations:
                logger.info(f"Extracted {len(annotations)} annotations from {json_file.name}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {json_file}: {e}")
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")

    logger.info(f"Total annotations collected: {len(all_annotations)}")
    return all_annotations


def create_unique_patterns(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create unique annotation patterns by deduplicating based on content.

    Args:
        annotations: List of collected annotations

    Returns:
        List of unique annotation patterns with occurrence counts
    """
    pattern_counts = defaultdict(int)
    pattern_examples = {}

    for ann in annotations:
        # Create a signature for this annotation pattern
        chinese_content = ann['chinese_content']
        label_en = ann['label_en']
        geometry_type = ann['geometry_type']

        # Convert chinese_content to a hashable signature
        content_items = []
        for key in sorted(chinese_content.keys()):
            value = chinese_content[key]
            if isinstance(value, list):
                value = tuple(sorted(value))
            content_items.append((key, value))

        signature = (label_en, geometry_type, tuple(content_items))
        pattern_counts[signature] += 1

        # Keep the first example of each pattern
        if signature not in pattern_examples:
            pattern_examples[signature] = ann

    # Convert back to list format with counts
    unique_patterns = []
    for signature, count in pattern_counts.items():
        example = pattern_examples[signature]
        # Convert signature to string for JSON serialization
        signature_str = f"{signature[0]}|{signature[1]}|{len(signature[2])}_fields"

        pattern = {
            'pattern_signature': signature_str,
            'occurrence_count': count,
            'label_en': example['label_en'],
            'geometry_type': example['geometry_type'],
            'chinese_content': example['chinese_content']
        }
        unique_patterns.append(pattern)

    # Sort by occurrence count (descending)
    unique_patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)

    return unique_patterns


def analyze_annotations(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the collected annotations to generate comprehensive statistics.

    Args:
        annotations: List of collected annotations

    Returns:
        Dictionary containing analysis results
    """
    # Basic counts
    label_counts = Counter()
    geometry_counts = Counter()
    chinese_field_counts = defaultdict(Counter)

    # Advanced analysis
    label_geometry_combinations = Counter()
    chinese_field_combinations = defaultdict(set)
    files_per_label = defaultdict(set)

    # Collect all unique Chinese fields and their values
    all_chinese_fields = set()

    for ann in annotations:
        label_en = ann['label_en']
        geometry_type = ann['geometry_type']
        chinese_content = ann['chinese_content']
        file_path = ann['file_info']['file_path']

        label_counts[label_en] += 1
        geometry_counts[geometry_type] += 1
        label_geometry_combinations[(label_en, geometry_type)] += 1
        files_per_label[label_en].add(file_path)

        # Analyze Chinese content fields
        field_set = set()
        for field, value in chinese_content.items():
            all_chinese_fields.add(field)
            field_set.add(field)

            # Handle different value types
            if isinstance(value, list):
                for item in value:
                    chinese_field_counts[field][str(item)] += 1
            else:
                chinese_field_counts[field][str(value)] += 1

        # Track field combinations for each label
        chinese_field_combinations[label_en].add(frozenset(field_set))

    # Create unique patterns
    unique_patterns = create_unique_patterns(annotations)

    # Calculate additional statistics
    label_field_diversity = {}
    for label, field_sets in chinese_field_combinations.items():
        label_field_diversity[label] = len(field_sets)

    files_coverage = {}
    for label, file_set in files_per_label.items():
        files_coverage[label] = len(file_set)

    # Convert tuple keys to strings for JSON serialization
    label_geometry_combinations_str = {
        f"{label}+{geom}": count
        for (label, geom), count in label_geometry_combinations.items()
    }

    analysis = {
        'total_annotations': len(annotations),
        'unique_patterns': len(unique_patterns),
        'label_distribution': dict(label_counts),
        'geometry_distribution': dict(geometry_counts),
        'label_geometry_combinations': label_geometry_combinations_str,
        'chinese_fields': sorted(all_chinese_fields),
        'chinese_field_values': dict(chinese_field_counts),
        'files_processed': len(set(ann['file_info']['file_path'] for ann in annotations)),
        'files_coverage_per_label': files_coverage,
        'field_diversity_per_label': label_field_diversity,
        'patterns': unique_patterns
    }

    return analysis


def save_results(annotations: List[Dict[str, Any]], analysis: Dict[str, Any], output_file: str, summary_only: bool = False):
    """
    Save the collected annotations and analysis to a JSON file.

    Args:
        annotations: List of collected annotations
        analysis: Analysis results
        output_file: Path to output file
        summary_only: If True, save only summary without detailed patterns
    """
    # Extract patterns from analysis for cleaner output
    patterns = analysis.pop('patterns', [])

    if summary_only:
        # For summary-only mode, just save the analysis summary
        results = {
            'summary': analysis,
            'pattern_count': len(patterns),
            'top_10_patterns': [
                {
                    'signature': p['pattern_signature'],
                    'count': p['occurrence_count'],
                    'label': p['label_en'],
                    'geometry': p['geometry_type']
                }
                for p in patterns[:10]
            ]
        }
    else:
        results = {
            'summary': analysis,
            'unique_patterns': patterns,
            'raw_annotations_sample': annotations[:10] if annotations else []  # Keep only first 10 as examples
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")
    if summary_only:
        logger.info(f"Saved summary with {len(patterns)} unique patterns (top 10 included)")
    else:
        logger.info(f"Saved {len(patterns)} unique patterns and {len(annotations)} total annotations")


def print_summary(analysis: Dict[str, Any]):
    """Print a comprehensive summary of the analysis results."""
    print("\n" + "="*80)
    print("CHINESE ANNOTATIONS DATASET ANALYSIS SUMMARY")
    print("="*80)

    # Basic statistics
    print(f"📊 BASIC STATISTICS")
    print(f"  Total annotations collected: {analysis['total_annotations']:,}")
    print(f"  Unique annotation patterns: {analysis['unique_patterns']:,}")
    print(f"  Deduplication ratio: {analysis['unique_patterns']/analysis['total_annotations']:.2%}")
    print(f"  Files processed: {analysis['files_processed']:,}")

    # Label distribution with coverage
    print(f"\n🏷️  LABEL DISTRIBUTION & COVERAGE")
    for label, count in analysis['label_distribution'].items():
        file_coverage = analysis['files_coverage_per_label'][label]
        field_diversity = analysis['field_diversity_per_label'][label]
        coverage_pct = file_coverage / analysis['files_processed'] * 100
        print(f"  {label:15} | Count: {count:4} | Files: {file_coverage:3} ({coverage_pct:5.1f}%) | Field patterns: {field_diversity}")

    # Geometry distribution
    print(f"\n📐 GEOMETRY TYPE DISTRIBUTION")
    for geom_type, count in analysis['geometry_distribution'].items():
        pct = count / analysis['total_annotations'] * 100
        print(f"  {geom_type:15} | {count:4} ({pct:5.1f}%)")

    # Label-Geometry combinations
    print(f"\n🔗 LABEL-GEOMETRY COMBINATIONS")
    for combo_str, count in sorted(analysis['label_geometry_combinations'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
        label, geom = combo_str.split('+')
        print(f"  {label:15} + {geom:15} | {count:4}")

    # Chinese fields overview
    print(f"\n🇨🇳 CHINESE ANNOTATION FIELDS ({len(analysis['chinese_fields'])} total)")
    for field in analysis['chinese_fields']:
        field_total = sum(analysis['chinese_field_values'][field].values())
        print(f"  - {field} ({field_total} values)")

    # Top patterns by occurrence
    patterns = analysis.get('patterns', [])
    if patterns:
        print(f"\n🔥 TOP 10 MOST COMMON ANNOTATION PATTERNS")
        for i, pattern in enumerate(patterns[:10], 1):
            print(f"  {i:2}. {pattern['label_en']:15} | {pattern['geometry_type']:15} | "
                  f"Count: {pattern['occurrence_count']:4} | Fields: {len(pattern['chinese_content'])}")

    # Field value diversity
    print(f"\n📈 FIELD VALUE DIVERSITY (Top 5 fields)")
    field_diversity = [(field, len(values)) for field, values in analysis['chinese_field_values'].items()]
    field_diversity.sort(key=lambda x: x[1], reverse=True)

    for field, unique_values in field_diversity[:5]:
        print(f"\n  {field}:")
        values_counter = Counter(analysis['chinese_field_values'][field])
        for value, count in values_counter.most_common(5):
            pct = count / sum(values_counter.values()) * 100
            print(f"    '{value}': {count} ({pct:.1f}%)")
        if unique_values > 5:
            print(f"    ... and {unique_values - 5} more values")


def main():
    parser = argparse.ArgumentParser(description='Collect Chinese annotations from raw JSON files')
    parser.add_argument('input_folder', help='Path to folder containing JSON annotation files')
    parser.add_argument('--output', '-o', default='chinese_annotations_analysis.json',
                       help='Output file path (default: chinese_annotations_analysis.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--patterns-only', '-p', action='store_true',
                       help='Save only unique patterns (set version) without raw annotations')
    parser.add_argument('--summary-only', '-s', action='store_true',
                       help='Generate only a concise summary without detailed patterns')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Collect all annotations
        annotations = collect_all_annotations(args.input_folder)

        if not annotations:
            logger.warning("No annotations found!")
            return

        # Analyze the collected data
        analysis = analyze_annotations(annotations)

        # Save results based on mode
        if args.summary_only:
            # For summary-only mode, save just the key statistics
            save_results([], analysis, args.output, summary_only=True)
        elif args.patterns_only:
            # For set version, don't save raw annotations
            save_results([], analysis, args.output)
        else:
            save_results(annotations, analysis, args.output)

        # Print summary
        print_summary(analysis)

        # Additional insights
        print(f"\n💡 INSIGHTS")
        patterns = analysis.get('patterns', [])
        if patterns:
            total_unique = len(patterns)
            high_freq_patterns = len([p for p in patterns if p['occurrence_count'] > 5])
            single_occurrence = len([p for p in patterns if p['occurrence_count'] == 1])

            print(f"  • {high_freq_patterns} patterns occur more than 5 times")
            print(f"  • {single_occurrence} patterns occur only once ({single_occurrence/total_unique:.1%})")
            print(f"  • Most common pattern occurs {patterns[0]['occurrence_count']} times")

            # Label diversity
            label_pattern_counts = defaultdict(int)
            for pattern in patterns:
                label_pattern_counts[pattern['label_en']] += 1

            print(f"  • Pattern diversity by label:")
            for label, pattern_count in sorted(label_pattern_counts.items(), key=lambda x: x[1], reverse=True):
                total_label_annotations = analysis['label_distribution'][label]
                diversity_ratio = pattern_count / total_label_annotations
                print(f"    - {label}: {pattern_count} unique patterns from {total_label_annotations} annotations (diversity: {diversity_ratio:.2f})")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()