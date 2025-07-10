#!/usr/bin/env python3
"""
Detailed performance analysis for BBU equipment detection model.

This script provides comprehensive analysis of model performance beyond overall metrics,
including per-class analysis, spatial error patterns, semantic confusion matrices,
and failure/success pattern detection.

Usage:
    python eval/detailed_analysis.py \
        --predictions_file experiments/1_teacher/val/inference/predictions.json \
        --output_dir experiments/1_teacher/val/detailed_analysis \
        --label_vocab data/label_vocabulary.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BBoxAnalysis:
    """Analysis results for a single bounding box prediction."""
    iou: float
    predicted_label: str
    ground_truth_label: str
    bbox_pred: List[int]
    bbox_gt: List[int]
    image_width: int
    image_height: int
    
    @property
    def bbox_size_pred(self) -> float:
        """Calculate predicted bbox area as percentage of image."""
        w = self.bbox_pred[2] - self.bbox_pred[0]
        h = self.bbox_pred[3] - self.bbox_pred[1]
        return (w * h) / (self.image_width * self.image_height)
    
    @property
    def bbox_size_gt(self) -> float:
        """Calculate ground truth bbox area as percentage of image."""
        w = self.bbox_gt[2] - self.bbox_gt[0]
        h = self.bbox_gt[3] - self.bbox_gt[1]
        return (w * h) / (self.image_width * self.image_height)
    
    @property
    def center_error(self) -> float:
        """Calculate center point error in pixels."""
        pred_center = [(self.bbox_pred[0] + self.bbox_pred[2]) / 2, 
                       (self.bbox_pred[1] + self.bbox_pred[3]) / 2]
        gt_center = [(self.bbox_gt[0] + self.bbox_gt[2]) / 2, 
                     (self.bbox_gt[1] + self.bbox_gt[3]) / 2]
        return np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a class or overall."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    avg_iou: float
    avg_center_error: float
    size_error_ratio: float
    
    @classmethod
    def from_matches(cls, matches: List[BBoxAnalysis], predictions: List[dict], ground_truths: List[dict]) -> 'PerformanceMetrics':
        """Calculate metrics from bbox analysis results."""
        tp = len([m for m in matches if m.iou > 0.5])
        fp = len(predictions) - tp
        fn = len(ground_truths) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_iou = np.mean([m.iou for m in matches if m.iou > 0.5]) if matches else 0.0
        avg_center_error = np.mean([m.center_error for m in matches if m.iou > 0.5]) if matches else 0.0
        
        size_errors = [abs(m.bbox_size_pred - m.bbox_size_gt) / m.bbox_size_gt 
                      for m in matches if m.iou > 0.5 and m.bbox_size_gt > 0]
        size_error_ratio = np.mean(size_errors) if size_errors else 0.0
        
        return cls(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_iou=avg_iou,
            avg_center_error=avg_center_error,
            size_error_ratio=size_error_ratio
        )


class DetailedAnalyzer:
    """Comprehensive analyzer for BBU detection model performance."""
    
    def __init__(self, label_vocab_path: str):
        """Initialize analyzer with label vocabulary."""
        self.logger = logging.getLogger(__name__)
        
        # Load label vocabulary
        with open(label_vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.all_labels = set(vocab_data['vocabulary']['full_descriptions'])
        self.object_types = set(vocab_data['vocabulary']['object_types'])
        self.vendors = {'中兴', '华为', '爱立信'}
        
        # Analysis results storage
        self.bbox_analyses: List[BBoxAnalysis] = []
        self.class_metrics: Dict[str, PerformanceMetrics] = {}
        self.confusion_matrix: Dict[str, Counter] = defaultdict(Counter)
        self.failure_patterns: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions(self, predictions: List[dict], ground_truths: List[dict], 
                         image_width: int, image_height: int) -> List[BBoxAnalysis]:
        """Match predictions with ground truth using IoU threshold."""
        analyses = []
        used_gt_indices = set()
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in used_gt_indices:
                    continue
                    
                iou = self.calculate_iou(pred['bbox_2d'], gt['bbox_2d'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                used_gt_indices.add(best_gt_idx)
                gt = ground_truths[best_gt_idx]
                
                analysis = BBoxAnalysis(
                    iou=best_iou,
                    predicted_label=pred['label'],
                    ground_truth_label=gt['label'],
                    bbox_pred=pred['bbox_2d'],
                    bbox_gt=gt['bbox_2d'],
                    image_width=image_width,
                    image_height=image_height
                )
                analyses.append(analysis)
                
                # Update confusion matrix
                self.confusion_matrix[gt['label']][pred['label']] += 1
            else:
                # False positive - no matching ground truth
                analysis = BBoxAnalysis(
                    iou=0.0,
                    predicted_label=pred['label'],
                    ground_truth_label="",
                    bbox_pred=pred['bbox_2d'],
                    bbox_gt=[0, 0, 0, 0],
                    image_width=image_width,
                    image_height=image_height
                )
                analyses.append(analysis)
                
                # Update confusion matrix for false positive
                self.confusion_matrix[""][pred['label']] += 1
        
        # Handle false negatives - ground truths without matches
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in used_gt_indices:
                self.confusion_matrix[gt['label']][""] += 1
        
        return analyses
    
    def analyze_predictions(self, predictions_file: str) -> Dict[str, Any]:
        """Analyze predictions from inference results."""
        self.logger.info(f"Loading predictions from {predictions_file}")
        
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        
        self.logger.info(f"Analyzing {len(predictions_data)} samples")
        
        # Group analyses by class
        class_analyses: Dict[str, List[BBoxAnalysis]] = defaultdict(list)
        all_analyses = []
        
        for sample_idx, sample in enumerate(predictions_data):
            try:
                # Parse ground truth and predictions
                gt_objs = json.loads(sample['ground_truth'])
                pred_objs = json.loads(sample['pred_result'])
                
                # Match predictions with ground truth
                sample_analyses = self.match_predictions(
                    pred_objs, gt_objs, 
                    sample['width'], sample['height']
                )
                
                all_analyses.extend(sample_analyses)
                
                # Group by class
                for analysis in sample_analyses:
                    if analysis.ground_truth_label:
                        class_analyses[analysis.ground_truth_label].append(analysis)
                        
            except json.JSONDecodeError as e:
                self.logger.warning(f"Skipping sample {sample_idx} due to JSON error: {e}")
                self.logger.debug(f"Sample data: {sample}")
                continue
            except Exception as e:
                self.logger.warning(f"Skipping sample {sample_idx} due to error: {e}")
                continue
        
        self.bbox_analyses = all_analyses
        
        # Calculate per-class metrics
        for class_name, analyses in class_analyses.items():
            # Get all predictions and ground truths for this class
            class_predictions = [{'bbox_2d': a.bbox_pred, 'label': a.predicted_label} 
                               for a in analyses if a.predicted_label == class_name]
            class_ground_truths = [{'bbox_2d': a.bbox_gt, 'label': a.ground_truth_label} 
                                 for a in analyses if a.ground_truth_label == class_name]
            
            self.class_metrics[class_name] = PerformanceMetrics.from_matches(
                analyses, class_predictions, class_ground_truths
            )
        
        # Analyze failure and success patterns
        self._analyze_failure_patterns()
        self._analyze_success_patterns()
        
        return self._generate_analysis_report()
    
    def _analyze_failure_patterns(self):
        """Identify common failure patterns."""
        self.logger.info("Analyzing failure patterns")
        
        # Low IoU but correct label
        low_iou_correct = [a for a in self.bbox_analyses 
                          if a.iou < 0.5 and a.predicted_label == a.ground_truth_label]
        
        # High IoU but wrong label
        high_iou_wrong = [a for a in self.bbox_analyses 
                         if a.iou > 0.5 and a.predicted_label != a.ground_truth_label]
        
        # Common misclassifications
        common_mistakes = []
        for gt_label, pred_counter in self.confusion_matrix.items():
            if gt_label:  # Skip empty labels
                total = sum(pred_counter.values())
                for pred_label, count in pred_counter.most_common(3):
                    if pred_label != gt_label and count > 1:
                        common_mistakes.append({
                            'ground_truth': gt_label,
                            'predicted': pred_label,
                            'count': count,
                            'percentage': count / total * 100
                        })
        
        self.failure_patterns = [
            {
                'type': 'localization_error',
                'description': 'Correct label but poor localization (IoU < 0.5)',
                'count': len(low_iou_correct),
                'examples': [{'predicted': a.predicted_label, 'iou': a.iou} 
                           for a in low_iou_correct[:5]]
            },
            {
                'type': 'classification_error',
                'description': 'Good localization but wrong label (IoU > 0.5)',
                'count': len(high_iou_wrong),
                'examples': [{'predicted': a.predicted_label, 'ground_truth': a.ground_truth_label, 'iou': a.iou} 
                           for a in high_iou_wrong[:5]]
            },
            {
                'type': 'common_misclassifications',
                'description': 'Most frequent label confusions',
                'count': len(common_mistakes),
                'examples': common_mistakes[:10]
            }
        ]
    
    def _analyze_success_patterns(self):
        """Identify patterns in successful predictions."""
        self.logger.info("Analyzing success patterns")
        
        # High IoU and correct label
        high_quality = [a for a in self.bbox_analyses 
                       if a.iou > 0.8 and a.predicted_label == a.ground_truth_label]
        
        # Best performing classes
        best_classes = sorted(self.class_metrics.items(), 
                             key=lambda x: x[1].f1_score, reverse=True)[:5]
        
        # Size categories that work well
        size_categories = {'small': [], 'medium': [], 'large': []}
        for analysis in high_quality:
            size = analysis.bbox_size_gt
            if size < 0.1:
                size_categories['small'].append(analysis)
            elif size < 0.4:
                size_categories['medium'].append(analysis)
            else:
                size_categories['large'].append(analysis)
        
        self.success_patterns = [
            {
                'type': 'high_quality_predictions',
                'description': 'Predictions with IoU > 0.8 and correct label',
                'count': len(high_quality),
                'avg_iou': np.mean([a.iou for a in high_quality]) if high_quality else 0,
                'labels': Counter([a.predicted_label for a in high_quality]).most_common(5)
            },
            {
                'type': 'best_performing_classes',
                'description': 'Classes with highest F1 scores',
                'examples': [{'class': name, 'f1_score': metrics.f1_score, 'precision': metrics.precision, 'recall': metrics.recall} 
                           for name, metrics in best_classes]
            },
            {
                'type': 'size_performance',
                'description': 'Performance by object size categories',
                'small_objects': len(size_categories['small']),
                'medium_objects': len(size_categories['medium']),
                'large_objects': len(size_categories['large'])
            }
        ]
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        # Overall statistics
        total_predictions = len(self.bbox_analyses)
        correct_predictions = len([a for a in self.bbox_analyses 
                                 if a.iou > 0.5 and a.predicted_label == a.ground_truth_label])
        
        # Vendor performance
        vendor_performance = {}
        for vendor in self.vendors:
            vendor_analyses = [a for a in self.bbox_analyses 
                             if vendor in a.ground_truth_label]
            if vendor_analyses:
                correct_vendor = len([a for a in vendor_analyses 
                                    if a.iou > 0.5 and a.predicted_label == a.ground_truth_label])
                vendor_performance[vendor] = {
                    'total': len(vendor_analyses),
                    'correct': correct_vendor,
                    'accuracy': correct_vendor / len(vendor_analyses) * 100
                }
        
        # Object type performance
        object_type_performance = {}
        for obj_type in self.object_types:
            obj_analyses = [a for a in self.bbox_analyses 
                           if obj_type in a.ground_truth_label]
            if obj_analyses:
                correct_obj = len([a for a in obj_analyses 
                                 if a.iou > 0.5 and a.predicted_label == a.ground_truth_label])
                object_type_performance[obj_type] = {
                    'total': len(obj_analyses),
                    'correct': correct_obj,
                    'accuracy': correct_obj / len(obj_analyses) * 100
                }
        
        return {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_samples': total_predictions,
                'correct_predictions': correct_predictions,
                'overall_accuracy': correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
            },
            'class_performance': {
                name: {
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'avg_iou': metrics.avg_iou,
                    'avg_center_error': metrics.avg_center_error,
                    'size_error_ratio': metrics.size_error_ratio,
                    'true_positives': metrics.true_positives,
                    'false_positives': metrics.false_positives,
                    'false_negatives': metrics.false_negatives
                } for name, metrics in self.class_metrics.items()
            },
            'vendor_performance': vendor_performance,
            'object_type_performance': object_type_performance,
            'confusion_matrix': {
                gt_label: dict(pred_counter) 
                for gt_label, pred_counter in self.confusion_matrix.items()
            },
            'failure_patterns': self.failure_patterns,
            'success_patterns': self.success_patterns,
            'spatial_analysis': self._analyze_spatial_patterns(),
            'recommendations': self._generate_recommendations()
        }
    
    def _analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial patterns in predictions."""
        # Divide images into grid regions and analyze performance
        grid_size = 3
        grid_performance = {}
        
        for analysis in self.bbox_analyses:
            if analysis.ground_truth_label:
                # Calculate center position as grid coordinate
                center_x = (analysis.bbox_gt[0] + analysis.bbox_gt[2]) / 2 / analysis.image_width
                center_y = (analysis.bbox_gt[1] + analysis.bbox_gt[3]) / 2 / analysis.image_height
                
                grid_x = min(int(center_x * grid_size), grid_size - 1)
                grid_y = min(int(center_y * grid_size), grid_size - 1)
                grid_key = f"{grid_x}_{grid_y}"
                
                if grid_key not in grid_performance:
                    grid_performance[grid_key] = {'total': 0, 'correct': 0}
                
                grid_performance[grid_key]['total'] += 1
                if analysis.iou > 0.5 and analysis.predicted_label == analysis.ground_truth_label:
                    grid_performance[grid_key]['correct'] += 1
        
        # Calculate accuracy per grid cell
        for grid_key in grid_performance:
            total = grid_performance[grid_key]['total']
            correct = grid_performance[grid_key]['correct']
            grid_performance[grid_key]['accuracy'] = correct / total * 100 if total > 0 else 0
        
        return {
            'grid_performance': grid_performance,
            'grid_size': grid_size,
            'description': f"Performance analysis using {grid_size}x{grid_size} grid overlay"
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check for classes with low performance
        low_performance_classes = [name for name, metrics in self.class_metrics.items() 
                                  if metrics.f1_score < 0.3]
        if low_performance_classes:
            recommendations.append({
                'category': 'Low Performance Classes',
                'issue': f"Classes with F1 < 0.3: {', '.join(low_performance_classes[:3])}",
                'suggestion': "Consider data augmentation, class balancing, or additional training samples for these classes"
            })
        
        # Check for localization issues
        localization_issues = [a for a in self.bbox_analyses 
                             if a.iou < 0.5 and a.predicted_label == a.ground_truth_label]
        if len(localization_issues) > len(self.bbox_analyses) * 0.2:
            recommendations.append({
                'category': 'Localization Issues',
                'issue': f"{len(localization_issues)} predictions have correct label but poor localization",
                'suggestion': "Focus on improving bbox regression loss or anchor box design"
            })
        
        # Check for classification confusion
        classification_issues = [a for a in self.bbox_analyses 
                               if a.iou > 0.5 and a.predicted_label != a.ground_truth_label]
        if len(classification_issues) > len(self.bbox_analyses) * 0.15:
            recommendations.append({
                'category': 'Classification Issues',
                'issue': f"{len(classification_issues)} predictions have good localization but wrong label",
                'suggestion': "Improve classification head or add more discriminative features"
            })
        
        # Vendor-specific recommendations
        vendor_scores = {}
        for vendor in self.vendors:
            vendor_analyses = [a for a in self.bbox_analyses if vendor in a.ground_truth_label]
            if vendor_analyses:
                correct = len([a for a in vendor_analyses 
                             if a.iou > 0.5 and a.predicted_label == a.ground_truth_label])
                vendor_scores[vendor] = correct / len(vendor_analyses) * 100
        
        if vendor_scores:
            worst_vendor = min(vendor_scores, key=vendor_scores.get)
            if vendor_scores[worst_vendor] < 50:
                recommendations.append({
                    'category': 'Vendor-Specific Issues',
                    'issue': f"Poor performance on {worst_vendor} equipment ({vendor_scores[worst_vendor]:.1f}% accuracy)",
                    'suggestion': f"Collect more training data for {worst_vendor} equipment or review annotation quality"
                })
        
        return recommendations
    
    def save_analysis(self, output_dir: str, analysis_report: Dict[str, Any]):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main analysis report
        analysis_file = os.path.join(output_dir, 'detailed_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        self._save_human_readable_report(output_dir, analysis_report)
        
        # Save confusion matrix as CSV
        self._save_confusion_matrix_csv(output_dir)
        
        # Generate visualization report
        try:
            from eval.visualization_utils import generate_visualization_report
            viz_file = os.path.join(output_dir, 'visual_analysis.txt')
            generate_visualization_report(analysis_file, viz_file)
        except Exception as e:
            self.logger.warning(f"Could not generate visualization report: {e}")
        
        self.logger.info(f"Analysis results saved to {output_dir}")
    
    def _save_human_readable_report(self, output_dir: str, report: Dict[str, Any]):
        """Save human-readable analysis report."""
        report_path = os.path.join(output_dir, 'analysis_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BBU Equipment Detection Model - Detailed Performance Analysis\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall metrics
            meta = report['metadata']
            f.write(f"Overall Performance:\n")
            f.write(f"  Total Predictions: {meta['total_samples']}\n")
            f.write(f"  Correct Predictions: {meta['correct_predictions']}\n")
            f.write(f"  Overall Accuracy: {meta['overall_accuracy']:.2f}%\n\n")
            
            # Class performance
            f.write("Per-Class Performance:\n")
            f.write("-" * 40 + "\n")
            for class_name, metrics in report['class_performance'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1 Score: {metrics['f1_score']:.3f}\n")
                f.write(f"  Avg IoU: {metrics['avg_iou']:.3f}\n")
                f.write(f"  TP/FP/FN: {metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']}\n\n")
            
            # Vendor performance
            f.write("Vendor Performance:\n")
            f.write("-" * 40 + "\n")
            for vendor, metrics in report['vendor_performance'].items():
                f.write(f"{vendor}: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']})\n")
            f.write("\n")
            
            # Object type performance
            f.write("Object Type Performance:\n")
            f.write("-" * 40 + "\n")
            for obj_type, metrics in report['object_type_performance'].items():
                f.write(f"{obj_type}: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']})\n")
            f.write("\n")
            
            # Failure patterns
            f.write("Failure Patterns:\n")
            f.write("-" * 40 + "\n")
            for pattern in report['failure_patterns']:
                f.write(f"{pattern['type']}: {pattern['count']} cases\n")
                f.write(f"  {pattern['description']}\n")
                if 'examples' in pattern and pattern['examples']:
                    f.write(f"  Examples: {pattern['examples'][:3]}\n")
                f.write("\n")
            
            # Success patterns
            f.write("Success Patterns:\n")
            f.write("-" * 40 + "\n")
            for pattern in report['success_patterns']:
                f.write(f"{pattern['type']}: {pattern.get('count', 'N/A')} cases\n")
                f.write(f"  {pattern['description']}\n")
                if 'examples' in pattern and pattern['examples']:
                    f.write(f"  Examples: {pattern['examples'][:3]}\n")
                f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 40 + "\n")
            for rec in report['recommendations']:
                f.write(f"{rec['category']}:\n")
                f.write(f"  Issue: {rec['issue']}\n")
                f.write(f"  Suggestion: {rec['suggestion']}\n\n")
    
    def _save_confusion_matrix_csv(self, output_dir: str):
        """Save confusion matrix as CSV file."""
        csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
        
        # Get all unique labels
        all_labels = set()
        for gt_label, pred_counter in self.confusion_matrix.items():
            all_labels.add(gt_label)
            all_labels.update(pred_counter.keys())
        
        all_labels = sorted(all_labels)
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Ground Truth / Predicted," + ",".join(all_labels) + "\n")
            
            # Write rows
            for gt_label in all_labels:
                row = [gt_label]
                for pred_label in all_labels:
                    count = self.confusion_matrix[gt_label].get(pred_label, 0)
                    row.append(str(count))
                f.write(",".join(row) + "\n")


def main():
    """Main function to run detailed analysis."""
    parser = argparse.ArgumentParser(description='Detailed performance analysis for BBU detection model')
    parser.add_argument('--predictions_file', required=True, help='Path to inference predictions JSON file')
    parser.add_argument('--output_dir', required=True, help='Output directory for analysis results')
    parser.add_argument('--label_vocab', default='data/label_vocabulary.json', help='Path to label vocabulary JSON')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize analyzer
    analyzer = DetailedAnalyzer(args.label_vocab)
    
    # Run analysis
    analysis_report = analyzer.analyze_predictions(args.predictions_file)
    
    # Save results
    analyzer.save_analysis(args.output_dir, analysis_report)
    
    print(f"✅ Detailed analysis completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📋 Summary: {os.path.join(args.output_dir, 'analysis_summary.txt')}")


if __name__ == "__main__":
    main()