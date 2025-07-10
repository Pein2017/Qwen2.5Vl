#!/usr/bin/env python3
"""
Visualization utilities for BBU detection model analysis.

This module provides simple text-based visualizations for performance analysis
that work without external visualization libraries.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from collections import Counter


class TextVisualizer:
    """Simple text-based visualizations for analysis results."""
    
    @staticmethod
    def create_performance_bar_chart(data: Dict[str, float], title: str, width: int = 50) -> str:
        """Create a simple text-based bar chart."""
        if not data:
            return f"{title}\n(No data available)\n"
        
        # Normalize values to chart width
        max_val = max(data.values()) if data.values() else 1.0
        
        chart = [f"{title}"]
        chart.append("=" * len(title))
        
        for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((value / max_val) * width) if max_val > 0 else 0
            bar = "█" * bar_length + "░" * (width - bar_length)
            chart.append(f"{label:<30} |{bar}| {value:.3f}")
        
        return "\n".join(chart) + "\n"
    
    @staticmethod
    def create_confusion_matrix_text(confusion_data: Dict[str, Dict[str, int]], 
                                   top_n: int = 10) -> str:
        """Create text representation of confusion matrix focusing on top confusions."""
        chart = ["Confusion Matrix - Top Misclassifications"]
        chart.append("=" * 50)
        
        # Find top confusions (excluding correct predictions)
        confusions = []
        for gt_label, predictions in confusion_data.items():
            if gt_label:  # Skip empty ground truth
                for pred_label, count in predictions.items():
                    if pred_label != gt_label and count > 0:  # Misclassification
                        confusions.append((gt_label, pred_label, count))
        
        # Sort by frequency
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        if not confusions:
            chart.append("(No misclassifications found)")
        else:
            chart.append(f"{'Ground Truth':<30} -> {'Predicted':<30} Count")
            chart.append("-" * 70)
            
            for gt, pred, count in confusions[:top_n]:
                # Truncate long labels
                gt_short = gt[:28] + ".." if len(gt) > 30 else gt
                pred_short = pred[:28] + ".." if len(pred) > 30 else pred
                chart.append(f"{gt_short:<30} -> {pred_short:<30} {count:>5}")
        
        return "\n".join(chart) + "\n"
    
    @staticmethod
    def create_spatial_heatmap_text(grid_data: Dict[str, Dict], grid_size: int = 3) -> str:
        """Create text representation of spatial performance heatmap."""
        chart = [f"Spatial Performance Heatmap ({grid_size}x{grid_size} grid)"]
        chart.append("=" * 40)
        chart.append("Layout: accuracy% (correct/total)")
        chart.append("")
        
        # Create grid visualization
        for y in range(grid_size):
            row_parts = []
            for x in range(grid_size):
                grid_key = f"{x}_{y}"
                if grid_key in grid_data:
                    data = grid_data[grid_key]
                    accuracy = data.get('accuracy', 0)
                    correct = data.get('correct', 0)
                    total = data.get('total', 0)
                    
                    # Use symbols to indicate performance level
                    if accuracy >= 80:
                        symbol = "🟢"
                    elif accuracy >= 60:
                        symbol = "🟡"
                    elif accuracy >= 40:
                        symbol = "🟠"
                    else:
                        symbol = "🔴"
                    
                    cell = f"{symbol}{accuracy:4.0f}%({correct}/{total})"
                else:
                    cell = "     --     "
                
                row_parts.append(cell)
            
            chart.append("  ".join(row_parts))
        
        chart.append("")
        chart.append("Legend: 🟢≥80% 🟡≥60% 🟠≥40% 🔴<40%")
        
        return "\n".join(chart) + "\n"
    
    @staticmethod
    def create_vendor_comparison_chart(vendor_data: Dict[str, Dict]) -> str:
        """Create vendor performance comparison chart."""
        chart = ["Vendor Performance Comparison"]
        chart.append("=" * 35)
        
        if not vendor_data:
            chart.append("(No vendor data available)")
            return "\n".join(chart) + "\n"
        
        # Sort vendors by accuracy
        vendors = sorted(vendor_data.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
        
        chart.append(f"{'Vendor':<15} {'Accuracy':<10} {'Correct/Total'}")
        chart.append("-" * 40)
        
        for vendor, data in vendors:
            accuracy = data.get('accuracy', 0)
            correct = data.get('correct', 0)
            total = data.get('total', 0)
            
            # Visual bar
            bar_length = int(accuracy / 10)  # Scale to 10 chars max
            bar = "█" * bar_length + "░" * (10 - bar_length)
            
            chart.append(f"{vendor:<15} {accuracy:6.1f}% |{bar}| {correct}/{total}")
        
        return "\n".join(chart) + "\n"
    
    @staticmethod
    def create_size_performance_analysis(size_data: Dict[str, Any]) -> str:
        """Create object size vs performance analysis."""
        chart = ["Object Size Performance Analysis"]
        chart.append("=" * 35)
        
        small = size_data.get('small_objects', 0)
        medium = size_data.get('medium_objects', 0)
        large = size_data.get('large_objects', 0)
        total = small + medium + large
        
        if total == 0:
            chart.append("(No size data available)")
            return "\n".join(chart) + "\n"
        
        chart.append(f"Small objects (<10% image):  {small:4d} ({small/total*100:5.1f}%)")
        chart.append(f"Medium objects (10-40%):     {medium:4d} ({medium/total*100:5.1f}%)")
        chart.append(f"Large objects (>40%):        {large:4d} ({large/total*100:5.1f}%)")
        chart.append("")
        
        # Simple bar chart
        max_count = max(small, medium, large) if total > 0 else 1
        
        for size_name, count in [("Small", small), ("Medium", medium), ("Large", large)]:
            bar_length = int((count / max_count) * 20) if max_count > 0 else 0
            bar = "█" * bar_length + "░" * (20 - bar_length)
            chart.append(f"{size_name:<8} |{bar}| {count}")
        
        return "\n".join(chart) + "\n"


def generate_visualization_report(analysis_file: str, output_file: str):
    """Generate comprehensive visualization report from analysis results."""
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    visualizer = TextVisualizer()
    
    # Prepare report sections
    sections = []
    
    # 1. Overall performance summary
    sections.append("BBU DETECTION MODEL - VISUAL PERFORMANCE ANALYSIS")
    sections.append("=" * 60)
    sections.append("")
    
    meta = analysis_data.get('metadata', {})
    sections.append(f"Analysis Date: {meta.get('analysis_date', 'Unknown')}")
    sections.append(f"Total Samples: {meta.get('total_samples', 0)}")
    sections.append(f"Overall Accuracy: {meta.get('overall_accuracy', 0):.2f}%")
    sections.append("")
    
    # 2. Class performance chart
    class_performance = analysis_data.get('class_performance', {})
    if class_performance:
        f1_scores = {name: data.get('f1_score', 0) for name, data in class_performance.items()}
        sections.append(visualizer.create_performance_bar_chart(f1_scores, "F1 Score by Class"))
        
        precision_scores = {name: data.get('precision', 0) for name, data in class_performance.items()}
        sections.append(visualizer.create_performance_bar_chart(precision_scores, "Precision by Class"))
        
        recall_scores = {name: data.get('recall', 0) for name, data in class_performance.items()}
        sections.append(visualizer.create_performance_bar_chart(recall_scores, "Recall by Class"))
    
    # 3. Vendor comparison
    vendor_data = analysis_data.get('vendor_performance', {})
    if vendor_data:
        sections.append(visualizer.create_vendor_comparison_chart(vendor_data))
    
    # 4. Confusion matrix
    confusion_data = analysis_data.get('confusion_matrix', {})
    if confusion_data:
        sections.append(visualizer.create_confusion_matrix_text(confusion_data))
    
    # 5. Spatial analysis
    spatial_data = analysis_data.get('spatial_analysis', {})
    grid_data = spatial_data.get('grid_performance', {})
    grid_size = spatial_data.get('grid_size', 3)
    if grid_data:
        sections.append(visualizer.create_spatial_heatmap_text(grid_data, grid_size))
    
    # 6. Size analysis
    success_patterns = analysis_data.get('success_patterns', [])
    size_pattern = next((p for p in success_patterns if p.get('type') == 'size_performance'), None)
    if size_pattern:
        sections.append(visualizer.create_size_performance_analysis(size_pattern))
    
    # 7. Key insights
    sections.append("KEY INSIGHTS")
    sections.append("=" * 20)
    
    recommendations = analysis_data.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):
            sections.append(f"{i}. {rec.get('category', 'Issue')}")
            sections.append(f"   Problem: {rec.get('issue', 'N/A')}")
            sections.append(f"   Solution: {rec.get('suggestion', 'N/A')}")
            sections.append("")
    else:
        sections.append("No specific recommendations generated.")
    
    # Write visualization report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(sections))
    
    print(f"✅ Visualization report generated: {output_file}")


def main():
    """Command line interface for visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization report from analysis results')
    parser.add_argument('--analysis_file', required=True, help='Path to detailed_analysis.json file')
    parser.add_argument('--output_file', required=True, help='Path for output visualization report')
    
    args = parser.parse_args()
    
    generate_visualization_report(args.analysis_file, args.output_file)


if __name__ == "__main__":
    main()