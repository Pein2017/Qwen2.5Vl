#!/usr/bin/env python3
"""
Simple experiment comparison script with configuration at top.
"""

###############################################################################
# COMPARISON CONFIGURATION - EDIT THESE PARAMETERS
###############################################################################

# Directory containing experiments
EXPERIMENTS_DIR = "experiments"

# Metrics to compare (will show top 3 for each)
METRICS_TO_COMPARE = ["mean_mAP", "mean_mAR", "mean_mF1"]

# Whether to generate plots (requires matplotlib)
GENERATE_PLOTS = False

# Output files
COMPARISON_CSV = "experiment_comparison.csv"
COMPARISON_REPORT = "experiment_report.txt"

###############################################################################
# EXECUTION - DO NOT EDIT BELOW THIS LINE
###############################################################################

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def discover_experiments(experiments_dir: str) -> Dict[str, Dict[str, Any]]:
    """Discover all experiments with evaluation summaries."""
    experiments = {}
    exp_path = Path(experiments_dir)
    
    if not exp_path.exists():
        return experiments
    
    for exp_dir in exp_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Look for evaluation summary
        summary_file = exp_dir / "evaluation" / "evaluation_summary.json"
        config_file = exp_dir / "config.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                config = {}
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                
                experiments[exp_dir.name] = {
                    "summary": summary,
                    "config": config,
                    "path": exp_dir
                }
            except Exception as e:
                print(f"Warning: Failed to load experiment {exp_dir.name}: {e}")
    
    return experiments


def print_comparison_table(experiments: Dict[str, Dict[str, Any]]):
    """Print a simple comparison table."""
    if not experiments:
        print("No experiments found.")
        return
    
    print("\n📊 EXPERIMENT COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Experiment':<30} {'Teachers':<8} {'mAP':<8} {'mAR':<8} {'mF1':<8} {'Status'}")
    print("-" * 80)
    
    # Sort by mAP
    sorted_experiments = []
    for exp_name, exp_data in experiments.items():
        summary = exp_data["summary"]
        config = exp_data["config"]
        
        aggregated = summary.get("aggregated_metrics", {})
        
        if "error" in aggregated:
            status = "FAILED"
            map_val = mar_val = f1_val = 0.0
        else:
            status = "OK"
            map_val = aggregated.get("mean_mAP", 0)
            mar_val = aggregated.get("mean_mAR", 0)
            f1_val = aggregated.get("mean_mF1", 0)
        
        teachers = config.get("teacher", {}).get("num_teachers", 0)
        
        sorted_experiments.append({
            "name": exp_name,
            "teachers": teachers,
            "mAP": map_val,
            "mAR": mar_val,
            "mF1": f1_val,
            "status": status
        })
    
    # Sort by mAP descending
    sorted_experiments.sort(key=lambda x: x["mAP"], reverse=True)
    
    # Print rows
    for exp in sorted_experiments:
        name = exp["name"][:29]  # Truncate long names
        print(f"{name:<30} {exp['teachers']:<8} {exp['mAP']:<8.4f} {exp['mAR']:<8.4f} {exp['mF1']:<8.4f} {exp['status']}")
    
    print("-" * 80)


def find_best_experiments(experiments: Dict[str, Dict[str, Any]]):
    """Find and print best performing experiments."""
    if not experiments:
        return
    
    print("\n🏆 BEST PERFORMING EXPERIMENTS")
    print("=" * 50)
    
    valid_experiments = []
    for exp_name, exp_data in experiments.items():
        summary = exp_data["summary"]
        aggregated = summary.get("aggregated_metrics", {})
        
        if "error" not in aggregated:
            valid_experiments.append({
                "name": exp_name,
                "metrics": aggregated
            })
    
    if not valid_experiments:
        print("No valid experiments found.")
        return
    
    for metric in METRICS_TO_COMPARE:
        # Sort by metric
        sorted_by_metric = sorted(
            valid_experiments, 
            key=lambda x: x["metrics"].get(metric, 0), 
            reverse=True
        )
        
        print(f"\n{metric}:")
        for i, exp in enumerate(sorted_by_metric[:3]):  # Top 3
            value = exp["metrics"].get(metric, 0)
            print(f"  {i+1}. {exp['name']}: {value:.4f}")


def generate_csv_report(experiments: Dict[str, Dict[str, Any]], output_file: str):
    """Generate CSV report with all experiment data."""
    if not experiments:
        print("No experiments to export.")
        return
    
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'experiment_id', 'experiment_name', 'timestamp',
            'model_name', 'model_checkpoint', 'num_teachers',
            'max_tokens', 'batch_size',
            'mean_mAP', 'mean_mAR', 'mean_mF1',
            'datasets_evaluated', 'status'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for exp_name, exp_data in experiments.items():
            summary = exp_data["summary"]
            config = exp_data["config"]
            
            aggregated = summary.get("aggregated_metrics", {})
            
            if "error" in aggregated:
                status = "FAILED"
                map_val = mar_val = f1_val = 0.0
                datasets_eval = 0
            else:
                status = "SUCCESS"
                map_val = aggregated.get("mean_mAP", 0)
                mar_val = aggregated.get("mean_mAR", 0)
                f1_val = aggregated.get("mean_mF1", 0)
                datasets_eval = aggregated.get("datasets_evaluated", 0)
            
            row = {
                'experiment_id': exp_name,
                'experiment_name': config.get("experiment_name", ""),
                'timestamp': config.get("timestamp", ""),
                'model_name': config.get("model", {}).get("name", ""),
                'model_checkpoint': config.get("model", {}).get("path", "").split("/")[-1] if config.get("model", {}).get("path") else "",
                'num_teachers': config.get("teacher", {}).get("num_teachers", 0),
                'max_tokens': config.get("generation", {}).get("max_new_tokens", 0),
                'batch_size': config.get("generation", {}).get("batch_size", 0),
                'mean_mAP': map_val,
                'mean_mAR': mar_val,
                'mean_mF1': f1_val,
                'datasets_evaluated': datasets_eval,
                'status': status
            }
            
            writer.writerow(row)
    
    print(f"📄 CSV report saved to: {output_file}")


def generate_text_report(experiments: Dict[str, Dict[str, Any]], output_file: str):
    """Generate detailed text report."""
    if not experiments:
        return
    
    lines = []
    lines.append("EXPERIMENT ANALYSIS REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total experiments: {len(experiments)}")
    lines.append("")
    
    # Summary statistics
    valid_count = 0
    failed_count = 0
    total_datasets = 0
    
    for exp_data in experiments.values():
        summary = exp_data["summary"]
        aggregated = summary.get("aggregated_metrics", {})
        
        if "error" in aggregated:
            failed_count += 1
        else:
            valid_count += 1
            total_datasets += aggregated.get("datasets_evaluated", 0)
    
    lines.append(f"Valid experiments: {valid_count}")
    lines.append(f"Failed experiments: {failed_count}")
    lines.append(f"Total datasets evaluated: {total_datasets}")
    lines.append("")
    
    # Best experiments
    lines.append("BEST PERFORMING EXPERIMENTS")
    lines.append("-" * 30)
    
    valid_experiments = []
    for exp_name, exp_data in experiments.items():
        summary = exp_data["summary"]
        aggregated = summary.get("aggregated_metrics", {})
        
        if "error" not in aggregated:
            valid_experiments.append({
                "name": exp_name,
                "metrics": aggregated
            })
    
    for metric in METRICS_TO_COMPARE:
        sorted_by_metric = sorted(
            valid_experiments, 
            key=lambda x: x["metrics"].get(metric, 0), 
            reverse=True
        )
        
        lines.append(f"\nBest {metric}:")
        for i, exp in enumerate(sorted_by_metric[:3]):
            value = exp["metrics"].get(metric, 0)
            lines.append(f"  {i+1}. {exp['name']}: {value:.4f}")
    
    lines.append("")
    
    # Detailed experiment info
    lines.append("DETAILED EXPERIMENT RESULTS")
    lines.append("-" * 35)
    
    for exp_name, exp_data in experiments.items():
        summary = exp_data["summary"]
        config = exp_data["config"]
        aggregated = summary.get("aggregated_metrics", {})
        
        lines.append(f"\n{exp_name}:")
        lines.append(f"  Model: {config.get('model', {}).get('name', 'unknown')}")
        lines.append(f"  Teachers: {config.get('teacher', {}).get('num_teachers', 0)}")
        lines.append(f"  Tokens: {config.get('generation', {}).get('max_new_tokens', 'unknown')}")
        
        if "error" in aggregated:
            lines.append(f"  Status: FAILED - {aggregated['error']}")
        else:
            lines.append(f"  mAP: {aggregated.get('mean_mAP', 0):.4f}")
            lines.append(f"  mAR: {aggregated.get('mean_mAR', 0):.4f}")
            lines.append(f"  mF1: {aggregated.get('mean_mF1', 0):.4f}")
            lines.append(f"  Datasets: {aggregated.get('datasets_evaluated', 0)}")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"📄 Text report saved to: {output_file}")


def main():
    """Main execution function."""
    print("🔍 Discovering experiments...")
    experiments = discover_experiments(EXPERIMENTS_DIR)
    
    if not experiments:
        print(f"❌ No experiments found in {EXPERIMENTS_DIR}")
        print("Make sure you have run inference and evaluation first.")
        return
    
    print(f"✅ Found {len(experiments)} experiments")
    
    # Print comparison table
    print_comparison_table(experiments)
    
    # Find best experiments
    find_best_experiments(experiments)
    
    # Generate reports
    print(f"\n📊 Generating reports...")
    generate_csv_report(experiments, COMPARISON_CSV)
    generate_text_report(experiments, COMPARISON_REPORT)
    
    # Optional: Generate plots
    if GENERATE_PLOTS:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Simple bar plot of mAP scores
            valid_experiments = []
            for exp_name, exp_data in experiments.items():
                summary = exp_data["summary"]
                aggregated = summary.get("aggregated_metrics", {})
                
                if "error" not in aggregated:
                    valid_experiments.append({
                        "name": exp_name,
                        "mAP": aggregated.get("mean_mAP", 0)
                    })
            
            if valid_experiments:
                valid_experiments.sort(key=lambda x: x["mAP"], reverse=True)
                
                names = [exp["name"] for exp in valid_experiments]
                maps = [exp["mAP"] for exp in valid_experiments]
                
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(names)), maps)
                plt.xlabel('Experiments')
                plt.ylabel('Mean Average Precision (mAP)')
                plt.title('Experiment Comparison - mAP Scores')
                plt.xticks(range(len(names)), names, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, v in enumerate(maps):
                    plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
                print("📊 Plot saved to: experiment_comparison.png")
                
        except ImportError:
            print("⚠️  Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"⚠️  Failed to generate plot: {e}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()