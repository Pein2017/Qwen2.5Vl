#!/usr/bin/env python3
"""
Compare experiments with the new directory structure.

Usage:
    python eval/compare_experiments_new.py
    python eval/compare_experiments_new.py --experiments 1_teacher no_teacher
    python eval/compare_experiments_new.py --dataset val
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_experiment_results(experiment_path: Path, dataset: str) -> Optional[Dict]:
    """Load evaluation results for a specific experiment and dataset."""
    metrics_file = experiment_path / dataset / "evaluation" / "metrics.json"
    
    if not metrics_file.exists():
        return None
        
    try:
        with open(metrics_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {metrics_file}: {e}")
        return None


def get_experiment_config(experiment_path: Path) -> Optional[Dict]:
    """Load experiment configuration."""
    config_file = experiment_path / "config.json"
    
    if not config_file.exists():
        return None
        
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {config_file}: {e}")
        return None


def find_experiments(experiments_dir: Path, filter_names: Optional[List[str]] = None) -> List[str]:
    """Find all available experiments."""
    if not experiments_dir.exists():
        return []
    
    experiments = []
    for item in experiments_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if filter_names is None or item.name in filter_names:
                experiments.append(item.name)
    
    return sorted(experiments)


def print_comparison_table(results: Dict[str, Dict[str, Dict]], dataset: str):
    """Print a comparison table for the specified dataset."""
    print(f"\n🏆 COMPARISON RESULTS - {dataset.upper()} DATASET")
    print("=" * 80)
    
    # Header
    print(f"{'Experiment':<20} {'mAP':<8} {'mAR':<8} {'mF1':<8} {'Samples':<10} {'Teachers':<10}")
    print("-" * 80)
    
    # Sort by mF1 score (descending)
    sorted_experiments = sorted(
        results.items(),
        key=lambda x: x[1][dataset]['overall_metrics']['mF1'] if dataset in x[1] and x[1][dataset] else 0,
        reverse=True
    )
    
    for exp_name, exp_data in sorted_experiments:
        if dataset in exp_data and exp_data[dataset]:
            metrics = exp_data[dataset]['overall_metrics']
            info = exp_data[dataset]['evaluation_info']
            config = exp_data.get('config', {})
            
            teachers = config.get('teacher', {}).get('num_teachers', 'N/A')
            samples = info.get('total_samples', 'N/A')
            
            print(f"{exp_name:<20} "
                  f"{metrics['mAP']:<8.4f} "
                  f"{metrics['mAR']:<8.4f} "
                  f"{metrics['mF1']:<8.4f} "
                  f"{samples:<10} "
                  f"{teachers:<10}")
        else:
            print(f"{exp_name:<20} {'No data':<40}")


def print_detailed_breakdown(results: Dict[str, Dict[str, Dict]], dataset: str):
    """Print detailed category breakdown."""
    print(f"\n📊 DETAILED CATEGORY BREAKDOWN - {dataset.upper()} DATASET")
    print("=" * 120)
    
    # Collect all categories
    all_categories = set()
    for exp_data in results.values():
        if dataset in exp_data and exp_data[dataset] and 'category_metrics' in exp_data[dataset]:
            all_categories.update(exp_data[dataset]['category_metrics'].keys())
    
    if not all_categories:
        print("No category data available")
        return
    
    # Sort categories
    sorted_categories = sorted(all_categories)
    
    for category in sorted_categories:
        print(f"\n📋 Category: {category}")
        print(f"{'Experiment':<20} {'mAP':<8} {'mAR':<8} {'mF1':<8}")
        print("-" * 50)
        
        for exp_name, exp_data in results.items():
            if (dataset in exp_data and exp_data[dataset] and 
                'category_metrics' in exp_data[dataset] and 
                category in exp_data[dataset]['category_metrics']):
                
                cat_metrics = exp_data[dataset]['category_metrics'][category]
                print(f"{exp_name:<20} "
                      f"{cat_metrics['mAP']:<8.4f} "
                      f"{cat_metrics['mAR']:<8.4f} "
                      f"{cat_metrics['mF1']:<8.4f}")
            else:
                print(f"{exp_name:<20} {'No data':<24}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to compare')
    parser.add_argument('--dataset', choices=['train', 'val', 'both'], default='both', 
                       help='Dataset to compare (default: both)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed category breakdown')
    parser.add_argument('--experiments-dir', default='experiments', help='Experiments directory')
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    # Find experiments
    experiment_names = find_experiments(experiments_dir, args.experiments)
    
    if not experiment_names:
        print("❌ No experiments found")
        if args.experiments:
            print(f"Requested: {args.experiments}")
        print(f"Searched in: {experiments_dir.absolute()}")
        return
    
    print(f"🔍 Found {len(experiment_names)} experiments: {', '.join(experiment_names)}")
    
    # Load all results
    results = {}
    datasets_to_compare = ['train', 'val'] if args.dataset == 'both' else [args.dataset]
    
    for exp_name in experiment_names:
        exp_path = experiments_dir / exp_name
        results[exp_name] = {}
        
        # Load config
        config = get_experiment_config(exp_path)
        if config:
            results[exp_name]['config'] = config
        
        # Load results for each dataset
        for dataset in datasets_to_compare:
            exp_results = load_experiment_results(exp_path, dataset)
            if exp_results:
                results[exp_name][dataset] = exp_results
                print(f"✅ Loaded {exp_name}/{dataset}")
            else:
                print(f"⚠️  No results for {exp_name}/{dataset}")
    
    # Print comparisons
    for dataset in datasets_to_compare:
        print_comparison_table(results, dataset)
        
        if args.detailed:
            print_detailed_breakdown(results, dataset)
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 80)
    
    for dataset in datasets_to_compare:
        print(f"\n{dataset.upper()} Dataset:")
        
        # Find best performing experiment
        best_exp = None
        best_f1 = 0
        
        for exp_name, exp_data in results.items():
            if dataset in exp_data and exp_data[dataset]:
                f1 = exp_data[dataset]['overall_metrics']['mF1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_exp = exp_name
        
        if best_exp:
            config = results[best_exp].get('config', {})
            teachers = config.get('teacher', {}).get('num_teachers', 'N/A')
            print(f"   🏆 Best: {best_exp} (mF1: {best_f1:.4f}, Teachers: {teachers})")
        else:
            print(f"   ❌ No valid results")
    
    print(f"\n📁 Detailed results available in: {experiments_dir.absolute()}")


if __name__ == "__main__":
    main()