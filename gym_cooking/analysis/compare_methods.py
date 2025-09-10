#!/usr/bin/env python3
"""
Analysis and comparison tools for experiment results.

This script provides utilities to compare multiple methods and generate reports.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.experiment import MultiMethodTracker
from utils.metrics import ComparisonMetrics

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_experiments_from_directory(experiments_dir: str) -> List[str]:
    """Find all experiment directories in a given path."""
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        return []
    
    experiments = []
    for item in exp_path.iterdir():
        if item.is_dir():
            # Check if it's a valid experiment (has metadata file)
            metadata_file = item / "experiment_metadata.json"
            if metadata_file.exists():
                experiments.append(str(item))
    
    return experiments


def compare_experiments(experiment_dirs: List[str], output_dir: str = "./comparisons"):
    """Compare multiple experiments and generate report."""
    tracker = MultiMethodTracker(output_dir)
    
    # Load all experiments
    loaded_count = 0
    for exp_dir in experiment_dirs:
        summary = tracker.load_experiment(exp_dir)
        if summary:
            loaded_count += 1
            print(f"✓ Loaded experiment: {summary.method_name} ({exp_dir})")
        else:
            print(f"✗ Failed to load experiment: {exp_dir}")
    
    if loaded_count == 0:
        print("No valid experiments found!")
        return
    
    # Generate comparison
    comparison = tracker.generate_comparison()
    report_file = tracker.save_comparison_report()
    
    print(f"\n🎉 Comparison complete!")
    print(f"📄 Report saved to: {report_file}")
    print(f"🏆 Best method: {comparison.best_method}")
    
    # Print summary table
    print("\n📊 Summary Table:")
    print("-" * 80)
    table = comparison.get_comparison_table()
    
    # Header
    print(f"{'Method':<20} {'Mean Reward':<12} {'Std':<8} {'Best':<8} {'Episodes':<9} {'Runtime':<8}")
    print("-" * 80)
    
    # Sort by mean reward (descending)
    sorted_methods = sorted(table.keys(), key=lambda m: table[m]['mean_reward'], reverse=True)
    for method in sorted_methods:
        stats = table[method]
        print(f"{method:<20} {stats['mean_reward']:<12.3f} {stats['std_reward']:<8.2f} "
              f"{stats['best_reward']:<8.1f} {stats['total_episodes']:<9} {stats['runtime']:<8.1f}s")
    
    return comparison


def plot_comparison(comparison: ComparisonMetrics, output_dir: str = "./comparisons"):
    """Generate comparison plots."""
    if not PLOTTING_AVAILABLE:
        print("⚠️  Plotting not available (matplotlib/pandas not installed)")
        return
    
    output_path = Path(output_dir)
    table = comparison.get_comparison_table()
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(table).T
    
    # Plot 1: Mean reward comparison
    plt.figure(figsize=(10, 6))
    methods = df.index
    rewards = df['mean_reward']
    errors = df['std_reward']
    
    bars = plt.bar(methods, rewards, yerr=errors, capsize=5, alpha=0.7)
    plt.title('Mean Reward Comparison Across Methods')
    plt.ylabel('Mean Reward')
    plt.xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    
    # Highlight best method
    best_idx = list(methods).index(comparison.best_method)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(output_path / "reward_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved reward comparison plot: {output_path / 'reward_comparison.png'}")
    
    # Plot 2: Performance vs Runtime
    plt.figure(figsize=(10, 6))
    plt.scatter(df['runtime'], df['mean_reward'], s=100, alpha=0.7)
    
    for i, method in enumerate(methods):
        plt.annotate(method, (df.loc[method, 'runtime'], df.loc[method, 'mean_reward']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Mean Reward')
    plt.title('Performance vs Runtime')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "performance_vs_runtime.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved performance vs runtime plot: {output_path / 'performance_vs_runtime.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments-dir", type=str, default="./experiments",
                       help="Directory containing experiment subdirectories")
    parser.add_argument("--experiment-dirs", nargs="+", type=str,
                       help="Specific experiment directories to compare")
    parser.add_argument("--output-dir", type=str, default="./comparisons",
                       help="Output directory for comparison results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    # Determine which experiments to compare
    if args.experiment_dirs:
        experiment_dirs = args.experiment_dirs
    else:
        experiment_dirs = load_experiments_from_directory(args.experiments_dir)
        if not experiment_dirs:
            print(f"No experiments found in {args.experiments_dir}")
            return
    
    print(f"🔍 Found {len(experiment_dirs)} experiment(s) to compare")
    
    # Run comparison
    comparison = compare_experiments(experiment_dirs, args.output_dir)
    if not comparison:
        return
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(comparison, args.output_dir)


if __name__ == "__main__":
    main()
