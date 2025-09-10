#!/usr/bin/env python3
"""
Visualization tools for individual experiment analysis.

This script provides detailed visualizations for single experiments.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.experiment import ExperimentTracker
from utils.metrics import ExperimentSummary

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load episode data from CSV file."""
    if not PLOTTING_AVAILABLE:
        return None
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"⚠️  CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"⚠️  Error loading CSV: {e}")
        return None


def plot_learning_curve(df: pd.DataFrame, output_path: Path, window_size: int = 10):
    """Plot learning curve with moving average."""
    if df is None or df.empty:
        return
    
    episodes = df['episode']
    rewards = df['total_reward']
    
    # Calculate moving average
    rolling_mean = rewards.rolling(window=window_size, min_periods=1).mean()
    rolling_std = rewards.rolling(window=window_size, min_periods=1).std()
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards (lighter)
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    plt.plot(episodes, rolling_mean, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    
    # Add confidence band
    if rolling_std is not None:
        plt.fill_between(episodes, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std,
                        alpha=0.2, color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "learning_curve.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved learning curve: {output_path / 'learning_curve.png'}")


def plot_performance_metrics(df: pd.DataFrame, output_path: Path):
    """Plot various performance metrics over time."""
    if df is None or df.empty:
        return
    
    # Check what metrics are available
    metric_columns = ['total_reward', 'completion_time', 'efficiency', 'waste_penalty']
    available_metrics = [col for col in metric_columns if col in df.columns]
    
    if not available_metrics:
        print("⚠️  No performance metrics found in data")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        episodes = df['episode']
        values = df[metric]
        
        # Plot raw values
        ax.plot(episodes, values, alpha=0.6, marker='o', markersize=2)
        
        # Add moving average
        rolling_mean = values.rolling(window=10, min_periods=1).mean()
        ax.plot(episodes, rolling_mean, color='red', linewidth=2)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Episode')
    plt.suptitle('Performance Metrics Over Time')
    plt.tight_layout()
    plt.savefig(output_path / "performance_metrics.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved performance metrics: {output_path / 'performance_metrics.png'}")


def plot_episode_distribution(df: pd.DataFrame, output_path: Path):
    """Plot distribution of episode rewards."""
    if df is None or df.empty or 'total_reward' not in df.columns:
        return
    
    rewards = df['total_reward']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards.mean():.2f}')
    ax1.axvline(rewards.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {rewards.median():.2f}')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(rewards)
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Reward Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "reward_distribution.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved reward distribution: {output_path / 'reward_distribution.png'}")


def plot_convergence_analysis(df: pd.DataFrame, output_path: Path, window_size: int = 50):
    """Analyze convergence of the learning process."""
    if df is None or df.empty or 'total_reward' not in df.columns:
        return
    
    rewards = df['total_reward']
    episodes = df['episode']
    
    # Calculate rolling statistics
    rolling_mean = rewards.rolling(window=window_size, min_periods=1).mean()
    rolling_std = rewards.rolling(window=window_size, min_periods=1).std()
    rolling_var = rewards.rolling(window=window_size, min_periods=1).var()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Mean and variance
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(episodes, rolling_mean, color='blue', label='Rolling Mean', linewidth=2)
    line2 = ax1_twin.plot(episodes, rolling_var, color='red', alpha=0.7, label='Rolling Variance')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rolling Mean Reward', color='blue')
    ax1_twin.set_ylabel('Rolling Variance', color='red')
    ax1.set_title(f'Convergence Analysis (window={window_size})')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Coefficient of variation (stability measure)
    cv = rolling_std / np.abs(rolling_mean)
    cv = cv.replace([np.inf, -np.inf], np.nan)  # Replace infinities
    
    ax2.plot(episodes, cv, color='purple', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Learning Stability (Lower is More Stable)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "convergence_analysis.png", dpi=300, bbox_inches='tight')
    print(f"📈 Saved convergence analysis: {output_path / 'convergence_analysis.png'}")


def generate_experiment_report(experiment_dir: str, output_dir: str = None):
    """Generate comprehensive report for a single experiment."""
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = exp_path / "analysis"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🔍 Analyzing experiment: {experiment_dir}")
    
    # Load experiment metadata
    metadata_file = exp_path / "experiment_metadata.json"
    if not metadata_file.exists():
        print("⚠️  No metadata file found")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Method: {metadata.get('method_name', 'Unknown')}")
    print(f"⏰ Duration: {metadata.get('total_runtime', 0):.2f}s")
    print(f"📈 Episodes: {metadata.get('total_episodes', 0)}")
    
    # Load CSV data if available
    csv_file = exp_path / "episodes.csv"
    df = load_csv_data(str(csv_file))
    
    if df is not None and PLOTTING_AVAILABLE:
        print("📈 Generating visualizations...")
        plot_learning_curve(df, output_path)
        plot_performance_metrics(df, output_path)
        plot_episode_distribution(df, output_path)
        plot_convergence_analysis(df, output_path)
        
        print(f"✅ Analysis complete! Results saved to: {output_path}")
    else:
        if not PLOTTING_AVAILABLE:
            print("⚠️  Plotting libraries not available")
        else:
            print("⚠️  No episode data found for visualization")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("experiment_dir", type=str,
                       help="Directory containing the experiment to analyze")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for visualizations (default: experiment_dir/analysis)")
    
    args = parser.parse_args()
    
    generate_experiment_report(args.experiment_dir, args.output_dir)


if __name__ == "__main__":
    main()
