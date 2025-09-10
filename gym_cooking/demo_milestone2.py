#!/usr/bin/env python3
"""
Demo script for Milestone 2: Logging and Metrics Framework

This script demonstrates the comprehensive logging and metrics system,
including TensorBoard support and analysis tools.
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config
from solution_methods.runner import create_runner_from_config
from utils.experiment import ExperimentTracker, MultiMethodTracker
from analysis import compare_experiments, generate_experiment_report


def run_demo_experiments():
    """Run demo experiments with different methods and logging."""
    print("🚀 Starting Milestone 2 Demo: Logging and Metrics Framework")
    print("=" * 80)
    
    # Methods to test
    methods = ["dispatching", "random"]
    experiments_dir = "./demo_experiments_m2"
    
    # Ensure clean start
    import shutil
    if os.path.exists(experiments_dir):
        shutil.rmtree(experiments_dir)
    
    experiment_dirs = []
    
    for method in methods:
        print(f"\n📊 Running {method} method...")
        
        # Load config
        config_path = f"configs/training/{method}.toml"
        config = load_config(config_path)
        
        # Override with demo settings (fewer episodes for quick demo)
        config['training']['episodes'] = 20
        config['logging']['enabled'] = True
        config['logging']['tensorboard'] = True
        config['logging']['save_episodes'] = True
        
        # Create runner
        runner = create_runner_from_config(config)
        
        # Create experiment tracker
        exp_name = f"demo_{method}_m2"
        exp_dir = f"{experiments_dir}/{exp_name}"
        tracker = ExperimentTracker(
            experiment_name=exp_name,
            output_dir=exp_dir,
            config=config
        )
        
        # Run experiment
        print(f"  🏃 Running {config['training']['episodes']} episodes...")
        start_time = time.time()
        
        with tracker.time_experiment():
            results = runner.run(
                episodes=config['training']['episodes'],
                experiment_tracker=tracker
            )
        
        # Save summary
        tracker.finalize_experiment()
        experiment_dirs.append(exp_dir)
        
        runtime = time.time() - start_time
        print(f"  ✅ {method} completed in {runtime:.2f}s")
        print(f"     Mean reward: {results.get('mean_reward', 0):.3f}")
        print(f"     Saved to: {exp_dir}")
    
    return experiment_dirs


def demo_analysis_tools(experiment_dirs):
    """Demonstrate analysis and visualization tools."""
    print("\n" + "=" * 80)
    print("📈 Demo: Analysis and Visualization Tools")
    print("=" * 80)
    
    # 1. Compare experiments
    print("\n🔍 Comparing experiments...")
    comparison = compare_experiments(experiment_dirs, "./demo_comparisons_m2")
    
    if comparison:
        print(f"\n🏆 Best method: {comparison.best_method}")
        table = comparison.get_comparison_table()
        
        print("\n📋 Detailed Results:")
        for method, stats in table.items():
            print(f"  {method}:")
            print(f"    Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
            print(f"    Best Reward: {stats['best_reward']:.1f}")
            print(f"    Episodes: {stats['total_episodes']}")
            print(f"    Runtime: {stats['runtime']:.1f}s")
    
    # 2. Analyze individual experiments
    print(f"\n📊 Generating individual analysis reports...")
    for exp_dir in experiment_dirs:
        print(f"  Analyzing: {Path(exp_dir).name}")
        generate_experiment_report(exp_dir)
        
        analysis_dir = Path(exp_dir) / "analysis"
        if analysis_dir.exists():
            files = list(analysis_dir.glob("*.png"))
            if files:
                print(f"    📈 Generated {len(files)} visualizations")
            else:
                print("    ℹ️  Visualizations require matplotlib/pandas")


def demo_logging_outputs():
    """Show the different logging outputs available."""
    print("\n" + "=" * 80)
    print("📂 Demo: Logging Outputs Overview") 
    print("=" * 80)
    
    exp_dir = "./demo_experiments_m2/demo_dispatching_m2"
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print("⚠️  No experiment data found")
        return
    
    print(f"\n📁 Experiment Structure: {exp_dir}")
    
    # Show directory structure
    for item in sorted(exp_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(exp_path)
            size = item.stat().st_size
            print(f"  📄 {rel_path} ({size} bytes)")
    
    # Show CSV data sample
    csv_file = exp_path / "episodes.csv"
    if csv_file.exists():
        print(f"\n📊 Sample Episode Data:")
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            print("  " + lines[0].strip())  # Header
            if len(lines) > 1:
                print("  " + lines[1].strip())  # First episode
            if len(lines) > 2:
                print("  ...")
            if len(lines) > 1:
                print("  " + lines[-1].strip())  # Last episode
    
    # Show TensorBoard info
    tb_dir = exp_path / "tensorboard"
    if tb_dir.exists():
        print(f"\n📈 TensorBoard Logs:")
        print(f"  📁 {tb_dir}")
        tb_files = list(tb_dir.glob("*"))
        print(f"  📄 {len(tb_files)} log files generated")
        print(f"  💡 View with: tensorboard --logdir={tb_dir}")


def print_milestone2_summary():
    """Print summary of Milestone 2 capabilities."""
    print("\n" + "=" * 80)
    print("🎯 Milestone 2 Summary: Logging and Metrics Framework")
    print("=" * 80)
    
    features = [
        "✅ Comprehensive metrics collection (EpisodeMetrics, ExperimentSummary)",
        "✅ Multi-format logging (CSV, JSON, TensorBoard)",  
        "✅ Centralized ExperimentTracker with timing and metadata",
        "✅ Multi-experiment comparison and analysis",
        "✅ Rolling statistics and convergence analysis",
        "✅ Configurable logging via TOML files",
        "✅ TensorBoard integration for real-time monitoring",
        "✅ Analysis tools with matplotlib/pandas support",
        "✅ CLI tools for experiment comparison and visualization",
        "✅ Integration with existing solution methods"
    ]
    
    print("\n🔧 Key Features Implemented:")
    for feature in features:
        print(f"  {feature}")
    
    print("\n📈 Available Analysis Tools:")
    tools = [
        "compare_methods.py - Compare multiple experiments",
        "visualize_experiment.py - Analyze single experiment",
        "Learning curves with moving averages",
        "Performance metrics over time",
        "Reward distribution analysis",
        "Convergence and stability analysis"
    ]
    
    for tool in tools:
        print(f"  📊 {tool}")
    
    print("\n🚀 Usage Examples:")
    print("  # Run experiment with logging")
    print("  python run_experiment.py configs/training/dispatching.toml")
    print("")
    print("  # Compare multiple methods")
    print("  python analysis/compare_methods.py --experiments-dir ./experiments --plot")
    print("")
    print("  # Analyze single experiment")  
    print("  python analysis/visualize_experiment.py ./experiments/my_experiment")
    print("")
    print("  # View TensorBoard logs")
    print("  tensorboard --logdir ./experiments/my_experiment/tensorboard")


def main():
    """Main demo function."""
    print("🎯 Gym Cooking JSBE Refactor - Milestone 2 Demo")
    print("Logging and Metrics Framework with TensorBoard Support")
    
    try:
        # Run demo experiments
        experiment_dirs = run_demo_experiments()
        
        # Demo analysis tools
        demo_analysis_tools(experiment_dirs)
        
        # Show logging outputs
        demo_logging_outputs()
        
        # Print summary
        print_milestone2_summary()
        
        print("\n✨ Milestone 2 Demo Complete!")
        print("\nNext Steps:")
        print("  🔍 Explore the generated experiment data")
        print("  📊 Try the analysis tools on your own experiments") 
        print("  🚀 Ready for Milestone 3: Deep RL Integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
