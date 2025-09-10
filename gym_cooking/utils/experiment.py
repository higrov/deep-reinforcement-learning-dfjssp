"""
High-level experiment tracking and management.

This module provides ExperimentTracker for managing complete experimental runs,
integrating with the metrics logging system and providing comparison utilities.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .metrics import EpisodeMetrics, ExperimentSummary, ComparisonMetrics
from .logging_utils import MetricsLogger, TimingContext


class ExperimentTracker:
    """High-level experiment tracking and management."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 output_dir: str = "./experiments",
                 auto_name: bool = True):
        """
        Initialize experiment tracker.
        
        Args:
            config: Full experiment configuration
            output_dir: Base output directory
            auto_name: Generate automatic experiment name if not provided
        """
        self.config = config
        self.base_output_dir = Path(output_dir)
        
        # Generate experiment name
        self.experiment_name = self._generate_experiment_name(config, auto_name)
        self.output_dir = self.base_output_dir / self.experiment_name
        
        # Initialize logger based on config
        logging_config = config.get('logging', {})
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            output_dir=str(self.output_dir),
            enable_csv=logging_config.get('save_csv', True),
            enable_json=logging_config.get('save_json', True),
            enable_tensorboard=logging_config.get('save_tensorboard', False),
            tensorboard_dir=logging_config.get('tensorboard_dir', None)
        )
        
        # Experiment metadata
        self.method_name = config.get('method', {}).get('name', 'unknown')
        self.start_time = time.time()
        self.current_episode = 0
        
        # Save config
        self._save_config()
    
    def _generate_experiment_name(self, config: Dict[str, Any], auto_name: bool) -> str:
        """Generate experiment name based on config."""
        # Check for explicit name
        exp_name = config.get('logging', {}).get('experiment_name')
        if exp_name:
            return exp_name
        
        if not auto_name:
            return "experiment"
        
        # Auto-generate based on method and timestamp
        method = config.get('method', {}).get('name', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{method}_{timestamp}"
    
    def _save_config(self):
        """Save experiment configuration."""
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_episode(self, 
                   reward: float,
                   num_operations: int,
                   jobs_completed: int,
                   **kwargs) -> EpisodeMetrics:
        """
        Log metrics for a single episode.
        
        Args:
            reward: Episode reward
            num_operations: Number of operations executed
            jobs_completed: Number of jobs completed
            **kwargs: Additional metrics (epsilon, loss, execution_time, etc.)
        
        Returns:
            EpisodeMetrics object that was logged
        """
        # Create metrics object
        metrics = EpisodeMetrics(
            episode=self.current_episode,
            reward=reward,
            num_operations=num_operations,
            jobs_completed=jobs_completed,
            **kwargs
        )
        
        # Log to underlying logger
        self.logger.log_episode(metrics)
        
        # Log summary stats periodically
        summary_freq = self.config.get('logging', {}).get('summary_every', 100)
        if summary_freq > 0:
            self.logger.log_summary_stats(summary_freq)
        
        self.current_episode += 1
        return metrics
    
    def log_evaluation_metrics(self, metrics: EpisodeMetrics):
        """Log metrics from evaluation episodes (separate from training)."""
        # Add evaluation tag
        if metrics.method_specific is None:
            metrics.method_specific = {}
        metrics.method_specific['evaluation'] = True
        
        self.logger.log_episode(metrics)
    
    def time_episode(self) -> TimingContext:
        """Context manager for timing episodes."""
        return TimingContext()
    
    def finalize(self) -> ExperimentSummary:
        """Finalize experiment and return summary."""
        summary = self.logger.finalize_experiment(self.method_name, self.config)
        
        # Save additional experiment metadata
        metadata = {
            'experiment_name': self.experiment_name,
            'total_episodes': self.current_episode,
            'config': self.config,
            'summary': summary.to_dict() if summary else None
        }
        
        metadata_file = self.output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return summary
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current summary statistics."""
        return self.logger.get_current_stats()


class MultiMethodTracker:
    """Track and compare multiple methods/experiments."""
    
    def __init__(self, output_dir: str = "./comparisons"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentSummary] = {}
    
    def add_experiment(self, summary: ExperimentSummary):
        """Add an experiment summary for comparison."""
        self.experiments[summary.method_name] = summary
    
    def load_experiment(self, experiment_dir: str) -> Optional[ExperimentSummary]:
        """Load experiment summary from directory."""
        exp_path = Path(experiment_dir)
        summary_file = exp_path / "experiment_metadata.json"
        
        if not summary_file.exists():
            return None
        
        with open(summary_file, 'r') as f:
            metadata = json.load(f)
        
        summary_data = metadata.get('summary')
        if not summary_data:
            return None
        
        # Reconstruct ExperimentSummary
        summary = ExperimentSummary(**summary_data)
        self.experiments[summary.method_name] = summary
        return summary
    
    def generate_comparison(self) -> ComparisonMetrics:
        """Generate comparison metrics across all loaded experiments."""
        if not self.experiments:
            raise ValueError("No experiments loaded for comparison")
        
        methods = list(self.experiments.keys())
        
        # Find best method by mean reward
        best_method = max(methods, key=lambda m: self.experiments[m].mean_reward)
        
        # Generate rankings
        sorted_methods = sorted(methods, key=lambda m: self.experiments[m].mean_reward, reverse=True)
        rankings = {method: i+1 for i, method in enumerate(sorted_methods)}
        
        comparison = ComparisonMetrics(
            methods=methods,
            summaries=self.experiments,
            best_method=best_method,
            reward_rankings=rankings
        )
        
        return comparison
    
    def save_comparison_report(self, filename: str = "comparison_report.json"):
        """Save detailed comparison report."""
        comparison = self.generate_comparison()
        
        report = {
            'comparison_timestamp': time.time(),
            'methods': comparison.methods,
            'best_method': comparison.best_method,
            'rankings': comparison.reward_rankings,
            'comparison_table': comparison.get_comparison_table(),
            'detailed_summaries': {m: s.to_dict() for m, s in comparison.summaries.items()}
        }
        
        report_file = self.output_dir / filename
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file


def create_experiment_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """Factory function to create experiment tracker from config."""
    return ExperimentTracker(
        config=config,
        output_dir=config.get('logging', {}).get('output_dir', './experiments'),
        auto_name=config.get('logging', {}).get('auto_name', True)
    )
