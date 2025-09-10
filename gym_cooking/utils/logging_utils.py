"""
Centralized logging system with multiple output formats.

This module provides MetricsLogger with support for CSV, JSON, and TensorBoard
output, following JSBE-style structured logging principles.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time

from .metrics import EpisodeMetrics, ExperimentSummary, calculate_summary_stats

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class MetricsLogger:
    """Centralized logger for episode metrics with multiple output formats."""
    
    def __init__(self, 
                 experiment_name: str = "experiment",
                 output_dir: str = "./experiments",
                 enable_csv: bool = True,
                 enable_json: bool = True,
                 enable_tensorboard: bool = False,
                 tensorboard_dir: Optional[str] = None):
        
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.episode_metrics: List[EpisodeMetrics] = []
        self.start_time = time.time()
        
        # Initialize outputs
        self._init_csv()
        self._init_tensorboard(tensorboard_dir)
        
        # Setup standard logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup standard Python logging."""
        log_file = self.output_dir / f"{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def _init_csv(self):
        """Initialize CSV output."""
        if not self.enable_csv:
            return
            
        self.csv_file = self.output_dir / f"{self.experiment_name}_episodes.csv"
        self.csv_writer = None
        self.csv_fieldnames = None
    
    def _init_tensorboard(self, tensorboard_dir: Optional[str]):
        """Initialize TensorBoard logging."""
        if not self.enable_tensorboard:
            self.tb_writer = None
            return
            
        if not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard requested but PyTorch not available. Disabling TensorBoard.")
            self.enable_tensorboard = False
            self.tb_writer = None
            return
        
        if tensorboard_dir is None:
            tensorboard_dir = str(self.output_dir / "tensorboard")
        
        tb_path = Path(tensorboard_dir) / self.experiment_name
        self.tb_writer = SummaryWriter(log_dir=str(tb_path))
        self.logger.info(f"TensorBoard logging to: {tb_path}")
    
    def log_episode(self, metrics: EpisodeMetrics):
        """Log metrics for a single episode."""
        self.episode_metrics.append(metrics)
        
        # Log to CSV
        if self.enable_csv:
            self._log_csv(metrics)
        
        # Log to TensorBoard
        if self.enable_tensorboard and self.tb_writer:
            self._log_tensorboard(metrics)
        
        # Log key metrics to console
        self.logger.info(
            f"Episode {metrics.episode}: reward={metrics.reward:.3f}, "
            f"jobs={metrics.jobs_completed}, ops={metrics.num_operations}"
        )
    
    def _log_csv(self, metrics: EpisodeMetrics):
        """Log episode metrics to CSV file."""
        row_data = metrics.to_csv_row()
        
        # Initialize CSV writer if first episode
        if self.csv_writer is None:
            self.csv_fieldnames = list(row_data.keys())
            with open(self.csv_file, 'w', newline='') as f:
                self.csv_writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
        
        # Write row
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            # Fill missing fields with None/empty
            complete_row = {field: row_data.get(field, '') for field in self.csv_fieldnames}
            writer.writerow(complete_row)
    
    def _log_tensorboard(self, metrics: EpisodeMetrics):
        """Log episode metrics to TensorBoard."""
        episode = metrics.episode
        
        # Core metrics
        self.tb_writer.add_scalar('Episode/Reward', metrics.reward, episode)
        self.tb_writer.add_scalar('Episode/Jobs_Completed', metrics.jobs_completed, episode)
        self.tb_writer.add_scalar('Episode/Num_Operations', metrics.num_operations, episode)
        
        # Training metrics
        if metrics.epsilon is not None:
            self.tb_writer.add_scalar('Training/Epsilon', metrics.epsilon, episode)
        if metrics.loss is not None:
            self.tb_writer.add_scalar('Training/Loss', metrics.loss, episode)
        
        # Performance metrics
        if metrics.execution_time is not None:
            self.tb_writer.add_scalar('Performance/Execution_Time', metrics.execution_time, episode)
        if metrics.makespan is not None:
            self.tb_writer.add_scalar('Performance/Makespan', metrics.makespan, episode)
        
        # Environment metrics
        if metrics.num_machines_used is not None:
            self.tb_writer.add_scalar('Environment/Machines_Used', metrics.num_machines_used, episode)
        if metrics.avg_machine_utilization is not None:
            self.tb_writer.add_scalar('Environment/Avg_Machine_Utilization', metrics.avg_machine_utilization, episode)
        
        # Method-specific metrics
        if metrics.method_specific:
            for key, value in metrics.method_specific.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Method/{key}', value, episode)
    
    def log_summary_stats(self, every_n_episodes: int = 100):
        """Log rolling summary statistics."""
        if len(self.episode_metrics) % every_n_episodes != 0:
            return
        
        recent_metrics = self.episode_metrics[-every_n_episodes:]
        stats = calculate_summary_stats(recent_metrics)
        
        episode = len(self.episode_metrics)
        self.logger.info(
            f"Episodes {episode-every_n_episodes+1}-{episode}: "
            f"mean_reward={stats.get('mean_reward', 0):.3f}±{stats.get('std_reward', 0):.3f}, "
            f"best={stats.get('max_reward', 0):.3f}"
        )
        
        if self.enable_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar('Summary/Mean_Reward', stats['mean_reward'], episode)
            self.tb_writer.add_scalar('Summary/Std_Reward', stats['std_reward'], episode)
            self.tb_writer.add_scalar('Summary/Best_Reward', stats['max_reward'], episode)
    
    def finalize_experiment(self, method_name: str, config: Dict[str, Any]) -> ExperimentSummary:
        """Finalize experiment and generate summary."""
        end_time = time.time()
        total_runtime = end_time - self.start_time
        
        if not self.episode_metrics:
            self.logger.warning("No episode metrics collected!")
            return None
        
        stats = calculate_summary_stats(self.episode_metrics)
        
        summary = ExperimentSummary(
            method_name=method_name,
            total_episodes=len(self.episode_metrics),
            total_runtime=total_runtime,
            mean_reward=stats['mean_reward'],
            std_reward=stats['std_reward'],
            min_reward=stats['min_reward'],
            max_reward=stats['max_reward'],
            mean_jobs_completed=stats['mean_jobs_completed'],
            mean_operations=stats['mean_operations'],
            best_episode=stats['best_episode'],
            best_reward=stats['best_reward'],
            config=config,
            start_time=self.start_time,
            end_time=end_time
        )
        
        # Save summary
        if self.enable_json:
            summary_file = self.output_dir / f"{self.experiment_name}_summary.json"
            with open(summary_file, 'w') as f:
                f.write(summary.to_json())
        
        # Log final summary
        self.logger.info(
            f"Experiment completed: {len(self.episode_metrics)} episodes, "
            f"runtime={total_runtime:.2f}s, mean_reward={stats['mean_reward']:.3f}"
        )
        
        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        return summary
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current summary statistics."""
        return calculate_summary_stats(self.episode_metrics)
    
    def save_raw_data(self, filename: Optional[str] = None):
        """Save raw metrics data as JSON."""
        if filename is None:
            filename = f"{self.experiment_name}_raw_metrics.json"
        
        filepath = self.output_dir / filename
        raw_data = [m.to_dict() for m in self.episode_metrics]
        
        with open(filepath, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        self.logger.info(f"Raw metrics saved to: {filepath}")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
