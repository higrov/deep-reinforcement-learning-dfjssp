"""
Core metrics data classes and structures for experiment tracking.

This module provides JSBE-style structured metrics collection with support for
multiple output formats including CSV, JSON, and TensorBoard.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import time
import json
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """Metrics collected for a single episode."""
    
    episode: int
    reward: float
    num_operations: int
    jobs_completed: int
    
    # Optional training metrics
    epsilon: Optional[float] = None
    loss: Optional[float] = None
    
    # Optional performance metrics
    execution_time: Optional[float] = None
    makespan: Optional[float] = None
    
    # Optional environment metrics
    num_machines_used: Optional[int] = None
    avg_machine_utilization: Optional[float] = None
    
    # Optional method-specific metrics
    method_specific: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to CSV-compatible row (flatten nested dicts)."""
        row = self.to_dict()
        
        # Flatten method_specific metrics
        if self.method_specific:
            for key, value in self.method_specific.items():
                row[f"method_{key}"] = value
            del row["method_specific"]
        
        return row


@dataclass
class ExperimentSummary:
    """Summary statistics for a complete experiment."""
    
    method_name: str
    total_episodes: int
    total_runtime: float
    
    # Reward statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    # Performance statistics
    mean_jobs_completed: float
    mean_operations: float
    
    # Best performance
    best_episode: int
    best_reward: float
    
    # Configuration
    config: Dict[str, Any]
    
    # Timestamps
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ComparisonMetrics:
    """Metrics for comparing multiple methods."""
    
    methods: List[str]
    summaries: Dict[str, ExperimentSummary]
    
    # Cross-method comparisons
    best_method: str
    reward_rankings: Dict[str, int]
    statistical_significance: Optional[Dict[str, float]] = None
    
    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """Generate comparison table for easy visualization."""
        table = {}
        for method, summary in self.summaries.items():
            table[method] = {
                'mean_reward': summary.mean_reward,
                'std_reward': summary.std_reward,
                'best_reward': summary.best_reward,
                'mean_jobs_completed': summary.mean_jobs_completed,
                'total_episodes': summary.total_episodes,
                'runtime': summary.total_runtime
            }
        return table


def calculate_summary_stats(metrics_list: List[EpisodeMetrics]) -> Dict[str, float]:
    """Calculate summary statistics from a list of episode metrics."""
    if not metrics_list:
        return {}
    
    rewards = [m.reward for m in metrics_list]
    jobs_completed = [m.jobs_completed for m in metrics_list]
    operations = [m.num_operations for m in metrics_list]
    
    import statistics
    
    stats = {
        'mean_reward': statistics.mean(rewards),
        'median_reward': statistics.median(rewards),
        'std_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        'min_reward': min(rewards),
        'max_reward': max(rewards),
        'mean_jobs_completed': statistics.mean(jobs_completed),
        'mean_operations': statistics.mean(operations),
    }
    
    # Find best episode
    best_idx = rewards.index(max(rewards))
    stats['best_episode'] = metrics_list[best_idx].episode
    stats['best_reward'] = max(rewards)
    
    return stats
