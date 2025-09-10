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
    total_reward: float
    completion_time: float
    efficiency: float
    waste_penalty: float
    steps: int
    success: bool
    
    # Optional training metrics
    epsilon: Optional[float] = None
    loss: Optional[float] = None
    
    # Optional performance metrics
    execution_time: Optional[float] = None
    makespan: Optional[float] = None
    
    # Optional environment metrics - backwards compatibility
    reward: Optional[float] = None  # alias for total_reward
    num_operations: Optional[int] = None  # alias for steps
    jobs_completed: Optional[int] = None  # calculated from efficiency
    
    # Optional method-specific metrics
    method_specific: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        
        # Set aliases for backward compatibility
        if self.reward is None:
            self.reward = self.total_reward
        if self.num_operations is None:
            self.num_operations = self.steps
        if self.jobs_completed is None:
            self.jobs_completed = int(self.efficiency * 10)  # rough estimate
    
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
    
    @classmethod
    def from_episodes(cls, method_name: str, episodes: List[EpisodeMetrics], runtime: float) -> 'ExperimentSummary':
        """Create ExperimentSummary from episode list."""
        if not episodes:
            raise ValueError("No episodes provided")
        
        stats = calculate_summary_stats(episodes)
        start_time = time.time() - runtime
        
        return cls(
            method_name=method_name,
            total_episodes=len(episodes),
            total_runtime=runtime,
            mean_reward=stats['mean_reward'],
            std_reward=stats['std_reward'],
            min_reward=stats['min_reward'],
            max_reward=stats['max_reward'],
            mean_jobs_completed=stats['mean_jobs_completed'],
            mean_operations=stats['mean_operations'],
            best_episode=stats['best_episode'],
            best_reward=stats['best_reward'],
            config={},
            start_time=start_time,
            end_time=time.time()
        )


class ComparisonMetrics:
    """Metrics for comparing multiple methods."""
    
    def __init__(self, summaries: List[ExperimentSummary]):
        self.summaries = {s.method_name: s for s in summaries}
        self.methods = list(self.summaries.keys())
        
        # Find best method by mean reward
        self.best_method = max(self.methods, key=lambda m: self.summaries[m].mean_reward)
        
        # Create rankings
        sorted_methods = sorted(self.methods, key=lambda m: self.summaries[m].mean_reward, reverse=True)
        self.reward_rankings = {method: idx + 1 for idx, method in enumerate(sorted_methods)}
        
        self.statistical_significance: Optional[Dict[str, float]] = None
    
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
    
    rewards = [m.total_reward for m in metrics_list]
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
