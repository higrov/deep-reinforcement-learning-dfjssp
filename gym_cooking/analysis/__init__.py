"""
Analysis and visualization tools for experiment results.

This module provides utilities for:
- Comparing multiple experiments
- Visualizing individual experiment results
- Generating comprehensive reports
"""

from .compare_methods import compare_experiments, plot_comparison
from .visualize_experiment import generate_experiment_report

__all__ = [
    'compare_experiments',
    'plot_comparison', 
    'generate_experiment_report'
]
