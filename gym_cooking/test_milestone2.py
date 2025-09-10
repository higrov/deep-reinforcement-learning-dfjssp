#!/usr/bin/env python3
"""
Test script for Milestone 2: Logging and Metrics Framework

This script provides comprehensive tests for all logging and metrics components.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics():
    """Test metrics data classes."""
    print("🧪 Testing metrics data classes...")
    
    from utils.metrics import EpisodeMetrics, ExperimentSummary, ComparisonMetrics
    
    # Test EpisodeMetrics
    episode = EpisodeMetrics(
        episode=1,
        total_reward=100.5,
        completion_time=120.0,
        efficiency=0.85,
        waste_penalty=-5.0,
        steps=50,
        success=True
    )
    
    assert episode.episode == 1
    assert episode.total_reward == 100.5
    assert episode.to_dict()['success'] is True
    print("  ✅ EpisodeMetrics working")
    
    # Test ExperimentSummary
    episodes = [episode, EpisodeMetrics(2, 95.0, 115.0, 0.82, -3.0, 48, True)]
    summary = ExperimentSummary.from_episodes("test_method", episodes, 5.0)
    
    assert summary.method_name == "test_method"
    assert summary.total_episodes == 2
    assert summary.mean_reward == 97.75
    print("  ✅ ExperimentSummary working")
    
    # Test ComparisonMetrics  
    summaries = [summary]
    comparison = ComparisonMetrics(summaries)
    table = comparison.get_comparison_table()
    
    assert "test_method" in table
    assert table["test_method"]["mean_reward"] == 97.75
    print("  ✅ ComparisonMetrics working")


def test_logging_utils():
    """Test logging utilities."""
    print("🧪 Testing logging utilities...")
    
    from utils.logging_utils import MetricsLogger
    from utils.metrics import EpisodeMetrics
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test CSV logging
        logger = MetricsLogger(output_dir=tmp_dir, enable_csv=True, enable_tensorboard=False)
        
        episode = EpisodeMetrics(1, 100.0, 120.0, 0.85, -5.0, 50, True)
        logger.log_episode(episode)
        logger.close()
        
        csv_file = Path(tmp_dir) / "episodes.csv"
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            content = f.read()
            assert "episode,total_reward" in content
            assert "1,100.0" in content
        
        print("  ✅ CSV logging working")


def test_experiment_tracker():
    """Test experiment tracking."""
    print("🧪 Testing experiment tracker...")
    
    from utils.experiment import ExperimentTracker
    from utils.metrics import EpisodeMetrics
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {
            'logging': {
                'enabled': True,
                'csv': True,
                'json': True,
                'tensorboard': False
            }
        }
        
        tracker = ExperimentTracker(
            experiment_name="test_exp",
            output_dir=tmp_dir,
            config=config
        )
        
        # Log some episodes
        episodes = [
            EpisodeMetrics(1, 100.0, 120.0, 0.85, -5.0, 50, True),
            EpisodeMetrics(2, 95.0, 115.0, 0.82, -3.0, 48, True)
        ]
        
        for episode in episodes:
            tracker.log_episode(episode)
        
        # Finalize
        tracker.finalize_experiment()
        
        # Check outputs
        csv_file = Path(tmp_dir) / "episodes.csv"
        json_file = Path(tmp_dir) / "experiment_summary.json"
        metadata_file = Path(tmp_dir) / "experiment_metadata.json"
        
        assert csv_file.exists()
        assert json_file.exists()
        assert metadata_file.exists()
        
        print("  ✅ ExperimentTracker working")


def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing config loading...")
    
    from utils.config import load_config
    
    # Test loading existing config
    config = load_config("configs/training/dispatching.toml")
    
    assert 'method' in config
    assert 'training' in config
    assert config['method']['type'] == 'dispatching'
    
    print("  ✅ Config loading working")


def test_runner_integration():
    """Test runner integration with logging."""
    print("🧪 Testing runner integration...")
    
    from solution_methods.runner import create_runner_from_config
    from utils.config import load_config
    from utils.experiment import ExperimentTracker
    
    # Load minimal config
    config = load_config("configs/training/random.toml")
    config['training']['episodes'] = 2  # Just 2 episodes for test
    config['logging'] = {
        'enabled': True,
        'csv': True,
        'tensorboard': False
    }
    
    runner = create_runner_from_config(config)
    assert runner is not None
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(
            experiment_name="test_integration",
            output_dir=tmp_dir,
            config=config
        )
        
        results = runner.run(
            episodes=2,
            experiment_tracker=tracker
        )
        
        tracker.finalize_experiment()
        
        # Check results
        assert 'mean_reward' in results
        assert 'episodes_completed' in results
        
        # Check files were created
        csv_file = Path(tmp_dir) / "episodes.csv"
        assert csv_file.exists()
        
        print("  ✅ Runner integration working")


def test_analysis_tools():
    """Test analysis tools."""
    print("🧪 Testing analysis tools...")
    
    from analysis import compare_experiments
    from utils.experiment import ExperimentTracker
    from utils.metrics import EpisodeMetrics
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two mock experiments
        experiment_dirs = []
        
        for i, method in enumerate(["method_a", "method_b"]):
            exp_dir = Path(tmp_dir) / f"exp_{method}"
            exp_dir.mkdir()
            
            config = {
                'logging': {
                    'enabled': True,
                    'csv': True,
                    'tensorboard': False
                }
            }
            
            tracker = ExperimentTracker(
                experiment_name=method,
                output_dir=str(exp_dir),
                config=config
            )
            
            # Create mock episodes
            episodes = [
                EpisodeMetrics(1, 100 + i*10, 120.0, 0.85, -5.0, 50, True),
                EpisodeMetrics(2, 95 + i*10, 115.0, 0.82, -3.0, 48, True)
            ]
            
            for episode in episodes:
                tracker.log_episode(episode)
            
            tracker.finalize_experiment()
            experiment_dirs.append(str(exp_dir))
        
        # Test comparison
        comparison_dir = Path(tmp_dir) / "comparison"
        comparison = compare_experiments(experiment_dirs, str(comparison_dir))
        
        assert comparison is not None
        assert comparison.best_method in ["method_a", "method_b"]
        
        # Check comparison files
        assert (comparison_dir / "comparison_report.json").exists()
        
        print("  ✅ Analysis tools working")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Milestone 2 Tests")
    print("=" * 50)
    
    tests = [
        test_metrics,
        test_logging_utils,
        test_experiment_tracker,
        test_config_loading,
        test_runner_integration,
        test_analysis_tools
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Milestone 2 is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return failed == 0


def main():
    """Main test function."""
    print("🎯 Gym Cooking JSBE Refactor - Milestone 2 Tests")
    print("Testing Logging and Metrics Framework")
    print()
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 Milestone 2 implementation is working correctly!")
        print("Ready to proceed with demo and further development.")
    else:
        print("\n⚠️  Some components need attention before proceeding.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
