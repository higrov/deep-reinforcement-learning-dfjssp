#!/usr/bin/env python3
"""
Minimal test script for Milestone 2 core functionality.

This script tests the logging and metrics system without requiring simpy or gym environments.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics_basic():
    """Test basic metrics functionality."""
    print("🧪 Testing basic metrics...")
    
    from utils.metrics import EpisodeMetrics, ExperimentSummary, ComparisonMetrics
    
    # Test EpisodeMetrics with new constructor
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
    episodes = [
        episode,
        EpisodeMetrics(2, 95.0, 115.0, 0.82, -3.0, 48, True)
    ]
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


def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing config loading...")
    
    from utils.config import load_config
    
    # Test loading existing config
    config = load_config("configs/training/dispatching_example.toml")
    
    assert 'method' in config
    assert 'env' in config
    assert config['method']['name'] == 'dispatching'
    
    print("  ✅ Config loading working")


def test_logging_basic():
    """Test basic logging functionality."""
    print("🧪 Testing basic logging...")
    
    from utils.logging_utils import MetricsLogger
    from utils.metrics import EpisodeMetrics
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test CSV logging (without TensorBoard)
        logger = MetricsLogger(
            experiment_name="test",
            output_dir=tmp_dir, 
            enable_csv=True, 
            enable_tensorboard=False
        )
        
        episode = EpisodeMetrics(1, 100.0, 120.0, 0.85, -5.0, 50, True)
        logger.log_episode(episode)
        logger.close()
        
        csv_file = Path(tmp_dir) / "episodes.csv"
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            content = f.read()
            assert "episode,total_reward" in content
            assert "1,100.0" in content
        
        print("  ✅ Basic logging working")


def test_experiment_tracker_basic():
    """Test basic experiment tracking."""
    print("🧪 Testing experiment tracker...")
    
    from utils.experiment import ExperimentTracker
    from utils.metrics import EpisodeMetrics
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {
            'logging': {
                'enabled': True,
                'save_csv': True,
                'save_json': True,
                'save_tensorboard': False
            }
        }
        
        tracker = ExperimentTracker(
            experiment_name="test_exp",
            config=config,
            output_dir=tmp_dir
        )
        
        # Log some episodes
        episodes = [
            EpisodeMetrics(1, 100.0, 120.0, 0.85, -5.0, 50, True),
            EpisodeMetrics(2, 95.0, 115.0, 0.82, -3.0, 48, True)
        ]
        
        for episode in episodes:
            tracker.log_episode(episode)
        
        # Finalize
        summary = tracker.finalize_experiment()
        assert summary is not None
        
        # Check outputs
        csv_file = Path(tmp_dir) / "test_exp" / "episodes.csv"
        json_file = Path(tmp_dir) / "test_exp" / "experiment_summary.json"
        metadata_file = Path(tmp_dir) / "test_exp" / "experiment_metadata.json"
        
        assert csv_file.exists()
        assert json_file.exists()
        assert metadata_file.exists()
        
        print("  ✅ ExperimentTracker working")


def run_minimal_tests():
    """Run minimal core tests."""
    print("🧪 Running Minimal Milestone 2 Tests")
    print("=" * 50)
    
    tests = [
        test_metrics_basic,
        test_config_loading,
        test_logging_basic,
        test_experiment_tracker_basic
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
        print("✅ Core Milestone 2 functionality is working!")
    else:
        print("❌ Some core components need attention.")
    
    return failed == 0


def main():
    """Main test function."""
    print("🎯 Gym Cooking JSBE Refactor - Minimal Milestone 2 Tests")
    print("Testing core logging and metrics functionality")
    print()
    
    success = run_minimal_tests()
    
    if success:
        print("\n🎉 Milestone 2 core implementation is working correctly!")
        print("The logging and metrics framework is ready for use.")
    else:
        print("\n⚠️  Some core components need attention before proceeding.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
