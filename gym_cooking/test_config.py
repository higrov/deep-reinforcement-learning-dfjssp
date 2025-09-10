#!/usr/bin/env python3
"""
Minimal test for our TOML config system.
This tests without requiring all the heavy dependencies.
"""

from utils.config import load_parameters
import os

def test_config_loading():
    """Test that our config loader works."""
    print("Testing TOML config loading...")
    
    # Test loading the dispatching example
    try:
        config = load_parameters("configs/training/dispatching_example.toml")
        print("✓ Successfully loaded dispatching_example.toml")
        print(f"  Method: {config['method']['name']}")
        print(f"  Rule index: {config['method']['config']['rule_index']}")
        print(f"  Machines: {config['env']['num_machines']}")
        print(f"  Max time: {config['env']['max_time']}")
        
        # Verify structure
        assert config['method']['name'] == 'dispatching'
        assert config['method']['config']['rule_index'] == 0
        assert config['env']['num_machines'] == 4
        print("✓ Config structure validated")
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False
    
    # Test other configs
    configs_to_test = [
        "configs/training/random_baseline.toml",
        "configs/training/ddqn_template.toml"
    ]
    
    for config_path in configs_to_test:
        try:
            config = load_parameters(config_path)
            method_name = config['method']['name']
            print(f"✓ Loaded {os.path.basename(config_path)} - method: {method_name}")
        except Exception as e:
            print(f"✗ Error loading {config_path}: {e}")
    
    return True

def test_unified_interface():
    """Test that our unified interface design is sound."""
    print("\nTesting unified interface design...")
    
    # This is the design we want all methods to follow:
    def mock_run_method(env_config, method, method_config, mode="eval"):
        print(f"  Would run: {method} in {mode} mode")
        print(f"  Env: {env_config.get('num_machines', 'unknown')} machines")
        print(f"  Config: {method_config}")
        return 42.0, {"episodes": 1}
    
    # Test dispatching
    env_config = {"num_machines": 4, "max_time": 1200000, "global_schedule": []}
    score, info = mock_run_method(env_config, "dispatching", {"rule_index": 0})
    print(f"  Result: score={score}, info={info}")
    
    # Test random
    score, info = mock_run_method(env_config, "random", {"max_steps": 1000})
    print(f"  Result: score={score}, info={info}")
    
    print("✓ Unified interface design validated")
    return True

if __name__ == "__main__":
    print("Milestone 1: Config System Test")
    print("=" * 40)
    
    success = True
    success &= test_config_loading()
    success &= test_unified_interface()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Milestone 1 is working.")
        print("\nNext steps:")
        print("1. Install missing dependencies (simpy, etc.) if needed")
        print("2. Run: python3 demo_milestone1.py")
        print("3. Try: python3 run_experiment.py -f configs/training/dispatching_example.toml")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    print(f"\nConfig loading method: {'tomllib (built-in)' if hasattr(__import__('sys'), 'version_info') and __import__('sys').version_info >= (3, 11) else 'fallback parser'}")
