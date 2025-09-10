#!/usr/bin/env python3
"""
Demo script for Milestone 1: JSBE-inspired refactoring

This script demonstrates:
1. TOML config loading
2. Unified runner interface
3. Dispatching rules and random baseline
4. Integration with existing ScheduleGenerator

Run with:
    python demo_milestone1.py
"""

from typing import List
from utils.config import load_parameters
from solution_methods.runner import run_method
from schedule_generator import ScheduleGenerator
from utils.core import Order


def get_sample_schedule() -> List[Order]:
    """Get a sample schedule for testing.
    
    Falls back to a basic schedule if ScheduleGenerator fails.
    """
    try:
        generator = ScheduleGenerator()
        schedules = generator.generateSchedule()
        # Get first schedule from the generated schedules
        if schedules and len(schedules) > 0:
            first_key = list(schedules.keys())[0]
            return schedules[first_key]
    except Exception as e:
        print(f"Warning: ScheduleGenerator failed: {e}")
        print("Using fallback empty schedule for demo")
        return []
    
    return []


def demo_dispatching():
    """Demo dispatching rule."""
    print("\\n=== Demo: Dispatching Rule ===")
    
    # Get a sample schedule
    schedule = get_sample_schedule()
    print(f"Using schedule with {len(schedule)} orders")
    
    # Create config programmatically (mimics TOML loading)
    config = {
        "env": {
            "global_schedule": schedule,
            "num_machines": 4,
            "max_time": 1200000
        },
        "method": {
            "name": "dispatching",
            "config": {
                "rule_index": 0  # Use first dispatching rule
            }
        }
    }
    
    try:
        score, info = run_method(
            config["env"], 
            config["method"]["name"], 
            config["method"]["config"], 
            mode="eval"
        )
        print(f"Dispatching Rule Result: Score={score:.3f}, Info={info}")
    except Exception as e:
        print(f"Error running dispatching rule: {e}")


def demo_random():
    """Demo random baseline."""
    print("\\n=== Demo: Random Baseline ===")
    
    schedule = get_sample_schedule()
    
    config = {
        "env": {
            "global_schedule": schedule,
            "num_machines": 4,
            "max_time": 1200000
        },
        "method": {
            "name": "random",
            "config": {
                "max_steps": 1000  # Limit steps for quick demo
            }
        }
    }
    
    try:
        score, info = run_method(
            config["env"], 
            config["method"]["name"], 
            config["method"]["config"], 
            mode="eval"
        )
        print(f"Random Baseline Result: Score={score:.3f}, Info={info}")
    except Exception as e:
        print(f"Error running random baseline: {e}")


def demo_config_loading():
    """Demo TOML config loading."""
    print("\\n=== Demo: TOML Config Loading ===")
    
    try:
        config = load_parameters("configs/training/dispatching_example.toml")
        print("Successfully loaded TOML config:")
        print(f"  Method: {config['method']['name']}")
        print(f"  Rule index: {config['method']['config']['rule_index']}")
        print(f"  Machines: {config['env']['num_machines']}")
        print(f"  Max time: {config['env']['max_time']}")
    except Exception as e:
        print(f"Error loading TOML config: {e}")


if __name__ == "__main__":
    print("Gym Cooking Milestone 1 Demo")
    print("============================")
    
    demo_config_loading()
    demo_dispatching()
    demo_random()
    
    print("\\n=== Next Steps ===")
    print("1. Run: python run_experiment.py -f configs/training/dispatching_example.toml")
    print("2. (You'll need to populate global_schedule programmatically)")
    print("3. Try different rule_index values (0-3 based on schedulingrules.py)")
    print("4. Compare with random baseline")
