from typing import Dict, Any, Tuple
import random
import time

from envs.jobshop_gym_env import JobShopEnv
from schedulingrules import scheduling_rules

# Import metrics (with fallback if not available)
try:
    from utils.experiment import create_experiment_tracker
    from utils.metrics import EpisodeMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def run_dispatching(env: JobShopEnv,
                    cfg: Dict[str, Any],
                    mode: str = "eval") -> Tuple[float, Dict[str, Any]]:
    """Run a fixed dispatching rule policy inside the Gym environment.

    Config options:
        rule_index: int (default 0) – index into scheduling_rules
        episodes: int (default 1) – number of episodes to run
        use_metrics: bool (default False) – whether to use metrics logging

    Returns total accumulated reward and an info dict.
    """
    rule_index = int(cfg.get("rule_index", 0))
    episodes = int(cfg.get("episodes", 1))
    use_metrics = cfg.get("use_metrics", False) and METRICS_AVAILABLE
    
    # Defensive: clamp rule index into valid range
    rule_index = max(0, min(rule_index, len(scheduling_rules) - 1))
    
    # Initialize metrics tracking if requested
    tracker = None
    if use_metrics:
        # Create a minimal config for metrics
        metrics_config = {
            'method': {'name': f'dispatching_rule_{rule_index}'},
            'logging': cfg.get('logging', {'save_csv': True, 'save_tensorboard': False})
        }
        tracker = create_experiment_tracker(metrics_config)
    
    total_rewards = []
    all_info = []
    
    for episode in range(episodes):
        start_time = time.time()
        done = False
        state = env.reset()
        episode_reward = 0.0
        num_operations = 0
        
        while not done:
            # Use the chosen rule index as the action for this step
            action = rule_index
            _state, reward, done, info = env.step(action)
            episode_reward += reward
            num_operations += info.get('num_ops', 0)
        
        execution_time = time.time() - start_time
        jobs_completed = info.get('jobs_completed', 0)
        
        # Log metrics if tracker available
        if tracker:
            tracker.log_episode(
                reward=episode_reward,
                num_operations=num_operations,
                jobs_completed=jobs_completed,
                execution_time=execution_time,
                method_specific={'rule_index': rule_index, 'rule_name': f'rule_{rule_index}'}
            )
        
        total_rewards.append(episode_reward)
        all_info.append({
            'episode_reward': episode_reward,
            'jobs_completed': jobs_completed,
            'num_operations': num_operations,
            'execution_time': execution_time
        })
    
    # Finalize metrics tracking
    if tracker:
        summary = tracker.finalize()
    
    # Return average reward and aggregated info
    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward, {
        "episodes": episodes,
        "rule_index": rule_index,
        "total_rewards": total_rewards,
        "avg_reward": avg_reward,
        "episode_info": all_info,
        "metrics_logged": use_metrics
    }
