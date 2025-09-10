from typing import Dict, Any, Tuple
import random
import time

from envs.jobshop_gym_env import JobShopEnv

# Import metrics (with fallback if not available)
try:
    from utils.experiment import create_experiment_tracker
    from utils.metrics import EpisodeMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def run_random(env: JobShopEnv,
               cfg: Dict[str, Any],
               mode: str = "eval") -> Tuple[float, Dict[str, Any]]:
    """Run a random policy as a baseline.

    Config options:
        max_steps: int (optional) - limit episode length
        episodes: int (default 1) - number of episodes to run
        use_metrics: bool (default False) - whether to use metrics logging
    """
    max_steps = cfg.get("max_steps", None)
    episodes = int(cfg.get("episodes", 1))
    use_metrics = cfg.get("use_metrics", False) and METRICS_AVAILABLE
    
    # Initialize metrics tracking if requested
    tracker = None
    if use_metrics:
        metrics_config = {
            'method': {'name': 'random_baseline'},
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
        steps = 0
        num_operations = 0
        
        n_actions = env.action_space.n
        while not done:
            action = random.randrange(n_actions)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            num_operations += info.get('num_ops', 0)
            steps += 1
            
            if max_steps is not None and steps >= max_steps:
                break
        
        execution_time = time.time() - start_time
        jobs_completed = info.get('jobs_completed', 0)
        
        # Log metrics if tracker available
        if tracker:
            tracker.log_episode(
                reward=episode_reward,
                num_operations=num_operations,
                jobs_completed=jobs_completed,
                execution_time=execution_time,
                method_specific={'steps': steps, 'max_steps': max_steps}
            )
        
        total_rewards.append(episode_reward)
        all_info.append({
            'episode_reward': episode_reward,
            'jobs_completed': jobs_completed,
            'num_operations': num_operations,
            'steps': steps,
            'execution_time': execution_time
        })
    
    # Finalize metrics tracking
    if tracker:
        summary = tracker.finalize()
    
    # Return average reward and aggregated info
    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward, {
        "episodes": episodes,
        "total_rewards": total_rewards,
        "avg_reward": avg_reward,
        "episode_info": all_info,
        "metrics_logged": use_metrics
    }
