from typing import Dict, Any, Tuple
import random

from envs.jobshop_gym_env import JobShopEnv


def run_random(env: JobShopEnv,
               cfg: Dict[str, Any],
               mode: str = "eval") -> Tuple[float, Dict[str, Any]]:
    """Run a random policy as a baseline.

    Ignores cfg except for optional max_steps to truncate episodes.
    """
    done = False
    state = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = cfg.get("max_steps", None)

    n_actions = env.action_space.n
    while not done:
        action = random.randrange(n_actions)
        state, reward, done, _info = env.step(action)
        total_reward += reward
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return total_reward, {"episodes": 1, "steps": steps}
