from typing import Dict, Any, Tuple
import random

from envs.jobshop_gym_env import JobShopEnv
from schedulingrules import scheduling_rules


def run_dispatching(env: JobShopEnv,
                    cfg: Dict[str, Any],
                    mode: str = "eval") -> Tuple[float, Dict[str, Any]]:
    """Run a fixed dispatching rule policy inside the Gym environment.

    Config options:
        rule_index: int (default 0) – index into scheduling_rules

    Returns total accumulated reward and an info dict.
    """
    rule_index = int(cfg.get("rule_index", 0))
    # Defensive: clamp into valid range
    rule_index = max(0, min(rule_index, len(scheduling_rules) - 1))

    done = False
    state = env.reset()
    total_reward = 0.0

    while not done:
        # Use the chosen rule index as the action for this step
        action = rule_index
        _state, reward, done, _info = env.step(action)
        total_reward += reward

    return total_reward, {"episodes": 1, "rule_index": rule_index}
