from typing import Any, Dict, Tuple

from envs.jobshop_gym_env import JobShopEnv


def run_method(env_config: Dict[str, Any],
               method: str,
               method_config: Dict[str, Any],
               mode: str = "eval") -> Tuple[float, Dict[str, Any]]:
    """Unified entry point to run a scheduling method.

    Args:
        env_config: Dict with environment configuration. Must include at least:
            - global_schedule: list of Orders (domain-specific objects)
            - num_machines: int
            - max_time: int
        method: Name of the method to run (e.g., "ddqn", "dispatching", "random").
        method_config: Dict with method-specific configuration.
        mode: "train" or "eval". Not all methods use this.

    Returns:
        Tuple of (score, info). Score is a scalar (e.g., total reward). Info is a
        dict with optional metadata.
    """
    env = JobShopEnv(
        global_schedule=env_config["global_schedule"],
        num_machines=env_config.get("num_machines", 4),
        max_time=env_config.get("max_time", 1_200_000),
    )

    if method == "dispatching":
        from solution_methods.dispatching_rules.rules import run_dispatching
        return run_dispatching(env, method_config, mode)

    if method == "random":
        from solution_methods.random_baseline import run_random
        return run_random(env, method_config, mode)

    if method == "ddqn":
        # Keep placeholder to integrate existing DDQN later without breaking current code.
        try:
            from ddqnscheduler.scheduler import SchedulingAgent as Scheduler  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise NotImplementedError(
                "DDQN pipeline not wired into runner yet."
            ) from exc
        # Minimal evaluation loop using existing Scheduler choose_action API
        done = False
        state = env.reset()
        total_reward = 0.0
        # Fallback defaults
        nb_input = method_config.get("nb_input_params", 4)
        nb_actions = method_config.get("nb_actions", env.action_space.n)
        agent = Scheduler(nb_total_operations=10_000,
                          nb_input_params=nb_input,
                          nb_actions=nb_actions,
                          train=(mode == "train"))
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _info = env.step(action)
            total_reward += reward
        return total_reward, {"episodes": 1}

    raise NotImplementedError(f"Unknown method: {method}")
