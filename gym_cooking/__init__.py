from gymnasium.envs.registration import register

register(
        id="overcookedEnv-v0",
        entry_point="envs:OvercookedEnvironment",
        )

register(
        id="leanOvercookedEnv-v0",
        entry_point="envs:LeanOvercookedEnvironment",
        )