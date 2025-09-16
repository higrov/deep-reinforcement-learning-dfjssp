# Environment imports moved to lazy loading to avoid dependency coupling
# Import specific environments only when needed:
#
# For job shop scheduling:
#   from envs.jobshop_gym_env import JobShopEnv
#
# For overcooked simulation:
#   from envs.overcooked_environment import OvercookedEnvironment
#
# This prevents forcing heavy dependencies when only lightweight components are needed
