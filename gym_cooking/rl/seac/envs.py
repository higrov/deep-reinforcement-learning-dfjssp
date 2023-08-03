import os

import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from envs.overcooked_environment import OvercookedEnvironment

import gymnasium

class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(env_name, env_arglist, env_id, image_obs, wrappers):
    def _thunk():

        env: OvercookedEnvironment = gymnasium.envs.make(env_name, arglist=env_arglist, env_id=env_id, image_obs=image_obs)
        
        for wrapper in wrappers:
            env = wrapper(env)
        
        return env

    return _thunk


def make_vec_envs(env_name, env_arglist, run_id, image_obs, parallel, wrappers, device):
    envs = [
        make_env(env_name, env_arglist, f'{run_id}_{i}', image_obs, wrappers) for i in range(parallel)
    ]

    if len(envs) == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs, start_method="fork")

    envs = VecPyTorch(envs, device)
    
    return envs

def make_vec_envs_sbln(env_name, env_arglist, run_id, image_obs, parallel, wrappers):
    envs = [
        make_env(env_name, env_arglist, f'{run_id}_{i}', image_obs, wrappers) for i in range(parallel)
    ]

    if len(envs) == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs, start_method="fork")

    return envs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        
    def reset(self):
        obs = self.venv.reset()
        return [torch.from_numpy(o).to(self.device) for o in obs]

    def step_async(self, actions): 
        actions = [a.squeeze(-1).cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return (
            [torch.from_numpy(o).float().to(self.device) for o in obs],
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )
