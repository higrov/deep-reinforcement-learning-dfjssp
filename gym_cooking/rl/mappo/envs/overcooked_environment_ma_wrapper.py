import logging
import random
import time
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import copy
from envs.observation_helpers import ObservationHelpers

from envs.overcooked_environment import OvercookedEnvironment


logger = logging.getLogger(__name__)


class OverCookedMAEnv(object):
    '''Wrapper to make Overcooked Environment compatible for Multi-Agent RL'''

    def __init__(self, env_name, arglist, env_id, share_reward, from_env=None):
        if from_env:
            self.env: OvercookedEnvironment = from_env
            arglist = self.env.arglist
        else:
            self.env: OvercookedEnvironment = gym.envs.make(env_name, arglist=arglist, env_id=env_id, early_termination=True)

        self.arglist = arglist
        self.num_agents = arglist.num_agents
        self.max_num_agents = 5
        self.max_num_orders = 5
        self.max_num_levels = 5
        # make env
        self.max_steps = arglist.num_total_timesteps
        self.share_reward = share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        for idx in range(self.max_num_agents):
            self.action_space.append(spaces.Discrete(
                n=self.env.action_space[idx].n
            ))
            osp = self.flatten_dict_space(self.env.observation_space[idx])
            self.observation_space.append(osp)
            self.share_observation_space.append(osp)

    def __copy__(self):
        newone = type(self)("", self.arglist, self.env_id, self.share_reward, from_env=self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone
    
    @classmethod
    def fromEnv(cls, env: OvercookedEnvironment, share_reward=True):
        return cls(env_name=None, arglist=None, env_id=None, share_reward=share_reward, from_env=env)
        
    def reset(self):
        obs = self.env.reset(reload_level=True)
        obs = self._obs_wrapper(obs)
        self.ep_start = time.perf_counter()
        return obs

    def step(self, actions):
        agent_actions = self._action_wrapper(actions)
        obs, reward, done, info = self.env.step(agent_actions)
        obs = self._obs_wrapper(obs)
        reward = self._reward_wrapper(reward)

        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.max_num_agents

        if all(done):
            info['episode_duration'] = time.perf_counter() - self.ep_start
            info['episode_length'] = self.env.t
            #logger.info(f"Environment {self.env.env_id} is done!!! {self.env.termination_info}")
        done = np.array(done)
        info = self._info_wrapper(info)

        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
            self.env.seed(1)
        else:
            random.seed(seed)
            self.env.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        _obs = np.array([ObservationHelpers.flatten_dict_obs(_) for _ in obs]) if self.env.n_agents > 0 else None
        #_obs = np.array([ObservationHelpers.normalize_flattened_obs(_) for _ in _obs])
        return _obs

    def _action_wrapper(self, actions):
        if type(actions) is dict:
            return actions
        return {f'agent-{i+1}': action for i, action in enumerate(actions)}

    def _reward_wrapper(self, reward):
        if self.num_agents == 1:
            return np.array(reward[0]).reshape(1, 1)
        else:
            return np.array(reward).reshape(self.max_num_agents, 1)

    def _info_wrapper(self, info):
        #state = self.env.unwrapped.get_state()
        #info.update(state._asdict())
        return info

    def generate_animation(self, t, suffix):
        return self.env.generate_animation(t, suffix)

    def get_animation_path(self):
        return self.env.get_animation_path()

    def flatten_dict_space(self, space: spaces.Dict):
        # dict of it all
        # obs_structure = {
        #     'agents': spaces.Dict({
        #         'self': agent_space,
        #         'others': spaces.Tuple([agent_space] * (ObservationHelpers.MAX_NUM_AGENTS - 1)),
        #     }),
        #     'dynamic_objects': spaces.Tuple([dynamic_object_space] * ObservationHelpers.MAX_NUM_OBJECTS),
        #     'stations': spaces.Dict({
        #         'prep_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_PREP_STATIONS),
        #         'delivery_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_DELIVERY_STATIONS),
        #     }),
        #     'orders': spaces.Tuple([order_space] * ObservationHelpers.MAX_NUM_ORDERS),
        # }
        
        return ObservationHelpers.get_observation_space_structure_flattened()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    
