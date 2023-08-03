
import copy
import math
from collections import deque
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium.wrappers import RecordVideo as GymMonitor


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if all(done):
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            info["episode_duration"] = int(perf_counter() - self.t0)

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)

            
        return observation, reward, done, info

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        multi_agent_flattened_space = spaces.Tuple(tuple(spaces.flatten_space(spc) for spc in env.observation_space))
        
        self.observation_space = spaces.flatten_space(multi_agent_flattened_space)

    def observation(self, observation):
        return spaces.flatten(self.observation_space, tuple(spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env.observation_space, observation)))

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone
class FlattenMultiObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenMultiObservation, self).__init__(env)

        multi_agent_flattened_space = spaces.Tuple(tuple(spaces.flatten_space(spc) for spc in env.observation_space))
        
        self.observation_space = multi_agent_flattened_space

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone


class DictifyActions(gym.ActionWrapper):
    r"""Wrapper that converts tensor actions to dictionary of {'agent-id': action}."""
    def __init__(self, env):
        super().__init__(env)

    def action(self, agent_actions):
        return {f'agent-{i+1}': action for i, action in enumerate(agent_actions)}

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone

class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone

class GlobalizeReward(gym.RewardWrapper):
    r"""Wrapper that converts individual rewards to global rewards."""

    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        return sum(reward) # converts a vector of rewards per agent to singular reward

    def __copy__(self):
        newone = type(self)(self.env)
        newone.__dict__.update(self.__dict__)
        newone.env = copy.copy(self.env)
        return newone

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not all(done)
            done = len(obs) * [True]
        return obs, reward, done, info

    
class ClearInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, {}

    def __copy__(self):
        copied = type(self)(self.env)
        copied.unwrapped = copy.copy(self.unwrapped)
        return copied
class Monitor(GymMonitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, sum(reward), done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done