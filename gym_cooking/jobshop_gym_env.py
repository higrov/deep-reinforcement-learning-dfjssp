import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from schedule_generator import ScheduleGenerator
from sklearn.model_selection import train_test_split
from envs.jobshop_env import JobShop

def getSchedule(train = True):
    scheduleGenerator =  ScheduleGenerator()
    listofglobalschedule = scheduleGenerator.generateSchedule()
    if train:
        train_schedule, _= train_test_split(listofglobalschedule, train_size=0.7)
        return train_schedule
    else:
        
        _, test_schedule= train_test_split(listofglobalschedule, train_size=0.7)
        return test_schedule

class JobShopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scheduler, num_machines=4, max_steps=1200000):
        super(JobShopEnv, self).__init__()
        self.scheduler = scheduler
        self.num_machines = num_machines
        self.max_steps = max_steps
        self.listofglobalschedule = getSchedule(train=False)
        self.action_space = spaces.Discrete(len(self.listofglobalschedule))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.listofglobalschedule),), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.j = 0
        self.max_reward = 0
        self.schedules = []
        self.test_log = pd.DataFrame(columns=['episode', 'reward', 'num_op_executed', 'jobs_completed'])
        self.globalSchedule = sorted(self.listofglobalschedule[self.j], key=lambda x: x.queued_at)
        self.job_shop = JobShop(scheduler=self.scheduler, num_machines=self.num_machines, globalSchedule=self.globalSchedule)
        return self._next_observation()

    def _next_observation(self):
        # Return the current state of the environment
        return np.array([self.current_step / self.max_steps])

    def step(self, action):
        self.current_step += 1
        self.globalSchedule = sorted(self.listofglobalschedule[action], key=lambda x: x.queued_at)
        self.job_shop = JobShop(scheduler=self.scheduler, num_machines=self.num_machines, globalSchedule=self.globalSchedule)
        self.job_shop.run(self.max_steps)

        reward = np.sum(self.job_shop.rewards)
        done = self.current_step >= self.max_steps

        if reward > self.max_reward:
            self.max_reward = reward
            self.schedules.append((self.job_shop.schedule, reward))

        self.j += 1
        if self.j >= len(self.listofglobalschedule):
            random.shuffle(self.listofglobalschedule)
            self.j = 0

        self.test_log = pd.concat([self.test_log, pd.DataFrame([[self.current_step, reward, self.job_shop.num_op_executed, self.job_shop.jobs_completed]], columns=self.test_log.columns)], axis=0, ignore_index=True)

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        # Implement visualization if needed
        pass

# Register the environment with Gym
gym.envs.registration.register(
    id='JobShop-v0',
    entry_point='__main__:JobShopEnv',
)

# Example usage
env = gym.make('JobShop-v0', scheduler=scheduler)
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your action selection logic
    obs, reward, done, info = env.step(action)
    env.render()