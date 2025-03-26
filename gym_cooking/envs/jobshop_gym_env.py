import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import simpy
from machine import Machine
from utils.agent import RealMachine, COLORS
from schedulingrules import *
import copy
from state_calculator import StateCalculator
from ddqnscheduler.scheduler import SchedulingAgent as Scheduler
from utils.core import Order


class JobShopEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scheduler: Scheduler, num_machines: int, globalSchedule, end_time=1200000):
        super(JobShopEnv, self).__init__()

        # Define the job shop simulation environment
        self.env = simpy.Environment()
        self.processable_jobs = []
        self.state = np.zeros(4)  # Initial Production State [0,0,0,0]
        self.state_calculator = StateCalculator()
        self.uncompleted_jobs = []
        self.scheduler = scheduler
        self.machines = []
        self.num_op_exceuted = 0
        self.schedule = pd.DataFrame(columns=['Time', 'Machine', 'Task', 'Points'])
        self.jobs_completed = 0
        self.deliverred_rewards = []
        self.globalschedule = globalSchedule
        self.end_time = end_time

        # Create num machines
        for i in range(num_machines):
            newMachine = RealMachine(
                jobshop_env=self.env,
                name='agent-' + str(len(self.machines) + 1),
                capacity=1,
                id_color=COLORS[len(self.machines)]
            )
            self.machines.append(newMachine)

        # Define action and observation space
        # Example: action space is discrete with number of machines
        self.action_space = spaces.Discrete(num_machines)
        # Example: observation space is a box with the state size
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()
        self.processable_jobs = []
        self.state = np.zeros(4)
        self.uncompleted_jobs = []
        self.num_op_exceuted = 0
        self.schedule = pd.DataFrame(columns=['Time', 'Machine', 'Task', 'Points'])
        self.jobs_completed = 0
        self.deliverred_rewards = []
        self.end_time = 0

        # Reset machines
        self.machines = []
        for i in range(len(self.machines)):
            newMachine = RealMachine(
                jobshop_env=self.env,
                name='agent-' + str(len(self.machines) + 1),
                capacity=1,
                id_color=COLORS[len(self.machines)]
            )
            self.machines.append(newMachine)

        return self.state

    def step(self, action):
        # Execute one time step within the environment
        # For simplicity, assume action is the index of the machine to process the next job
        if len(self.processable_jobs) == 0:
            done = True
            reward = 0
        else:
            selected_machine = self.machines[action]
            selected_job = self.processable_jobs.pop(0)
            self.env.process(self.job_process(selected_machine, selected_job, action))
            self.env.run(until=self.env.now + 1)  # Run the simulation for one time step
            reward = selected_job.last_task_reward
            done = self.calculate_done()

        self.state = self.state_calculator.calculate_state_features(self.uncompleted_jobs, self.machines)
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f"Current state: {self.state}")

    def job_process(self, machine: Machine, job: Order, policy):
        next_op = job.get_next_operation()
        with machine.queue.request() as request:
            yield request
            yield self.env.process(machine.process_job(job))
            if not job.get_completed():
                self.processable_jobs.append(job)

            self.num_op_exceuted += 1
            prev_state = self.state
            self.state = self.state_calculator.calculate_state_features(self.uncompleted_jobs, self.machines)
            reward = job.last_task_reward

            if job.get_completed():
                self.jobs_completed += 1
                self.deliverred_rewards.append(job.last_task_reward)

            self.rewards.append(reward)
            self.schedule = pd.concat([self.schedule, pd.DataFrame([[job.last_task_completion_timestamp, machine.name, str(next_op), reward]], columns=self.schedule.columns)], axis=0, ignore_index=True)
            self.scheduler.observation(prev_state, policy, reward, self.state, self.calculate_done())
            self.reschedule()

    def reschedule(self):
        if len(self.processable_jobs) > 0:
            policy = self.scheduler.choose_action(self.state)
            scheduling_rule = scheduling_rules[policy]
            selected_job, selected_machine = scheduling_rule(self.processable_jobs, self.machines)
            self.processable_jobs.remove(selected_job)
            self.env.process(self.job_process(selected_machine, selected_job, policy))

    def generate_jobs(self):
        for order in self.globalschedule:
            yield self.env.timeout(order.queued_at - self.env.now)
            job_name = order.full_name
            order2 = copy.deepcopy(order)
            self.processable_jobs.append(order2)
            self.uncompleted_jobs.append(order2)
            print(f"{self.env.now:.2f}: {job_name} arrived")
            self.reschedule()

    def calculate_done(self):
        if self.env.now == self.end_time:
            return True
        return False