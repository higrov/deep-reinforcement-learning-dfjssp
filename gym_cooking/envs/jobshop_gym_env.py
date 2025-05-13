import gymnasium as gym
from gymnasium import spaces
import numpy as np
import simpy
import copy
from state_calculator import StateCalculator
from utils.core import Order
from schedulingrules import scheduling_rules
from utils.agent import RealMachine, COLORS

class JobShopEnv(gym.Env):
    """
    OpenAI Gym environment wrapping a SimPy-based job-shop using predefined dispatching rules.

    - Action space: select one of the dispatching rules (0–3)
    - Observation: feature vector from StateCalculator
    - Reward: sum of last task rewards per step
    - Terminates when all jobs done or max_time reached
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, global_schedule: list[Order], num_machines: int,max_time: int = 1200000):
        super().__init__()
        self.global_schedule = global_schedule
        self.num_machines = num_machines
        self.state_calculator = StateCalculator()
        self.max_time = max_time

        # Gym spaces
        self.observation_space = None # to be defined in reset()

        self.action_space = spaces.Discrete(len(scheduling_rules))

        # Internal state
        self.env: simpy.Environment | None = None
        self.processable_jobs: list[Order] = []
        self.uncompleted_jobs: list[Order] = []
        self.machines: list[RealMachine] = []
        self.state: np.ndarray | None = None
        self.last_reward: float = 0.0

    def reset(self) -> np.ndarray:
        # Initialize SimPy environment and components
        self.env = simpy.Environment()
        self.processable_jobs = []
        self.uncompleted_jobs = []
        self.last_reward = 0.0

        # Create machines
        self.machines = []
        for i in range(self.num_machines):
            m = RealMachine(
                jobshop_env=self.env,
                name=f'agent-{i+1}',
                capacity=1,
                id_color=COLORS[i]
            )
            self.machines.append(m)

        # Schedule job arrivals
        self.env.process(self._generate_jobs())

        # Compute initial observation
        self.state = self.state_calculator.calculate_state_features(
            self.uncompleted_jobs, self.machines
        )
        # Define obs space based on state vector length
        obs_len = len(self.state)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_len,), dtype=np.float32
        )
        return self.state

    def step(self, action: int):
        # Dispatch a job if available
        if self.processable_jobs:
            rule_fn = scheduling_rules[action]
            job, machine = rule_fn(self.processable_jobs, self.machines)
            self.processable_jobs.remove(job)
            # Launch processing
            self.env.process(self._job_process(machine, job))

        # Advance simulation to next event or until max_time
        if self.env.now < self.max_time:
            # Step one event
            self.env.step()

        # Compute next observation
        obs = self.state_calculator.calculate_state_features(
            self.uncompleted_jobs, self.machines
        )
        self.state = obs

        # Reward is accumulated from last process
        reward = self.last_reward
        self.last_reward = 0.0

        # Done if time or jobs exhausted
        done = (self.env.now >= self.max_time) or (not self.uncompleted_jobs)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        assert self.env is not None
        print(f"Time={self.env.now:.1f}")
        for m in self.machines:
            busy = len(m.queue.users) > 0
            status = f"busy[{m.queue.users[0].job_name}]" if busy else "idle"
            print(f"  Machine {m.name}: {status}")

    def _generate_jobs(self):
        # Inject jobs based on global schedule
        for order in self.global_schedule:
            yield self.env.timeout(order.queued_at - self.env.now)
            job_copy = copy.deepcopy(order)
            self.processable_jobs.append(job_copy)
            self.uncompleted_jobs.append(job_copy)

    def _job_process(self, machine: RealMachine, job: Order):
        # SimPy routine for processing a job
        with machine.queue.request() as req:
            yield req
            # Actual processing
            yield self.env.process(machine.process_job(job))
            # Collect reward
            self.last_reward += job.last_task_reward
            # Update job queues
            if job.get_completed():
                self.uncompleted_jobs.remove(job)
            else:
                self.processable_jobs.append(job)
