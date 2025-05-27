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
        self.num_operations_executed: int = 0
        self.jobs_completed: int = 0
        self._previous_num_operations_executed: int = 0
        self.machines: list[RealMachine] = []
        self.state: np.ndarray | None = None
        self.last_reward: float = 0.0
        self._all_jobs_generated_from_schedule: bool = False

    def reset(self) -> np.ndarray:
        # Initialize SimPy environment and components
        self.env = simpy.Environment()
        self._all_jobs_generated_from_schedule = False
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
        self.num_operations_executed = 0
        self._previous_num_operations_executed = 0
        self.jobs_completed = 0
        self.env.process(self._generate_jobs())
        # Process any initial events, like t=0 job arrivals, so state is accurate
        self.env.step()

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

        # Compute next observation (self.state is updated by this call)
        obs = self.state_calculator.calculate_state_features(
            self.uncompleted_jobs, self.machines
        )
        self.state = obs

        # Calculate ops for this step
        ops_this_step = self.num_operations_executed - self._previous_num_operations_executed
        self._previous_num_operations_executed = self.num_operations_executed

        # Reward is accumulated from job_process calls during this env.step()
        reward_for_step = self.last_reward
        self.last_reward = 0.0

        # Done if time or jobs exhausted
        # Episode is done if max_time is reached, or if all scheduled jobs have been generated
        # and all of those jobs are now completed.
        done = (self.env.now >= self.max_time) or \
               (self._all_jobs_generated_from_schedule and not self.uncompleted_jobs)

        info = {'jobs_completed': self.jobs_completed,
                'num_ops': ops_this_step, # Operations executed in this specific step
                'last_reward': reward_for_step} # For potential debugging or detailed logging
        return obs, reward_for_step, done, info

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
        # All jobs from the schedule have been generated
        self._all_jobs_generated_from_schedule = True

    def _job_process(self, machine: RealMachine, job: Order):
        # SimPy routine for processing a job
        with machine.queue.request() as req:
            yield req
            # Actual processing
            yield self.env.process(machine.process_job(job))
            self.num_operations_executed += 1
            # Collect reward
            self.last_reward += job.last_task_reward
            # Update job queues
            if job.get_completed():
                self.uncompleted_jobs.remove(job)
                self.jobs_completed += 1
            else:
                self.processable_jobs.append(job)
