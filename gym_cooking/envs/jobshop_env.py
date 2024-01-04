import simpy
from state_calculator import StateCalculator
from utils.core import Order
from recipe_planner.recipe import *
import numpy as np
from ddqnscheduler.scheduler import SchedulingAgent as Scheduler
import copy 

from machine import Machine  
from utils.agent import RealMachine, COLORS
from schedulingrules import *
import pandas as pd


class JobShop:
    def __init__(self, scheduler: Scheduler, num_machines: int, globalSchedule):
        # Define the job shop simulation environment
        self.env = simpy.Environment()
        self.processable_jobs= []
        self.state = np.zeros(4) # Initial Production State [0,0,0,0]
        self.state_calculator = StateCalculator()
        self.uncompleted_jobs = []
        self.scheduler = scheduler
        self.machines = []
        self.schedule= pd.DataFrame(columns = ['Time', 'Machine', 'Task', 'Points'])

        self.globalschedule = globalSchedule

        # Create num machines
        for i in range(num_machines):
            newMachine = RealMachine(
                               jobshop_env=self.env,
                               name= 'agent-'+str(len(self.machines)+1),
                               capacity= 1,
                               id_color=COLORS[len(self.machines)]
                             )
            self.machines.append(newMachine)
        
        # Rewards
        self.rewards = []


    # Define a function to process jobs on machines
    def job_process(self, machine: Machine, job: Order, policy):
        next_op= job.get_next_operation()
        with machine.queue.request() as request:
            yield request
            yield self.env.process(machine.process_job(job))
            if not job.get_completed():
                self.processable_jobs.append(job)

            prev_state = self.state
            self.state = self.state_calculator.calculate_state_features(self.uncompleted_jobs, self.machines)
            reward = job.last_task_reward
            self.rewards.append(reward)
            self.schedule = pd.concat([self.schedule,  pd.DataFrame([[job.last_task_completion_timestamp, machine.name, str(next_op), reward]], columns = self.schedule.columns)], axis=0, ignore_index=True)
            self.scheduler.observation(prev_state,policy ,reward,self.state,self.calculate_done())
            self.reschedule()
    
    # Rescheduling event
    def reschedule(self):
        if len(self.processable_jobs) > 0:
            policy = self.scheduler.choose_action(self.state)
            scheduling_rule = scheduling_rules[policy]
            selected_job , selected_machine = scheduling_rule(self.processable_jobs, self.machines)
            self.processable_jobs.remove(selected_job)
            self.env.process(self.job_process(selected_machine, selected_job, policy))

    # Adding activated Jobs to open Jobs
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
        if self.env.now == 1200:
            return True
        return False

    def run(self, until):
        self.env.process(self.generate_jobs())
        # Run the simulation for a defined period
        self.env.run(until = until)