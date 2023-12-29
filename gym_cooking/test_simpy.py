import simpy
import random
from state_calculator import StateCalculator
from utils.core import Order
from recipe_planner.recipe import *
import numpy as np
from ddqnscheduler.scheduler import SchedulingAgent as Scheduler
import copy 

from machine import Machine    
from schedulingrules import *

from ddqnscheduler.parameter import *

schedule= [Order(Salad(), (0,7), 0, (30,60)), 
           Order(SimpleTomato(), (1,7), 2, (35,50)),
           Order(Salad(), (1,7), 10, (60,75)),
           Order(SimpleLettuce(), (1,7), 10, (40,55)),
           Order(Salad(), (1,7), 15, (45,75)),
           Order(SimpleTomato(), (1,7), 15, (45,60)),
           Order(SimpleTomato(), (0,7), 20, (50,65)), 
           Order(Salad(), (1,7), 35, (85,100)),
           Order(SimpleLettuce(), (0,7), 60, (90,105)),
           Order(Salad(), (0,7), 70, (120,150)),
           Order(Salad(), (0,7), 100, (160,200)),
           Order(SimpleTomato(), (0,7), 100, (130,180)),
           Order(SimpleLettuce(), (0,7), 120, (200,250)),
           Order(Salad(), (0,7), 160, (300,380)),
           Order(SimpleTomato(), (0,7), 160, (280,340)),
           Order(SimpleLettuce(), (0,7), 160, (270,360)),
           Order(Salad(), (0,7), 500, (650,720)),
           Order(SimpleTomato(), (0,7), 500, (600,680)),
           Order(SimpleLettuce(), (0,7), 500, (620,700)),
           Order(Salad(), (0,7), 630, (750,900)),
           Order(SimpleLettuce(), (0,7), 640, (780,820)),
           Order(SimpleTomato(), (0,7), 650, (800,860)),
           Order(Salad(), (0,7), 750, (1050,1200)),
           Order(SimpleTomato(), (0,7), 800, (1000,1060)),
           Order(SimpleLettuce(), (0,7), 820, (1020,1125)),
]





class JobShop:
    def __init__(self, scheduler: Scheduler):
        # Define the job shop simulation environment
        self.env = simpy.Environment()
        self.processable_jobs= []
        self.state = np.zeros(4)
        self.state_calculator = StateCalculator()
        self.uncompleted_jobs = []
        self.scheduler = scheduler
        self.reschedule_count = 0

        # Create two machines
        machine1 = Machine(self.env, "Agent 1", capacity=1)
        machine2 = Machine(self.env, "Agent 2", capacity=1)
        machine3 = Machine(self.env, "Agent 3", capacity=1)
        machine4 = Machine(self.env, "Agent 4", capacity=1)
        self.machines = [machine1, machine2, machine3, machine4]
        self.schedule= []

        # rewards
        self.rewards = []


    # Define a function to process jobs on machines
    def job_process(self, machine, job, policy):
        next_op = job.get_next_operation()
        with machine.queue.request() as request:
            yield request
            yield self.env.process(machine.process_job(job))
            self.schedule.append((machine.name, next_op))
            if not job.get_completed():
                self.processable_jobs.append(job)

            prev_state = self.state
            self.state, reward = self.state_calculator.calculate_state_features(self.uncompleted_jobs, self.machines)
            self.rewards.append(reward)
            self.scheduler.observation(prev_state,policy ,reward,self.state,self.calculate_done())
            self.reschedule()

    def reschedule(self):
        if len(self.processable_jobs) > 0:
            self.reschedule_count+= 1
            policy = self.scheduler.choose_action(self.state)
            scheduling_rule = scheduling_rules[policy]
            selected_job , selected_machine = scheduling_rule(self.processable_jobs, self.machines)
            self.processable_jobs.remove(selected_job)
            self.env.process(self.job_process(selected_machine, selected_job, policy))

    def generate_jobs(self):
        for order in schedule:
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

if __name__ == "__main__":
    scheduler = Scheduler(5000, 4, 4)
    rewards = []
    
    for i in range(4000): # Training episodes
        # Start the job generation process
        job_shop = JobShop(scheduler= scheduler)
        job_shop.run(1200)
        scheduler.replay()

        if i % 70 == 0:
            print("Target models update")
            scheduler.update_target_model()

        scheduler.policy.reset()
        
        if np.sum(job_shop.rewards) != 0:
            rewards.append((i, np.sum(job_shop.rewards)))
        max_reward = max(rewards, key= lambda x: x[1])

    scheduler.model.save_model("./gym_cooking/models/ddqn/pretrained/test.h5")

