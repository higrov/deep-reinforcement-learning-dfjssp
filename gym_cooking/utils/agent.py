# Recipe planning
from asyncio.log import logger
import os
import shutil

from recipe_planner.utils import *

from recipe_planner import Recipe, SimpleBun, BunLettuce, Burger, BunLettuceTomato


# Navigation planner


# Other core modules
from utils.core import Delivery, Object
from utils.utils import agent_settings, timeit

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

import simpy

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ["purple", "green", "blue", "yellow", "magenta"]
# Possible actions_performed by Agents
class RealMachine:
    """Real Agent object that performs task inference and plans."""
    possible_operations = {Get: 11, Merge: 50, Chop: 40, Grill:30, Deliver: 30}
    def __init__(
        self,
        name,
        id_color,
        jobshop_env,
        capacity
    ):

        self.name = name
        self.color = id_color
        self.holding: Object = None

        # JobShop Machine 

        self.jobshop_env = jobshop_env
        self.capacity = capacity
        self.queue = simpy.Resource(self.jobshop_env, capacity=self.capacity)

        self.last_operation_executed = None
        self.last_operation_executed_at = -1


    def __str__(self):
        return color(self.name[-1], 'red' if 'purple' == self.color else self.color)

    def __copy__(self):
        a = RealMachine(
            arglist=self.arglist,
            name=self.name,
            id_color=self.color,
            capacity=self.capacity,
            jobshop_env=self.jobshop_env

        )

        a.possible_operations = self.possible_operations 

        a.last_operation_executed = None
        a.last_operation_executed_at = None

        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return "None"
        return self.holding.full_name

    def get_holding_object(self):
        return self.holding

    def process_job(self, job):
        operation = job.get_next_operation()
        start_time = self.jobshop_env.now
        print(f"{start_time:.2f}: Job {job.full_name}, operation {str(operation)} started on {self.name}")
        processing_time = self.get_processing_time(operation)
        completed_time = start_time + processing_time
        yield self.jobshop_env.timeout(processing_time)  # Simulate processing time
        print(f"{completed_time:.2f}: Job {job}, operation {str(operation)} completed on {self.name}")
        self.last_operation_executed = operation
        self.last_operation_executed_at = completed_time
        job.add_completed_tasks(operation,self.name,completed_time)

    def get_possible_operations(self): 
        return self.possible_operations
    
    def set_last_operation_executed(self,val): self.last_operation_executed = val

    def set_last_operation_performed_at(self, val): self.last_operation_executed_at = val

    def get_processing_time(self, action): 
        processing_time = 24000

        if(action.__class__ == Get):
            if(action.args[0] == 'Plate'):
                processing_time += 1100
            if action.args[0] == 'Tomato' or action.args[0] == 'Meat' or action.args[0] == 'Lettuce':
                processing_time += 1538
            if(action.args[0] == 'Bun'):
                processing_time += 4000


        elif(action.__class__ == Chop):
            processing_time += 1538
            if action.args[0] == 'Tomato':
                 processing_time += 5000

        elif (action.__class__ == Merge):
            processing_time += 2656
            if action.args[0] == 'Tomato' or action.args[0] == 'Meat' or action.args[0] == 'Lettuce':
                processing_time += 17308
            if(action.args[0] == 'Bun'):
                processing_time += 17500
        
        elif (action.__class__ == Grill):
            processing_time += 10000
        
        elif (action.__class__ == Deliver):
            processing_time += 4746
            
        return processing_time


class SimAgent:
    """Simulation agent used in the environment object."""
    possible_operations = {Get: 5, Merge: 2, Chop: 1, Deliver: 5}
    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.spawn_location = location
        self.holding = None
        self.action = None

        self.last_action_performed = None
        self.last_action_performed_at = None

    def reset(self):
        self.location = self.spawn_location
        self.action = None
        self.possible_tasks = self.possible_operations
        if self.holding:
            self.holding.is_held = False
            self.holding = None

    def __str__(self):
        return color(self.name[-1], 'red' if 'purple' == self.color else self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color, location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self, fixed=False):
        return AgentRepr(
            name=self.name, location=self.location, holding=self.get_holding(fixed)
        )

    def get_holding(self, fixed=False):
        if self.holding is None:
            return "None" if not fixed else None
        return self.holding.full_name

    def print_status(self):
        logger.info(
            f"{color(self.name, 'red' if 'purple' == self.color else self.color)} currently at {self.location}, action {self.action}, holding {self.get_holding()}"
        )

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj)  # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
    
    def get_possible_tasks(self): 
        return self.possible_tasks
    
    def set_last_operation_executed(self,val: str): self.last_operation_executed = val

    def set_last_action_performed_at(self, val: int): self.last_action_performed_at = val

    def get_processing_time(self, action): 
        processing_time = 0
        #return processing_time
        if(action.__class__ == Get):
            if(action.args[0] == 'Bun'):
                processing_time += 17660


        elif(action.__class__ == Chop):
            processing_time += 4194
            if action.args[0] == 'Tomato':
                processing_time += 10000
            if action.args[0] == 'Lettuce':
                processing_time += 5000

        elif (action.__class__ == Merge):
            processing_time = 17308
            if action.args[0] == 'Tomato':
                processing_time += 17308
            if action.args[0] == 'Meat':
                processing_time += 17308
            if(action.args[0] == 'Bun'):
                processing_time += 35348
        
        elif (action.__class__ == Grill):
                processing_time += 15000
        
        elif (action.__class__ == Deliver):
            processing_time = 10
            if(action.args[0] == "Bun-Plate" ):
                processing_time = 10
            if(action.args[0] == "Bun-Lettuce-Plate" ):
                processing_time = 10
            if(action.args[0]== 'Bun-Lettuce-Plate-Tomato' ):
                processing_time += 50
            
            if(action.args[0] == 'Bun-Lettuce-Meat-Plate-Tomato' ):
                processing_time= 100
        return processing_time