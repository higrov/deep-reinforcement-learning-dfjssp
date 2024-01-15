# Recipe planning
from asyncio.log import logger
import os
import shutil

from recipe_planner.utils import *

from recipe_planner import Recipe, SimpleBun, BunLettuce, Burger, BunLettuceTomato


# Navigation planner
import navigation_planner.utils as nav_utils

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
        # a.subtask = self.subtask
        # a.new_subtask = self.new_subtask
        # a.subtask_agent_names = self.subtask_agent_names
        # a.new_subtask_agent_names = self.new_subtask_agent_names
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

    def select_action(self, t, obs, sim_agent, rnn_hxs=None, masks=None):
        """Return best next action for this agent given observations."""
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if t == 0 and self.model_type not in ["ppo", "seac", "mappo"]:
            self.setup_subtasks(obs)

        agent_idx = int(sim_agent.name[-1]) - 1
        # Select subtask based on Bayesian Delegation or MARL.
        if self.model_type == "mappo":
            self.action = self.planner.predict(obs, t)
            self.action = self.action[agent_idx]
        elif self.model_type == "seac":
            _, self.action, _, _ = self.planner.model.act(obs, rnn_hxs, masks)
        elif self.model_type == "ppo":
            self.action, _ = self.planner.predict(obs)
            self.action = self.action[agent_idx]
        else:
            self.update_subtasks(obs)
            self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(agent_name=self.name)
            self.plan(copy.copy(obs))
            self.action = obs.world.NAV_ACTIONS.index(self.action) if self.action != (0, 0) else -1

        return self.action

    def get_subtasks(self, order, world):
        """Return different subtask permutations for recipes."""
        # self.sw = STRIPSWorld(world)
        # # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        # subtasks = self.sw.get_subtasks(recipe__=order.recipe, max_path_length=self.arglist.max_num_subtasks
        # )
        # all_subtasks = [subtask for path in subtasks for subtask in path]

        # # Uncomment below to view graph for recipe path i
        # # i = 0
        # # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        # return all_subtasks
        return [task for task in order]

    def setup_subtasks(self, obs):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(order=obs.get_remaining_orders()[0], world=obs.world)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, remaining_orders, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        #self.subtask_complete = False
        #if self.subtask is None or len(self.subtask_agent_names) == 0:
        #logger.info(f"{color(self.name, self.color)} has no subtask")
        #return
        
        self.subtask_complete = self.is_subtask_complete(world)
        # print(
        #     f"{color(self.name, self.color)} done with {self.subtask} according to planner: {self.is_subtask_complete(world)}\nplanner has subtask {self.planner.subtask} with subtask object {self.planner.goal_obj}"
        # )

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True

        # logger.info(
        #     f'{color(self.name, self.color)} incomplete subtasks[{len(self.incomplete_subtasks)}]: {", ".join(str(t) for t in self.incomplete_subtasks)}'
        # )

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if (self.subtask is not None and self.subtask not in self.incomplete_subtasks) or self.delegator.should_reset_priors(obs=env, incomplete_subtasks=self.incomplete_subtasks):
            self.reset_subtasks()
            self.delegator.set_priors(obs=env, priors_type=self.priors,
                incomplete_subtasks=self.incomplete_subtasks)
        else:
            if self.subtask is None and any(env.get_remaining_orders()):
                self.delegator.set_priors(
                    obs=env,
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors,
                )
            else:
                self.delegator.bayes_update(
                    obs_tm1=env.obs_tm1,
                    actions_tm1=env.agent_actions,
                    beta=self.beta,
                )
    
    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if len(self.orders) > 0 or any(isinstance(t, Deliver) for t in self.incomplete_subtasks):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        # logger.info(
        #     f"right before planning, {self.name} had old subtask {self.subtask}, new subtask {self.new_subtask}, subtask complete {self.subtask_complete}"
        # )

        # Check whether this subtask is done.
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0 - self.none_action_prob) / (len(actions) - 1))
                    
            if (sum(probs) != 1.0):
                max_index = np.argmax(probs)
                probs[max_index] += 1.0 - sum(probs) # add the difference to the last action
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == "greedy" or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = (
                    self.new_subtask if self.new_subtask is not None else self.subtask
                )
                other_agent_planners = self.delegator.get_other_agent_planners(
                    obs=copy.copy(env), backup_subtask=backup_subtask
                )

            # logger.info(
            #     f"[{self.name} Planning ] Task: {self.new_subtask}, Task Agents: {self.new_subtask_agent_names}"
            # )

            action = self.planner.get_next_action(
                env=env,
                subtask=self.new_subtask,
                subtask_agent_names=self.new_subtask_agent_names,
                other_agent_planners=other_agent_planners,
            )

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = (
                    action[self.new_subtask_agent_names.index(self.name)]
                    if self.planner.is_joint
                    else action
                )

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        #logger.info(f"{self.name} proposed action: {self.action}\n")
        return self.action

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(
            subtask=self.new_subtask
        )
        self.subtask_action_object = nav_utils.get_subtask_action_obj(
            subtask=self.new_subtask
        )

        # Define termination conditions for agent subtask.
        # For Delivery subtask, desired object should be at a Delivery location.
        if isinstance(self.new_subtask, Delivery):
            self.cur_obj_count = len(
                list(
                    filter(
                        lambda o: o
                        in set(
                            env.world.get_all_object_locs(self.subtask_action_object)
                        ),
                        env.world.get_object_locs(obj=self.goal_obj, is_held=False),
                    )
                )
            )
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                len(
                    list(
                        filter(
                            lambda o: o
                            in set(
                                env.world.get_all_object_locs(
                                    obj=self.subtask_action_object
                                )
                            ),
                            w.get_object_locs(obj=self.goal_obj, is_held=False),
                        )
                    )
                )
            )
        # Otherwise, for other subtasks, check based on # of objects.
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = (
                lambda w: len(w.get_all_object_locs(obj=self.goal_obj))
                > self.cur_obj_count
            )


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
        processing_time = 4000

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