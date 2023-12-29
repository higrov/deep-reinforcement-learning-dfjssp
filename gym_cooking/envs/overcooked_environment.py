import copy
import logging
from asyncio.log import logger
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import combinations, product
from time import perf_counter
import logging
import copy
import networkx as nx
import numpy as np

# from recipe_planner.stripsworld import STRIPSWorld
from recipe_planner.recipe import *
import recipe_planner.utils as recipe_utils
import navigation_planner.utils as nav_utils
from utils.interact import ActionRepr, interact
from utils.core import *


import gymnasium as gym
from gymnasium import spaces

from utils.agent import COLORS, SimAgent
from utils.world import World
from misc.game.gameimage import GameImage

from envs.observation_helpers import ObservationHelpers
from envs.reward_helpers import RewardHelpers

import sys

logger = logging.getLogger(__name__)

StateRepr = namedtuple("StateRepr", "time agent_states item_states")

@dataclass
class AgentHistoryRepr:
    time: int
    location: tuple[int, int] 
    action_type: str
    holding: Object
    delivered: bool
    handed_over: bool
    collided: bool 
    shuffled: bool 
    invalid_actor: bool
    location_repeater: bool 
    holding_repeater: bool

# State Features

def calculate_average_utilization_rate(schedule, list_machines):
    machine_util = 0
    num_machines = len(list_machines)

    for machine in list_machines:
        job_util = 0
        for job in schedule:
            task_util = 0 
            completed_operations_machine =[oper for oper in job.get_completed_tasks() if oper[1] == machine]

            for (operation, machine, _) in completed_operations_machine:
                processing_time = machine.get_possible_tasks()[operation]
                task_util += processing_time
        
            job_util+= task_util
        
        machine_util += (job_util/machine.timestamp_last_operation_executed)

    average_utilization_rate = machine_util / num_machines  # Calculate average over tasks and jobs

    return average_utilization_rate

def calculate_estimated_earliness_tardiness_rate(schedule, list_machines):
    Tcur = sum([machine.timestamp_last_operation_executed for machine in list_machines])/len(list_machines)

    NJtard = 0
    NJearly = 0
    
    for job in schedule: 
        if len(job.completed_tasks)< len(job.tasks):
            Tleft = 0
            
            left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]

            for left_task in left_tasks: 
                tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                Tleft += tij
                if Tcur + Tleft > job.due_date:
                    NJtard += 1
                    break
            
            if Tleft + Tcur < job.due_date:
                NJearly+= 1
    

    Ete = (NJearly+NJtard) / len(schedule)

    print("Number of estimated early Jobs: ", NJearly)
    print("Number of estimated Tardy Jobs: ", NJtard)

    return Ete

def calculate_actual_earliness_tardiness_rate(schedule, list_machines):
    NJa_tard = 0
    NJa_early = 0

    for job in schedule:
        if len(job.completed_tasks)< len(job.tasks):
            Tleft = 0
            last_completed_task = job.get_completed_tasks()[-1]
            if last_completed_task[2] > job.due_date:
                NJa_tard += 1
            
            else:
                left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]

                for left_task in left_tasks:
                    tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                    Tleft+= tij
                    if last_completed_task[2]+Tleft> job.due_date:
                        NJa_tard +=1
                        break
                
                if last_completed_task[2] +Tleft < job.due_date:
                    NJa_early += 1
    
    ETa = (NJa_early+NJa_tard)/len(schedule)
    print("Number of actual early Jobs: ", NJa_early)
    print("Number of actual Tardy Jobs: ", NJa_tard)

    return ETa

def actual_penalty_cost(schedule, list_machines): 

    p_num_list = [0]
    p_den_list = [1]

    for job in schedule:
        if len(job.get_completed_tasks()) < len(job.tasks):
            Tleft = 0
            
            left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
            
            for left_task in left_tasks:
                tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                Tleft+= tij
            
            last_completed_task = job.get_completed_tasks()[-1]
            
            if(last_completed_task[2]> job.due_date):
                penalty =  job.earliness_tardiness_weights[1] * (last_completed_task[2]+ Tleft - job.due_date)
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
            
            if(last_completed_task[2] +Tleft < job.due_date):
                penalty = job.earliness_tardiness_weights[0] * (job.due_date - last_completed_task[2] - Tleft)
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
    
    p_total = sum(p_num_list) / sum(p_den_list)

    return p_total

# Scheduling Rules

def action_dispatching_rule1(schedule, list_machines):
    average_machine_completion_time= np.sum([machine.timestamp_last_operation_executed for machine in list_machines]) / len(list_machines)

    urgency_list= [(job,job.due_date - average_machine_completion_time) for job in schedule]

    select_func = lambda x: x[1]

    selected_job = min(urgency_list, key=select_func)

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time)
        temp2 = temp + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp2))
    
    selected_machine= min(machine_appro, key= select_func)

    return(selected_machine,selected_job[0])

def action_dispatching_rule2(uncompleted_jobs, list_machines):
    for job in uncompleted_jobs:
        execution_times : list = []
        next_uncompleted_tasks = [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
        job_execution_time= 0
        for task in next_uncompleted_tasks:
            machine_set = [machine for machine in list_machines if task in machine.possible_tasks.keys()]
            time = np.sum([machine.possible_tasks[task] for machine in machine_set])/len(machine_set)
            job_execution_time += time

        execution_times.append((job, job_execution_time))
        
    select_func = lambda x: x[1]
    selected_job = max(execution_times,key= select_func)


    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time) + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp))
    
    selected_machine= min(machine_appro, key= select_func)

    return (selected_job[0], selected_machine[0])

def action_dispatching_rule3(schedule, list_machines):
    weight_calc =[]
    uncompleted_jobs = [job for job in schedule if not job.isCompleted]
    for job in uncompleted_jobs:
        calc = (0.2* job.earliness_tardiness_weights[0]) + (0.8*job.earliness_tardiness_weights[1])
        weight_calc.append((job,calc))
    
    select_func = lambda x: x[1]

    selected_job = max(weight_calc, key= select_func)

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    suitable_machines = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys()]

    machine_load = []
    for machine in suitable_machines: 
        tasks_performed = []
        for job in uncompleted_jobs: 
            tasks_performed.extend([completed_task for completed_task in job.completed_tasks if machine== completed_task[1]])

        machine_load.append((machine,sum(integer for _,_ , integer in tasks_performed)))

    selected_machine = min(machine_load, key= select_func)

    return(selected_job[0], selected_machine[0])

def action_dispatching_rule4(uncompleted_jobs, list_machines): 
    mean_job_execution_times =[]
    for job in uncompleted_jobs:
        next_uncompleted_tasks = [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
        job_execution_time = 0
        for task in next_uncompleted_tasks:
            suitable_machines = [machine for machine in list_machines if task in machine.possible_tasks.keys()]

            average_execution_time = sum([machine.possible_tasks[task] for machine in suitable_machines])/len(suitable_machines)
            job_execution_time += average_execution_time
        
        mean_job_execution_times.append((job,job_execution_time))
    
    selected_job = min(mean_job_execution_times,key= lambda x: x[1])

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time)
        temp2 = temp + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp2))
    
    selected_machine= min(machine_appro, key= lambda x: x[1])

    return(selected_job[0], selected_machine[0])

class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""
    def __init__(self, arglist, env_id=0, early_termination=True, load_level=True):
        self.arglist = arglist
        self.t_0 = 0
        self.t = 0
        self.env_id = env_id
        self.early_termination = early_termination
        self.levels = "open-divider_salad" 
        self.curr_level = "open-divider_salad"
        self.filename = ''
        self.max_num_agents = 5
        self.max_num_orders = 5
        self.set_filename(arglist=arglist, suffix=f"/{env_id}")
        
        self.recipes: list[Recipe] = []
        self.sim_agents: list[SimAgent] = []
        # For maintaining state
        self.obs_tm1 = None
        self.rl_obs = None
        # For visualizing episode.
        self.rep = []
        # For tracking data during an episode.
        self.agent_history: dict[str, list[AgentHistoryRepr]] = {}

        # stats for info
        self.num_deliveries = 0
        self.num_handovers = 0
        self.num_collisions = 0
        self.num_shuffles = 0
        self.num_invalid_actions = 0
        self.num_location_repeaters = 0
        self.num_holding_repeaters = 0

        # flags for termination
        self.successful = False
        self.failed = False
        self.termination_info = ""
        self.termination_stats = {
            'successful': False,
            'failed': False,
            'episode_length': 0,
            'episode_duration': 0,
            'deliveries': 0,
            'handovers': 0,
            'collisions': 0,
            'shuffles': 0,
            'invalid_actions': 0,
            'location_repeaters': 0,
            'holding_repeaters': 0,
        }
        self.termination_stats.update({f'order_{i+1}_delivery': self.arglist.max_num_timesteps for i in range(self.max_num_orders)})

        # load world and level
        self.game = None
        self.default_world: World = None
        self.world: World = None
        if load_level:
            self.load_level(
                level=self.curr_level,
                num_agents=np.random.randint(1, 5) if all((False, self.arglist.train, self.arglist.randomize)) else self.arglist.num_agents,
                num_orders=self.arglist.num_orders,
                randomize=self.arglist.randomize,
                reload_level=True,
            )

        # Set up action and observation spaces
        # Only possible actions are up, down, left, right + no-op
        self.action_space = spaces.Discrete(5* self.max_num_agents)
        # global observation space = num_agents
        #self.observation_space = self.get_observation_space_structure()  

        observation_space_dict = {}
        for agent in self.sim_agents:        
            observation_space_dict[agent.name] = spaces.Discrete(4)
             
        self.observation_space = spaces.Dict(observation_space_dict)
        # number of RL agents being trained (for env wrappers)
        self.n_agents = 4

        # Recipe Planning stuff for BD
        self.any_bayesian = len(self.sim_agents) > self.n_agents
        self.all_subtasks = []

        # default empty obs, empty reward, empty done
        self.agent_padding = [None] * self.max_num_agents
        self.done_padding = [True] * self.max_num_agents
        self.reward_padding = [0.0] * self.max_num_agents
        self.order_padding = [None] * self.max_num_orders

    def get_repr(self):
        return self.world.get_repr() + tuple(agent.get_repr() for agent in self.sim_agents)

    def get_objects_repr(self, flat=False):
        return self.world.get_repr() if not flat else self.world.get_dynamic_objects_flat() 

    def get_agents_repr(self, fixed=False):
        return tuple([agent.get_repr(fixed) for agent in self.sim_agents])

    def get_history_repr(self, agent, action_type="Wait", delivered=False, handed_over=False, collided=False, shuffled=False, invalid_actor=False, location_repeater=False, holding_repeater=False):
        return AgentHistoryRepr(self.t, agent.location, action_type, agent.holding, delivered, handed_over, collided, shuffled, invalid_actor, location_repeater, holding_repeater)

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: "".join(map(lambda y: y + " ", x)), self.rep))
        return "\n".join(_display)

    def __eq__(self, other):
        return (self.get_repr() == other.get_repr()) if other is not None else self is None

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist, env_id=self.env_id, early_termination=self.early_termination, load_level=False)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.default_world = self.default_world
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances
        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                    location=a.location, desired_obj=None, find_held_objects=True
                )
        # print(sys.getsizeof(new_env))
        # print(sys.getsizeof(new_env.world))
        # print(sys.getsizeof(new_env.sim_agents))
        return new_env

    def set_filename(self, arglist, suffix=""):
        self.filename = (
            f"{self.curr_level}/agents-{arglist.num_agents}/orders-{arglist.num_orders}/"
        )
        model = ""
        if arglist.model1 is not None:
            model += f"model1-{arglist.model1}"
        if arglist.model2 is not None:
            model += f"_model2-{arglist.model2}"
        if arglist.model3 is not None:
            model += f"_model3-{arglist.model3}"
        if arglist.model4 is not None:
            model += f"_model4-{arglist.model4}"
        if arglist.model5 is not None:
            model += f"_model5-{arglist.model5}"

        self.filename += model + suffix

    def load_level(self, level, num_agents, num_orders, randomize, reload_level):
        if self.default_world is not None and not reload_level:
            self.world = copy.copy(self.default_world)
            # if randomize:
            #     self.randomize_world()
            self.distances = {}
            
        else:
            self.default_world = World()
            x = 0
            y = 0
            with open(f"utils/levels/{level}.txt", "r") as file:
                # Mark the phases of reading.
                phase = 1
                for line in file:
                    line = line.strip("\n")
                    if line == "":
                        phase += 1

                    # Phase 1: Read in kitchen map.
                    elif phase == 1:
                        for x, rep in enumerate(line):
                            # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                            if rep in "tlbopm":
                                counter = Counter(location=(x, y))
                                obj = Object(location=(x, y), contents=RepToClass[rep]())
                                counter.acquire(obj=obj)
                                self.default_world.insert(obj=counter)
                                self.default_world.insert(obj=obj, toDefault=True)
                            # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                            elif rep in RepToClass:
                                newobj = RepToClass[rep]((x, y))
                                self.default_world.objects.setdefault(newobj.name, []).append(
                                    newobj
                                )
                            else:
                                # Empty. Set a Floor tile.
                                f = Floor(location=(x, y))
                                self.default_world.objects.setdefault("Floor", []).append(f)
                        y += 1
                    # Phase 2: Read in recipe list.
                    elif phase == 2:
                        self.recipes.append(globals()[line]())

                    # Phase 3: Read in agent locations (up to num_agents).
                    elif phase == 3:
                        if len(self.sim_agents) < num_agents:
                            loc = line.split(" ")
                            sim_agent = SimAgent(
                                name=f"agent-{len(self.sim_agents)+1}",
                                id_color=COLORS[len(self.sim_agents)],
                                location=(int(loc[0]), int(loc[1])),
                            )
                            self.sim_agents.append(sim_agent)

            # generate order queue from recipe list for level
            self.default_world.objects.setdefault("Order", [])
            delivery_window = (0,10)
            for i in range(num_orders):  # append orders for level
                random_recipe = self.recipes[np.random.randint(0, len(self.recipes))]
                location = len(self.default_world.objects.get("Order")), y 
                nextOrder = RepToClass[Rep.ORDER](random_recipe, location, self.t, delivery_window)
                self.default_world.objects.get("Order").append(nextOrder)

            self.distances = {}
            self.default_world.width = x + 1
            self.default_world.height = y + 1  # + 1 for the orders queue
            self.default_world.perimeter = 2 * (self.default_world.width + self.default_world.height)

            self.world = copy.copy(self.default_world)
            # get all orders not just incomplete ones
            self.orders: tuple[Order] = tuple(self.world.objects.get("Order")) 
        
            # if randomize:
            #     self.randomize_world()

    def randomize_world(self, randomize_agents=True, randomize_objects=True, randomize_stations=False):
        # level_types = ('open-divider', 'partial-divider', 'full-divider', 'cross-divider', 'block-divider', 'ring-divider')
        random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
        if randomize_agents:
            # if only one agent don't spawn in center of ring spawned agent will be locked
            if 'ring' in self.curr_level and len(self.sim_agents) == 1:
                while (3, 3) in [f.location for f in random_floors]:
                    #print('ring', [f.location for f in random_floors])
                    random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
            # if full divider check that there is at least 1 agent on both sides
            elif 'full' in self.curr_level:
                while len(self.sim_agents) > 1 and all(f.location[0] < 3 for f in random_floors) or all(f.location[0] > 3 for f in random_floors):
                    #print('full', [f.location for f in random_floors])
                    random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
            # if cross-divider check that at most 2 agents in a block
            elif 'cross' in self.curr_level:
                blockTL = [f for f in random_floors if f.location[0] < 3 and f.location[1] < 3]
                blockTR = [f for f in random_floors if f.location[0] < 3 and f.location[1] > 3]
                blockBL = [f for f in random_floors if f.location[0] > 3 and f.location[1] < 3]
                blockBR = [f for f in random_floors if f.location[0] > 3 and f.location[1] > 3]
                while not all(len(block) < 3 for block in [blockTL, blockTR, blockBL, blockBR]):
                    #print('cross', [f.location for f in random_floors])
                    random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
                    blockTL = [f for f in random_floors if f.location[0] < 3 and f.location[1] < 3]
                    blockTR = [f for f in random_floors if f.location[0] < 3 and f.location[1] > 3]
                    blockBL = [f for f in random_floors if f.location[0] > 3 and f.location[1] < 3]
                    blockBR = [f for f in random_floors if f.location[0] > 3 and f.location[1] > 3]

            for agent, floor in zip(self.sim_agents, random_floors):
                agent.location = floor.location

        # important to do both together since the dynamic objects are also placed on counters 
        if randomize_stations or randomize_objects:
            stations = self.world.get_object_list(['Cutboard', 'Delivery'])
            ingredients = self.world.get_dynamic_object_list()

            non_corner_counters = [_ for _ in self.world.get_object_list(['Counter']) if self.world.is_accessible(_.location)]
            random_counters: list[Counter] = np.random.choice(non_corner_counters, len(stations) + len(ingredients))

            if randomize_stations:
                for station, counter in zip(stations, random_counters[:len(stations)]):
                    station.location, counter.location = counter.location, station.location
                    counter.update_holding_location()

            if randomize_objects:
                for ingredient, counter in zip(ingredients, random_counters[len(stations):]):
                    old_counter: Counter = self.world.get_gridsquare_at(ingredient.location)
                    old_counter.swap_holding(counter)


    def reset(self, reload_level=False):
        self.t = 0
        self.t_0 = 0
        for a in self.sim_agents:
            a.reset()
        for o in self.orders:
            o.reset(self.t, delivery_window=(0,10))
        self.agent_actions = {}
        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.num_deliveries = 0
        self.num_handovers = 0
        self.num_collisions = 0
        self.num_shuffles = 0
        self.num_invalid_actions = 0
        self.num_location_repeaters = 0
        self.num_holding_repeaters = 0

        self.successful = False
        self.failed = False
        self.termination_info = ""
        self.termination_stats = {k: 0 if 'delivery' not in k else self.arglist.max_num_timesteps for k in self.termination_stats.keys()}
        
        # load world and level
        self.world: World = None
        self.curr_level = random.choice(self.levels).strip() if reload_level else self.curr_level
        self.load_level(
            level=self.curr_level,
            num_agents=np.random.randint(1, 5) if all((False, self.arglist.train, self.arglist.randomize)) else self.arglist.num_agents,
            num_orders=self.arglist.num_orders,
            randomize=self.arglist.randomize,
            reload_level=reload_level,
        )

        # Load distances
        self.all_subtasks = self.run_recipes() if self.any_bayesian else []
        self.world.make_loc_to_gridsquare()
        if self.any_bayesian:
            self.world.make_reachability_graph()
            self.cache_distances()

        # for visualization and screenshots
        self.game = GameImage(
            filename=self.filename,
            env_id=self.env_id,
            world=self.world,
            sim_agents=self.sim_agents,
            record=self.arglist.record,
        )
        self.game.on_init()

        # obs
        self.obs_tm1 = None
        self.obs_tm1 = copy.copy(self)  # obs for BD
        # to track agent activity
        self.agent_history = {agent.name: deque([self.get_history_repr(agent)], maxlen=11) for agent in self.sim_agents}
        #self.rl_obs = self.get_rl_obs() if self.n_agents > 0 else None  # obs for RL

        if self.arglist.record:
            self.game.save_image_obs(self.t)

        return self

    def close(self):
        if self.game:
            self.game.on_cleanup()
        return
    
    def step(self, action):
        # Track internal environment info.
        if self.t == 0:
            self.t_0 = perf_counter()
        
        self.t += 1
        self.orders = tuple(self.world.objects.get("Order")) 

        # Parse action
        # for sim_agent in self.sim_agents:
        #     if sim_agent.name in action:
        #         action_idx = action[sim_agent.name]
        #         if 0 <= action_idx < len(self.world.NAV_ACTIONS):
        #             sim_agent.action = self.world.NAV_ACTIONS[action_idx]
        #         else:
        #             sim_agent.action = (0, 0) # BD has 50% chance of no-op, MAPPO 20% chance
        #     self.agent_history[sim_agent.name].append(self.get_history_repr(sim_agent))

        agent = next(agent for agent in self.sim_agents if agent.name in action)

        agent.action = action[agent.name]
        # set current state as previous
        self.obs_tm1 = None
        self.obs_tm1 = copy.copy(self)

        # Check collisions.
        #self.check_collisions() # stores collisions in self.collisions
        # Perform interaction
        interaction: ActionRepr = interact(agent=agent, world=self.world, t=self.t, play=self.arglist.play)
        #self.execute_navigation() # append to agent activity
        # Compute stats based on agent activity
        # if self.is_env_prime():
        #     self.display()
        #     self.print_agents()

        
        self.compute_stats()
        # Count shuffles, handovers, deliveries, invalid actions, location repeaters, holding repeaters
        if self.arglist.evaluate:
            self.record_stats() # calculated based on agent activity

        #self.render()
        
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        #new_obs = copy.copy(self)
        #new_obs.rl_obs = self.get_rl_obs() if self.n_agents > 0 else None  # NEW rl obs
        #new_obs.state = self.get_state()  # NEW state
        # remove redundant variables
        #new_obs.obs_tm1 = None
        #new_obs.game = None
        #new_obs.world = None

        done = False # CENTRALIZED DONE
        reward = 100 # CENTRALIZED REWARD
            
        info = {
            "t": self.t,
            "rep": self.get_repr(),
            "collisions": self.num_collisions,
            "shuffles": self.num_shuffles,
            "handovers": self.num_handovers,
            "deliveries": self.num_deliveries,
            "invalid_actions": self.num_invalid_actions,
            "location_repeaters": self.num_location_repeaters,
            "holding_repeaters": self.num_holding_repeaters,
            "done": False,
            "reward": 100,
            "termination_info": self.termination_info,
        }

        obs = {agent.name: agent.action for agent in self.sim_agents }
        
        return obs, reward, done, info

    def get_state(self):
        return StateRepr(self.t, self.get_agents_repr(fixed=True), self.get_objects_repr(True))
        
    def compute_stats(self, handover_lookback=5, shuffle_lookback=4, location_repeater_lookback=10, location_repeater_threshold=3, holding_repeater_lookback=10, holding_repeater_threshold=3):
        for agent in self.sim_agents:    
            self_history = self.agent_history[agent.name]
            curr = self.agent_history[agent.name][-1]
            slice = list(self_history)[-10:]
            # deliveries -- if someone took a deliver action this time step
            self.num_deliveries += curr.delivered
            
            # handovers -- if agent A holding an item previously held by agent B (within lookback) 
            if self.t > handover_lookback:
                others_holding_history = {other: [None if _.holding is None else _.holding.spawn_location for _ in list(history)[-handover_lookback:-1]] for other, history in self.agent_history.items() if other != agent.name}
                spawns = { loc for v in others_holding_history.values() for loc in v }
                curr.handed_over = curr.holding is not None and (curr.holding.spawn_location in spawns)
                self.num_handovers += curr.handed_over
                
            # collisions -- if someone collided this time step
            self.num_collisions += curr.collided

            # shuffles -- if an agent has moved to the same location while not having more than one item within the lookback window + 1 for None item
            if self.t > shuffle_lookback:
                curr.shuffled = curr.location == self_history[-shuffle_lookback].location and len(set(_.holding.get_repr() if _.holding else None for _ in slice[-shuffle_lookback:])) > 2
                self.num_shuffles += curr.shuffled

            # invalid actions -- if an agent has collided with a counter or not performed an action this step
            self.num_invalid_actions += curr.invalid_actor

            # location repeaters -- if an agent has too few new locations within the lookback window
            if self.t > location_repeater_lookback:
                curr.location_repeater = len(set(_.location for _ in slice[-location_repeater_lookback:])) < location_repeater_threshold
                self.num_location_repeaters += curr.location_repeater

            # holding repeaters -- if an agent has too few new items within the lookback window
            if self.t > holding_repeater_lookback:
                curr.holding_repeater = len(set(_.holding.get_repr() if _.holding else None for _ in slice[-holding_repeater_lookback:])) < holding_repeater_threshold
                self.num_holding_repeaters += curr.holding_repeater


    def record_stats(self):
        self.termination_stats['deliveries'] = self.num_deliveries
        self.termination_stats['handovers'] = self.num_handovers
        self.termination_stats['collisions'] = self.num_collisions
        self.termination_stats['shuffles'] = self.num_shuffles
        self.termination_stats['invalid_actions'] = self.num_invalid_actions
        self.termination_stats['location_repeaters'] = self.num_location_repeaters
        self.termination_stats['holding_repeaters'] = self.num_holding_repeaters
        for i, order in enumerate(self.orders):
            delivered_on = self.termination_stats[f'order_{i+1}_delivery']
            self.termination_stats[f'order_{i+1}_delivery'] = self.t if (order.delivered and self.t <= delivered_on) else delivered_on

    def render(self, mode="human"):
        logger.info(f"""\n=======================================\n[environment-{self.env_id}.step] @ TIMESTEP {self.t}\n=======================================""")
        self.display()
        self.print_agents()

    
    def is_env_prime(self):
        return self.env_id == 0 or 'eval' in self.env_id or ('_' in self.env_id and self.env_id.split('_')[-1] in ['0', 'eval'])

    def generate_animation(self, t, suffix=""):
        if self.env_id == 0 or ('_' in self.env_id and self.env_id.split('_')[-1] == '0'):
            return self.game.generate_animation(suffix)
        return None

    def get_animation_path(self):
        if self.is_env_prime():
            return self.game.get_animation_path()
        return ""

    def done(self):
        # Done if the episode maxes out (or queue empty) or no state change in past 5 actions
        # if self.t >= self.arglist.max_num_timesteps:
        #     self.successful = not any(self.get_remaining_orders())
        #     self.failed = not self.successful
        #     if self.arglist.record and not self.arglist.train:
        #         self.generate_animation(self.t)
        #     self.termination_info = self.get_termination_info(reason=f"Terminating because passed {self.arglist.max_num_timesteps} timesteps")

        # elif self.early_termination and any([len(self.get_remaining_orders()) == 3 and self.t >= 50, len(self.get_remaining_orders()) == 2 and self.t >= 100, len(self.get_remaining_orders()) == 1 and self.t >= 150]):
        #     self.successful = False
        #     self.failed = True
        #     if self.arglist.record and not self.arglist.train:
        #         self.generate_animation(self.t)
        #     self.termination_info = self.get_termination_info(reason=f"Terminating because of early return condition: Not enough orders delivered in past {self.t} timesteps")
        
        # elif all([order.delivered for order in self.orders]):
        #     self.successful = True
        #     self.failed = False
        #     if self.arglist.record and not self.arglist.train:
        #         self.generate_animation(self.t)
        #     self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")

        # elif self.any_bayesian:
        #     assert any([isinstance(subtask, recipe_utils.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        #     # Done if subtask is completed.
            
        #     if any(isinstance(subtask, recipe_utils.Deliver) or subtask.name == 'Deliver'for subtask in self.all_subtasks):
        #         # Double check all goal_objs are at Delivery.
        #         self.successful = True
                
        #         for subtask in self.all_subtasks:
        #             _, goal_obj = nav_utils.get_subtask_obj(subtask)

        #             delivery_loc = list(filter(lambda o: o.name == "Delivery", self.world.get_object_list()))[0].location
        #             goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
        #             if not any([gol == delivery_loc for gol in goal_obj_locs]):
        #                 self.successful = False
        #                 self.failed = False

        #         #self.successful = self.successful
        #         self.failed = False
        #         if self.arglist.record and not self.arglist.train:
        #             self.generate_animation(self.t)
        #         self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")
                
        #     else:
        #         self.successful = True
        #         self.failed = False
        #         if self.arglist.record and not self.arglist.train:
        #             self.generate_animation(self.t)
        #         self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")

        # done = (([self.successful or self.failed] * len(self.sim_agents)) + self.done_padding)[:self.max_num_agents]

        self.termination_stats['episode_length'] = self.t
        self.termination_stats['episode_duration'] = perf_counter() - self.t_0
        self.termination_stats['successful'] = self.successful
        self.termination_stats['failed'] = self.failed

        return False
    
    def get_termination_info(self, reason):
        #logger.info(f"{reason}")
        return (reason,
                f"Orders Delivered: {self.termination_stats['deliveries']}",
                f"Handovers: {self.termination_stats['handovers']}",
                f"Collisions: {self.termination_stats['collisions']}",
                f"Shuffles: {self.termination_stats['collisions']}",
                f"Invalid Actions: {self.termination_stats['invalid_actions']}",
                f"Repeated Locations: {self.termination_stats['location_repeaters']}",
                f"Repeated Holdings: {self.termination_stats['holding_repeaters']}",
                f"Successful: {self.successful}",
                f"Failed: {self.failed}",
                f"Recording: {self.game.get_animation_path()}"
        )

        
    def reward(self):
        # successful = self.successful
        # failed = self.failed
        # timestep = self.t
        # agents = self.sim_agents
        # curr = {a: _[-1] for a, _ in self.agent_history.items()}
        # prev = {a: _[-2] for a, _ in self.agent_history.items()}
        # prev_prev = {a: _[-3] for a, _ in self.agent_history.items()} if self.t > 2 else prev
        
        # stations = [o for o in self.world.get_object_list() if o.name in ['Cutboard', 'CuttingBoard', 'Stove', 'Grill', 'Delivery']]
        # dynamic_objects = self.world.get_dynamic_object_list()

        # orders = self.orders
        
        #return (RewardHelpers.compute_rewards(successful, failed, timestep, agents, stations, dynamic_objects, orders, curr, prev, prev_prev) + self.reward_padding)[:self.max_num_agents]
        return 100
    
    def get_observation_space_structure(self):
        max_num_timesteps = 200
        max_width, max_height = 8, 7 #self.world.width, self.world.height - 1 # -1 to account for the order queue
        max_num_agents = self.max_num_agents # len(self.sim_agents) # always train for max_num_agents so that we can train on any number of agents
        # orders and recipe details for this layout
        max_num_orders = self.max_num_orders # len(self.orders) #always train for max_num_orders so that we can train on any number of orders
        # all the holdable_objects in the world -- plate, food or dishes
        max_num_objects = 5 # len(self.world.get_dynamic_objects())
        # tiles of interest in the world -- prep stations, delivery stations
        max_num_prep_stations = 2 # len(self.world.objects['Cutboard'])
        max_num_delivery_stations = 2 # len(self.world.objects['Delivery'])
        # get spaces.Dict representation of the the world
        dict_structure = ObservationHelpers.get_observation_space_structure(max_num_timesteps, max_width, max_height, max_num_agents, max_num_orders, max_num_objects, max_num_prep_stations, max_num_delivery_stations)
        return dict_structure

    def get_image_observation_space_structure(self):
        image_dims = self.game.get_image_obs().shape if self.game is not None else (240, 200, 3)
        return spaces.Box(low=0, high=255, shape=image_dims, dtype=np.uint8)
    
    def get_rl_obs(self):
        # get spaces.Dict with value based on the state of the world
        max_agents = (self.sim_agents + self.agent_padding)[:self.max_num_agents]
        num_agents = len(self.sim_agents)
        max_orders = ([_ for _ in self.orders if not _.delivered] + self.order_padding)[:self.max_num_orders]
        num_orders = len(self.orders)
        layout = self.curr_level.split('_')[0]
        
        obs = ObservationHelpers.get_rl_obs(layout, self.t, max_agents, num_agents, self.world.get_dynamic_object_list(), self.world.objects['Cutboard'], self.world.objects['Delivery'], max_orders, num_orders)
        
        return obs
        
    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        logger.info(f'\n{str(self)}\n')

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)

    def get_agent_names(self):
        return tuple(agent.name for agent in self.sim_agents)

    def get_remaining_orders(self):
        return tuple(order for order in self.orders if not order.delivered)

    def get_delivered_orders(self):
        return tuple(order for order in self.orders if order.delivered)

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        # self.sw = STRIPSWorld(world=self.world)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        # subtasks = self.sw.get_subtasks(recipe__=self.get_remaining_orders()[0].recipe, max_path_length=self.arglist.max_num_subtasks)
        # all_subtasks = [subtask for path in subtasks for subtask in path]
        #print("Subtasks:", all_subtasks, "\n")
        # return all_subtasks
        pass

    def get_AB_locs_given_objs(
        self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj
    ):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe_utils.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: a.name in subtask_agent_names
                            and a.holding == start_obj,
                            self.sim_agents,
                        )
                    ),
                )
            )

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe_utils.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: a.name in subtask_agent_names
                            and a.holding == start_obj,
                            self.sim_agents,
                        )
                    ),
                )
            )
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe_utils.Merge):
            A_locs = self.world.get_object_locs(obj=start_obj[0], is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: a.name in subtask_agent_names
                            and a.holding == start_obj[0],
                            self.sim_agents,
                        )
                    ),
                )
            )
            B_locs = self.world.get_object_locs(obj=start_obj[1], is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: a.name in subtask_agent_names
                            and a.holding == start_obj[1],
                            self.sim_agents,
                        )
                    ),
                )
            )

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe_utils.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)
        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask_name=subtask.name,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty
           
    def execute_navigation(self):
        for agent in self.sim_agents:
            interaction: ActionRepr = interact(agent=agent, world=self.world, t=self.t, play=self.arglist.play)
            # add to agent history
            # ah = self.agent_history[agent.name][-1]
            # ah.action_type =interaction.action_type
            # ah.delivered = interaction.action_type == "Deliver"
            # ah.invalid_actor = (interaction.action_type == "Wait" and agent.action != (0, 0) and not ah.collided)
            # ah.holding = agent.holding if agent.holding is not None else None
            # ah.location = agent.location
            # self.agent_actions[agent.name] = agent.action

    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [
            name
            for name in self.world.objects
            if "Supply" in name
            or "Counter" in name
            or "Delivery" in name
            or "Cut" in name
        ]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = (
                    [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                )
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(
                            self.world.reachability_graph,
                            (source.location, source_edge),
                            (destination.location, dest_edge),
                        )
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)