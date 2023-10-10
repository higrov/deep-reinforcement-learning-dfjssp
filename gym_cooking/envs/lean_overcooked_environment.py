# Recipe planning
from utils.utils import timeit
# from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *
from time import perf_counter
from asyncio.log import logger
import logging

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import ActionRepr, interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple, deque

import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

from envs.overcooked_environment import AgentHistoryRepr


logger = logging.getLogger(__name__)

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class LeanOvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist, env_id=0, load_level=True):
        self.arglist = arglist
        self.t_0 = 0
        self.t = 0
        self.env_id = env_id
        self.set_filename()
        
        # For visualizing episode.
        self.rep = []
        self.recipes: list[Recipe] = []
        self.sim_agents = []
        self.levels = arglist.level
        self.max_num_agents = 5
        self.max_num_orders = 5
        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.successful = False
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
        self.obs_tm1 = None
        self.default_world = None
        self.world = None
        if load_level:
            self.load_level(
                level=arglist.level,
                num_agents=arglist.num_agents,
                num_orders=arglist.num_orders,
                randomize=arglist.randomize,
                reload_level=True)
        

    def get_history_repr(self, agent, action_type="Wait", delivered=False, handed_over=False, collided=False, shuffled=False, invalid_actor=False, location_repeater=False, holding_repeater=False):
        return AgentHistoryRepr(self.t, agent.location, action_type, agent.holding, delivered, handed_over, collided, shuffled, invalid_actor, location_repeater, holding_repeater)        

    def get_repr(self):
        agent_states = tuple(agent.get_repr() for agent in self.sim_agents)
        object_states = tuple(obj.get_repr() for obj in self.world.get_dynamic_object_list())
        return agent_states + object_states

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __hash__(self):
        return hash(self.get_repr())

    def __copy__(self):
        new_env = LeanOvercookedEnvironment(self.arglist, env_id=self.env_id, load_level=False)
        new_env.__dict__ = self.__dict__.copy()
        new_env.default_world = self.default_world
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

    def load_level(self, level, num_agents, num_orders, randomize, reload_level):
        if self.default_world is not None and not reload_level:
            self.world = copy.copy(self.default_world)
            if randomize:
                self.randomize_world(randomize_agents=True, randomize_objects=True, randomize_stations=False)
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
                            if rep in "tlop":
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
            for i in range(num_orders):  # append orders for level
                random_recipe = self.recipes[np.random.randint(0, len(self.recipes))]
                location = len(self.default_world.objects.get("Order")), y 
                nextOrder = RepToClass[Rep.ORDER](random_recipe, location, self.t)
                self.default_world.objects.get("Order").append(nextOrder)

            self.distances = {}
            self.default_world.width = x + 1
            self.default_world.height = y + 1  # + 1 for the orders queue
            self.default_world.perimeter = 2 * (self.default_world.width + (self.default_world.height - 1))

            self.world = copy.copy(self.default_world)
            # get all orders not just incomplete ones
            self.orders: tuple[Order] = tuple(self.world.objects.get("Order")) 
        
            if randomize:
                self.randomize_world(randomize_agents=True, randomize_objects=True, randomize_stations=False)

    def randomize_world(self, randomize_agents=True, randomize_objects=True, randomize_stations=False):
        # level_types = ('open-divider', 'partial-divider', 'full-divider', 'cross-divider', 'block-divider', 'ring-divider')
        random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
        if randomize_agents:
            # if only one agent don't spawn in center of ring spawned agent will be locked
            if 'ring' in self.arglist.level and len(self.sim_agents) == 1:
                while (3, 3) in [f.location for f in random_floors]:
                    #print('ring', [f.location for f in random_floors])
                    random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
            # if full divider check that there is at least 1 agent on both sides
            elif 'full' in self.arglist.level:
                while len(self.sim_agents) > 1 and all(f.location[0] < 3 for f in random_floors) or all(f.location[0] > 3 for f in random_floors):
                    #print('full', [f.location for f in random_floors])
                    random_floors = np.random.choice(self.world.get_object_list(['Floor']), len(self.sim_agents))
            # if cross-divider check that at most 2 agents in a block
            elif 'cross' in self.arglist.level:
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


    def reset(self):
        self.t = 0
        self.t_0 = 0
        self.agent_actions = {}
        for a in self.sim_agents:
            a.reset()
        for o in self.orders:
            o.reset(self.t)

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.num_deliveries = 0
        self.num_handovers = 0
        self.num_collisions = 0
        self.num_shuffles = 0
        self.num_invalid_actions = 0
        self.num_location_repeaters = 0
        self.num_holding_repeaters = 0
        self.termination_info = ""
        self.successful = False
        self.failed = False

        self.termination_stats = {k: 0 if 'delivery' not in k else self.arglist.max_num_timesteps for k in self.termination_stats.keys()}


        # load world and level
        self.world: World = None
        self.load_level(
            level=self.arglist.level,
            num_agents=self.arglist.num_agents,
            num_orders=self.arglist.num_orders,
            randomize=self.arglist.randomize,
            reload_level=False,
        )

        
        self.all_subtasks = self.run_recipes()
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = None
        self.obs_tm1 = copy.copy(self)

        self.agent_history = {agent.name: deque([self.get_history_repr(agent)], maxlen=11) for agent in self.sim_agents}

        # for visualization and screenshots
        # self.game = GameImage(
        #     filename=self.filename,
        #     env_id=self.env_id,
        #     world=self.world,
        #     sim_agents=self.sim_agents,
        #     record=self.arglist.record,
        # )
        # self.game.on_init()

        # if self.arglist.record:
        #     self.game.save_image_obs(self.t)

        return copy.copy(self)

    def close(self):
        return

    def render(self, mode="human"):
        logger.info(
            f"""\n=======================================\n[environment-{self.env_id}.step] @ TIMESTEP {self.t}\n======================================="""
        )
        self.display()
        self.print_agents()

    def step(self, action_dict):
        # Track internal environment info.
        if self.t == 0:
            self.t_0 = perf_counter()
            
        self.t += 1
        self.orders = tuple(self.world.objects.get("Order")) 
        
        # Get actions.
        for sim_agent in self.sim_agents:
            action_idx = action_dict[sim_agent.name]
            if 0 <= action_idx < len(self.world.NAV_ACTIONS):
                sim_agent.action = self.world.NAV_ACTIONS[action_idx]
            else:
                sim_agent.action = (0, 0) # BD has 50% chance of no-op, MAPPO 20% chance
            self.agent_history[sim_agent.name].append(self.get_history_repr(sim_agent))
        # Check collisions.
        self.check_collisions()
        if self.t > 2:
            self.obs_tm1.obs_tm1 = None
        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        # Visualize.
        self.display()
        self.print_agents()
        #if self.arglist.record:
            #self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        # new_obs = copy.copy(self)
        # Get an image observation
        #image_obs = self.game.get_image_obs()
        self.compute_stats()
        if self.arglist.evaluate:
            self.record_stats()

        done = self.done()
        reward = self.reward()
        info = {}
        return None, reward, done, info

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
                f"Recording:"
        )

    def done(self):
        # Done if the episode maxes out (or queue empty) or no state change in past 5 actions
        if self.t >= self.arglist.max_num_timesteps:
            self.successful = not any(self.get_remaining_orders())
            self.failed = not self.successful
            if self.arglist.record and not self.arglist.train:
                self.generate_animation(self.t)
            self.termination_info = self.get_termination_info(reason=f"Terminating because passed {self.arglist.max_num_timesteps} timesteps")

        elif all(order.delivered for order in self.orders):
            self.successful = True
            self.failed = False
            if self.arglist.record and not self.arglist.train:
                self.generate_animation(self.t)
            self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")

        else:
            assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

            # Done if subtask is completed.
            
            if any(isinstance(subtask, recipe.Deliver) or subtask.name == 'Deliver'for subtask in self.all_subtasks):
                # Double check all goal_objs are at Delivery.
                self.successful = True
                
                for subtask in self.all_subtasks:
                    _, goal_obj = nav_utils.get_subtask_obj(subtask)

                    delivery_loc = list(filter(lambda o: o.name == "Delivery", self.world.get_object_list()))[0].location
                    goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                    if not any([gol == delivery_loc for gol in goal_obj_locs]):
                        self.successful = False
                        self.failed = False

                #self.successful = self.successful
                self.failed = False
                if self.arglist.record and not self.arglist.train:
                    self.generate_animation(self.t)
                self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")
                
            else:
                self.successful = True
                self.failed = False
                if self.arglist.record and not self.arglist.train:
                    self.generate_animation(self.t)
                self.termination_info = self.get_termination_info(f"Terminating because all orders in queue delivered in {self.t} timesteps")

        done = (([self.successful or self.failed] * len(self.sim_agents)) + [True, True, True, True, True])[:len(self.sim_agents)]

        if all(done):
            self.termination_stats['episode_length'] = self.t
            self.termination_stats['successful'] = self.successful
            self.termination_stats['failed'] = self.failed
            self.termination_stats['episode_duration'] = perf_counter() - self.t_0

        return done

    def reward(self):
        return 1 if self.successful else 0

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def get_remaining_orders(self):
        return [order for order in self.orders if not order.delivered]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(
            recipe__=self.get_remaining_orders()[0].recipe,
            max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        # print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + [a.location for a in self.sim_agents if a.name in subtask_agent_names and a.holding == start_obj]
            
            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + [a.location for a in self.sim_agents if a.name in subtask_agent_names and a.holding == start_obj]
            A_locs = [a for a in A_locs if a not in B_locs]

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + [a.location for a in self.sim_agents if a.name in subtask_agent_names and a.holding == start_obj[0]]
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + [a.location for a in self.sim_agents if a.name in subtask_agent_names and a.holding == start_obj[1]]

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(subtask_agent_names))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
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

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        # print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
                self.agent_history[agent.name][-1].collided=True
            # print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            interaction: ActionRepr = interact(agent=agent, world=self.world, t=self.t, play=self.arglist.play)
            # add to agent history
            ah = self.agent_history[agent.name][-1]
            ah.action_type =interaction.action_type
            ah.delivered = interaction.action_type == "Deliver"
            ah.invalid_actor = (interaction.action_type == "Wait" and agent.action != (0, 0) and not ah.collided)
            ah.holding = agent.holding if agent.holding is not None else None
            ah.location = agent.location
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
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
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
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