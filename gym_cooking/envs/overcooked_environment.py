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

from recipe_planner.recipe import *
import recipe_planner.utils as recipe_utils

from utils.interact import ActionRepr, interact
from utils.core import *

import gymnasium as gym
from gymnasium import spaces

from utils.agent import COLORS, SimAgent
from utils.world import World
from misc.game.gameimage import GameImage

from recipe_planner.utils import Get, Chop, Merge, Deliver,Grill

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
       
        # For visualizing episode.
        self.rep = []
        # For tracking data during an episode.
        self.agent_history: dict[str, list[AgentHistoryRepr]] = {}

        
       
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
                        self.recipes.append(globals()['SimpleBun']())
                        self.recipes.append(globals()['BunLettuce']())
                        self.recipes.append(globals()['BunLettuceTomato']())
                        self.recipes.append(globals()['Burger']())

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
            j=0
            for i in range(num_orders):  # append orders for level
                
                random_recipe = self.recipes[j]
                location = len(self.default_world.objects.get("Order")), y 
                nextOrder = RepToClass[Rep.ORDER](random_recipe, location, self.t, delivery_window)
                self.default_world.objects.get("Order").append(nextOrder)
                j+=1
                if j==len(self.recipes):
                    j=0
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
    
    def step(self, action_dict):
        # Track internal environment info.
        if self.t == 0:
            self.t_0 = perf_counter()
        
        self.orders = tuple(self.world.objects.get("Order")) 

        # Parse action
        for sim_agent in self.sim_agents:
            if sim_agent.name in action_dict:
                action_idx = action_dict[sim_agent.name]
                if action_idx is None:
                    sim_agent.action = None
                else:
                    action_type, arg = action_idx[1].strip("()").split('(')
                    targets = [t.strip() for t in arg.split(',')]
                    action = globals().get(action_type)(*targets)
                    sim_agent.action = action
                    self.t = action_idx[0]
            self.agent_history[sim_agent.name].append(self.get_history_repr(sim_agent))

        # agent = next(agent for agent in self.sim_agents if agent.name in action_dict)

        
        # set current state as previous
        self.obs_tm1 = None
        self.obs_tm1 = copy.copy(self)

       
        self.execute_navigation() # append to agent activity
        
        if self.arglist.record:
            self.game.save_image_obs(self.t)

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
        return False
    
    def get_termination_info(self, reason):
        #logger.info(f"{reason}")
        return (reason, self.t, self.num_deliveries, self.num_handovers, self.num_collisions, self.num_shuffles, self.num_invalid_actions, self.num_location_repeaters, self.num_holding_repeaters, self.termination_info, self.termination_stats
        )

        
    def reward(self):
        
        return 100

    def get_image_observation_space_structure(self):
        image_dims = self.game.get_image_obs().shape if self.game is not None else (240, 200, 3)
        return spaces.Box(low=0, high=255, shape=image_dims, dtype=np.uint8)
        
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
  
    def execute_navigation(self):
        for agent in self.sim_agents:
            interaction: ActionRepr = interact(agent=agent, world=self.world, t=self.t, play=self.arglist.play)

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