

from functools import lru_cache
from typing import Callable, Union
from gymnasium import spaces
import numpy as np
from utils.agent import SimAgent

from utils.core import *

class ObservationHelpers:
    layout_types = ('open-divider', 'partial-divider', 'full-divider', 'cross-divider', 'block-divider', 'ring-divider')
    # possible object types -- dishes are given special importance, food may be fresh, chopped, merged or plated
    object_types = ('Plate', 'Ingredient', 'Dish')
    # possible object names
    object_names = ('Plate', 'Tomato', 'Lettuce', 'Onion')
    # possible object states
    object_states = ('is_plate', 'needs_chopped', 'is_chopped', 'is_merged', 'needs_plate', 'is_deliverable')
    # possible object goal types
    object_goal_types = ('Cutboard', 'Plate', 'Food', 'Delivery')
    # possible neighboring tile occupants
    neighbor_types = ('Plate', 'Food', 'Dish', 'Counter', 'Cutboard', 'Delivery', 'OtherAgent')
    # possible holder types
    holder_types = ('Counter', 'Self', 'OtherAgent', 'Cutboard', 'Delivery')
    # possible users
    user_types = ('Self', 'OtherAgent')
    # ingredients
    ingredients = ('Tomato', 'Lettuce', 'Onion')
    # recipes
    recipes = ('Salad', 'OnionSalad')

    # object type functions
    is_counter = lambda x: isinstance(x, Counter) or 'Counter' in x.name
    is_prep_station = lambda x: isinstance(x, Cutboard) or 'Cut' in x.name
    is_delivery_station = lambda x: isinstance(x, Delivery) or 'Delivery' in x.name
    is_plate = lambda x: isinstance(x, Object) and x.is_plate()
    is_food = lambda x: isinstance(x, Object) and not x.is_plate()
    # object state functions
    needs_prep = lambda x: x.needs_chopped()
    needs_food = lambda x: x.is_chopped()
    needs_plate = lambda x: x.is_merged() and not x.has_plate()
    needs_delivery = lambda x, is_dish: is_dish(x)
    
    # Location Limits
    MAX_NUM_LAYOUTS = 5
    MAX_NUM_TIMESTEPS = 200
    MAX_WIDTH, MAX_HEIGHT = 10, 10
    MAX_DISTANCE = 15
    MAX_NUM_AGENTS = 5
    MAX_NUM_ORDERS = 5
    MAX_NUM_OBJECTS = 5
    MAX_NUM_PREP_STATIONS = 2 
    MAX_NUM_DELIVERY_STATIONS = 2

    obs_space_structure = None
    flattened_obs_space_structure = None
    def __init__(self):
        pass

    @staticmethod
    def get_observation_space_structure(max_num_timesteps, max_width, max_height, max_num_agents, max_num_orders, max_num_dynamic_objects, max_num_prep_stations, max_num_delivery_stations):
        '''Environment state represents the layout of the world
        + the state of the agents (location, holding, holding_goal, next_to)
        + the state of the each object (location, state_of_object (recipe progress), held_by (if held by an agent | counter)
        + the location of each chopping board and delivery station, and whether they are occupied, whether they are in use (agent next to them)
        + the state of each order (recipe_type, status)'''
        if ObservationHelpers.obs_space_structure is not None:
            return ObservationHelpers.obs_space_structure

        # location of an agent, tile or object in 2D coordinates. (-1, -1) represents an invalid location for agents/objects that are not in the world
        ObservationHelpers.MAX_WIDTH = max_width + 1 # +1 for invalid location
        ObservationHelpers.MAX_HEIGHT = max_height + 1 # +1 for invalid location
        
        ObservationHelpers.MAX_NUM_TIMESTEPS = max(ObservationHelpers.MAX_NUM_TIMESTEPS, max_num_timesteps) # timestep
        ObservationHelpers.MAX_NUM_AGENTS = max(ObservationHelpers.MAX_NUM_AGENTS, max_num_agents)
        ObservationHelpers.MAX_NUM_OBJECTS = max(ObservationHelpers.MAX_NUM_OBJECTS, max_num_dynamic_objects)
        ObservationHelpers.MAX_NUM_PREP_STATIONS = max(ObservationHelpers.MAX_NUM_PREP_STATIONS, max_num_prep_stations)
        ObservationHelpers.MAX_NUM_DELIVERY_STATIONS = max(ObservationHelpers.MAX_NUM_DELIVERY_STATIONS, max_num_delivery_stations)
        ObservationHelpers.MAX_NUM_ORDERS = max(ObservationHelpers.MAX_NUM_ORDERS, max_num_orders)


        timestep_space = spaces.Discrete(ObservationHelpers.MAX_NUM_TIMESTEPS)

        location_space = spaces.Box(low=np.array([-1, -1]), high=np.array([ObservationHelpers.MAX_WIDTH, ObservationHelpers.MAX_HEIGHT]), shape=(2,), dtype=np.int32)
        
        distance_space = spaces.Discrete(ObservationHelpers.MAX_DISTANCE) # 0 for holding, 1 for 1 away...

        layout_space = spaces.Discrete(ObservationHelpers.MAX_NUM_LAYOUTS)
        
        # type of object held by an agent or station, can be nothing
        object_space = spaces.Dict({
            'location': location_space,
            'type': spaces.Discrete(len(ObservationHelpers.object_types) + 1), # 0, nothing, 1 plate, 2 food, 3 dish 
            'distance': distance_space, # 0 for holding, 1 for 1 away...
            'contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # [0..0] for nothing
            'state_index': spaces.Discrete(len(ObservationHelpers.object_states) + 1), # +1 for nothing
            'held_by': spaces.Discrete(len(ObservationHelpers.holder_types)), # object is always held by 
        })

        goal_space = spaces.Dict({
            'location': location_space,
            'distance': distance_space, # 0 for holding, 1 for 1 away...
            'type':  spaces.Discrete(len(ObservationHelpers.object_goal_types) + 1), # 0, nothing, 1 cutboard, 2 food, 3 plate, 4 delivery
        })
        
        # agents -- self and others
        agent_space = spaces.Dict({
            'location': location_space,
            'distance': distance_space, # 0 for holding, 1 for 1 away...
            'goal': goal_space,
            'next_to': spaces.MultiDiscrete([len(ObservationHelpers.neighbor_types) + 1] * 4), # can be 0 for floor, set of 4 for 4 directions
        })

        # prep and delivery stations - with object, in use by agent (next to)
        station_space = spaces.Dict({
            'location': location_space,
            'distance': distance_space, # 0 for adjacent, 1 for 1 away...
            'in_use_by': spaces.Discrete(len(ObservationHelpers.user_types) + 1), # 0 for not in use, 1 for in use by self, 2 for in use by other agent
        })

        #orders in the world - with recipe index
        order_space = spaces.Dict({
            'recipe': spaces.Discrete(len(ObservationHelpers.recipes)), # order always has a recipe, can't be nothing
            'recipe_contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # 0 represents that object not in recipe, 
            'status': spaces.Discrete(2), # 0 for not delivered, 1 for delivered
        })
        

        # dict of it all
        obs_structure = {
            'timestep': timestep_space,
            'layout_id': layout_space, 
            'num_agents': spaces.Discrete(ObservationHelpers.MAX_NUM_AGENTS),
            'num_orders': spaces.Discrete(ObservationHelpers.MAX_NUM_ORDERS),
            'agent_id': spaces.Discrete(ObservationHelpers.MAX_NUM_AGENTS),
            'alive': spaces.Discrete(2),
            'agents': spaces.Dict({
                'self': agent_space,
                'others': spaces.Tuple([agent_space] * (ObservationHelpers.MAX_NUM_AGENTS - 1)),
            }),
            'objects': spaces.Tuple([object_space] * ObservationHelpers.MAX_NUM_OBJECTS),
            'stations': spaces.Dict({
                'prep_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_PREP_STATIONS),
                'delivery_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_DELIVERY_STATIONS),
            }),
            'next_order': spaces.Tuple([order_space])
        }

        ObservationHelpers.obs_space_structure = spaces.Dict(obs_structure)
        return ObservationHelpers.obs_space_structure


    @staticmethod
    def get_rl_obs(layout, timestep, agents, num_agents, dynamic_objects, prep_stations, delivery_stations, orders, num_orders):
        '''
            Observation structure is a dict of the world state:
                >>> obs_structure = {
                        'timestep': spaces.Discrete(max_num_timesteps),
                        'layout_id': spaces.Discrete(max_num_layouts),
                        'num_agents': spaces.Discrete(max_num_agents),
                        'num_orders': spaces.Discrete(max_num_orders),
                        'agent_id': spaces.Discrete(max_num_agents),'
                        'alive': spaces.Discrete(2),
                        'agents': spaces.Dict({
                            'self': agent_space,
                            'others': spaces.Tuple([agent_space] * (max_num_agents - 1)),
                        }),
                        'objects': spaces.Tuple([object_space] * max_num_dynamic_objects),
                        'stations': spaces.Dict({
                            'prep_stations': spaces.Tuple([station_space] * max_num_prep_stations),
                            'delivery_stations': spaces.Tuple([station_space] * max_num_delivery_stations),
                        }),
                        'orders': spaces.Tuple([order_space] * max_num_orders),
                    }
        '''
        
        rl_obs = [{
            'timestep': timestep,
            'layout_id': ObservationHelpers.layout_types.index(layout),
            'num_agents': num_agents,
            'num_orders': num_orders,
            'agent_id': _,
            'alive': agents[_] is not None,
        } for _ in range(len(agents))]

        # agent states (max 4 agents)
        MNA = ObservationHelpers.MAX_NUM_AGENTS
        possible_neighbors = [_ for _ in [*dynamic_objects, *prep_stations, *delivery_stations, *agents] if _ is not None] # crucial -- do not change order
        possible_goals = [_ for _ in possible_neighbors if not isinstance(_, SimAgent)]
        all_recipes = {_.recipe for _ in orders if _ is not None}
        # to check if an object is fully ready
        is_dish = lambda x: any(r.full_plate_name == x.name for r in all_recipes)

        # to determine what is next to agent
        is_direct_neighbor_of = lambda loc, x, y: any(c == loc for c in [(x, y+1), (x, y-1), (x-1, y), (x+1, y)])
        get_neighbors = lambda agent: [] if agent is None else [n for n in possible_neighbors if n is not None and is_direct_neighbor_of(agent.location, *n.location) and n != agent]

        # order states
        MNO = ObservationHelpers.MAX_NUM_ORDERS
        order_state = [ObservationHelpers.get_order_state(o) for o in orders][:MNO][0]

        for i, agent in enumerate(agents):
            self_agent = agent
            self_location = agent.location if agent is not None else None
            others = (agents[:i] + agents[i+1:] + ([None] * MNA))[:MNA - 1]
            # agent states
            self_state = ObservationHelpers.get_agent_state(self_agent, self_location, get_neighbors(self_agent), possible_goals, is_dish)
            others_states = tuple(ObservationHelpers.get_agent_state(o, self_location, get_neighbors(o), possible_goals, is_dish) for o in others)
            # agent_space
            rl_obs[i]['agents'] = {
                'self': self_state,
                'others': others_states,
            }

            # dynamic object states
            MNB = ObservationHelpers.MAX_NUM_OBJECTS
            all_objects = (*dynamic_objects, *([None] * MNB))[:MNB]
            possible_holders = possible_neighbors[len(dynamic_objects):] # get stations and agents
            
            rl_obs[i]['objects'] = tuple(ObservationHelpers.get_object_state(o, self_location, possible_holders, is_dish) for o in all_objects)

            MNP = ObservationHelpers.MAX_NUM_PREP_STATIONS
            MND = ObservationHelpers.MAX_NUM_DELIVERY_STATIONS

            prep_stations = (*prep_stations, *([None] * MNP))[:MNP]
            delivery_stations = (*delivery_stations, *([None] * MND))[:MND]
            # station states
            rl_obs[i]['stations'] = {
                'prep_stations': tuple(ObservationHelpers.get_station_state(p, self_agent, others) for p in prep_stations),
                'delivery_stations': tuple(ObservationHelpers.get_station_state(d, self_agent, others) for d in delivery_stations),
            }

            # order states are common for each agent
            rl_obs[i]['next_order'] = order_state

        return rl_obs

    @staticmethod
    def flatten_dict_obs(obs: spaces.Dict):
        '''
            Observation structure is a dict of the world state:
                >>> obs_structure = {
                        'timestep': spaces.Discrete(max_num_timesteps),
                        'layout_id': spaces.Discrete(max_num_layouts),
                        'num_agents': spaces.Discrete(max_num_agents), 
                        'num_orders': spaces.Discrete(max_num_orders),
                        'agent_id': spaces.Discrete(max_num_agents),'
                        'alive': spaces.Discrete(2),
                        'agents': spaces.Dict({
                            'self': agent_space,
                            'others': spaces.Tuple([agent_space] * (max_num_agents - 1)),
                        }),
                        'objects': spaces.Tuple([object_space] * max_num_dynamic_objects),
                        'stations': spaces.Dict({
                            'prep_stations': spaces.Tuple([station_space] * max_num_prep_stations),
                            'delivery_stations': spaces.Tuple([station_space] * max_num_delivery_stations),
                        }),
                        'orders': spaces.Tuple([order_space] * max_num_orders),
                    }
        '''
        timestep = obs['timestep']
        layout_id = obs['layout_id']
        num_agents = obs['num_agents']
        num_orders = obs['num_orders']
        agent_id = obs['agent_id']
        alive = int(obs['alive'])
        
        rl_obs_flattened = np.array([timestep, layout_id, num_agents, num_orders, agent_id, alive])
        
        self_agent = obs['agents']['self']
        others_agent = obs['agents']['others']
        for a in [self_agent, *others_agent]:
            loc = a['location']
            d = a['distance']
            goal_loc, goal_distance, goal_type = a['goal'].values()
            neighbors = a['next_to']
            
            rl_obs_flattened = np.concatenate((
                rl_obs_flattened, 
                np.array([*loc, d, *goal_loc, goal_distance, goal_type, *neighbors])
                ))

        for obj in obs['objects']:
            loc = obj['location']
            otype = obj['type']
            distance = obj['distance']
            contents = obj['contents']
            state = obj['state_index']
            held_by = obj['held_by']

            rl_obs_flattened = np.concatenate((
                rl_obs_flattened, 
                np.array([*loc, otype, distance, *contents, state, held_by])
                ))
            
        stations = (*obs['stations']['prep_stations'], *obs['stations']['delivery_stations'])
        for station in stations:
            loc = station['location']
            d = station['distance']
            in_use_by = station['in_use_by']

            rl_obs_flattened = np.concatenate((
                rl_obs_flattened, 
                np.array([*loc, d, in_use_by])
                ))

        #for order in obs['orders']:
        recipe, recipe_contents, status = obs['next_order'].values()
        rl_obs_flattened = np.concatenate((rl_obs_flattened, np.array([recipe, *recipe_contents, status])))

        return rl_obs_flattened
    
    @staticmethod
    def get_order_state(order: Order):
        # order_space = spaces.Dict({
        #     'recipe': spaces.Discrete(len(ObservationHelpers.recipes)), # order always has a recipe, can't be nothing
        #     'recipe_contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # 0 represents that object not in recipe, 
        #     'status': spaces.Discrete(2), # 0 for not delivered, 1 for delivered
        # })
        
        RECIPE_LEN = len(ObservationHelpers.object_names)
        false_recipe = [0] * RECIPE_LEN # in case recipe has less than max_num_recipe_items
        # invalid order -- shown as a delivered order with no recipe
        if order is None:
            return {
                'recipe': 0,
                'recipe_contents': false_recipe,
                'status': 1 
            }

        # valid order
        contents = order.recipe.full_plate_name.split('-')
        recipe_contents = [int(ObservationHelpers.object_names[i] in contents) for i in range(RECIPE_LEN)]
        return {
            'recipe': ObservationHelpers.recipes.index(order.recipe.name), # +1 because 0 when invalid  order
            'recipe_contents': recipe_contents,
            'status': int(order.delivered),
        }

    @staticmethod
    def get_agent_state(agent: SimAgent, self_agent_location: tuple[int,int], neighbors: list[Union[Object, Cutboard, Delivery, GridSquare, SimAgent]], possible_goals: list[Union[Object, Cutboard, Delivery, GridSquare]], is_dish_checker: Callable[[Object], bool]):
        # agent_space = spaces.Dict({
        #     'location': location_space,
        #     'distance': distance_space, # 0 for self, 1 for adjacent, 2 for 2 away, 3 for 3 away, 4 for 4 away
        #     'goal': goal_space,
        #     'next_to': spaces.MultiDiscrete([len(ObservationHelpers.neighbor_types) + 1] * 4), # can be 0 for floor, set of 4 for 4 directions 
        # })

        MAX_NUM_DIRECTIONS = 4 # can be 9 for higher observation space and indirect neighbors
        # invalid agent
        if agent is None:
            return {
                'location': [-1, -1],
                'distance': 14, # distance to self
                'goal': ObservationHelpers.get_goal_state(None, None, None, None),
                'next_to': [0] * MAX_NUM_DIRECTIONS, # 0 for nothing of use such as floor or empty counter - 4 directions
            }

        # to determine if agent goal
        goal = ObservationHelpers.get_goal_state(self_agent_location, agent.holding, possible_goals, is_dish_checker)
        neighbor_types = ObservationHelpers.get_neighbor_types(agent.location, neighbors, is_dish_checker) 
        
        return {
            'location': [agent.location[0], agent.location[1]],
            'distance': ObservationHelpers.get_distance(agent.location, self_agent_location) if self_agent_location is not None else 14, # distance to self, 0 for self, 1 for adjacent, 2 for 2 away, 3 for 3 away, 4 for 4 away
            'goal': goal,
            'next_to': neighbor_types, # 0 for nothing of use such as floor or empty counter
        }

    @staticmethod
    def get_object_state(obj: Object, agent_location: tuple[int, int], possible_holders: list[Union[Object, GridSquare, SimAgent]],  is_dish: Callable[[Object], bool]):
        # object_space = spaces.Dict({
        #     'location': location_space,
        #     'type': spaces.Discrete(len(ObservationHelpers.object_types) + 1), # 0, nothing, 1 plate, 2 food, 3 dish 
        #     'distance': spaces.Discrete(15), # 0 for holding, 1 for 1 away...
        #     'contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # [0..0] for nothing
        #     'state_index': spaces.Discrete(len(ObservationHelpers.object_states) + 1), # +1 for nothing
        #     'held_by': spaces.Discrete(len(ObservationHelpers.holder_types)), # object is always held by someone, can't be nothing
        # })

        INGREDIENTS_LEN = len(ObservationHelpers.object_names)
        # station or agent not holding anything
        if obj is None:
            return {
                'location': [-1, -1],
                'type': 0,
                'distance': 14,
                'contents': [0] * INGREDIENTS_LEN,
                'state_index': 0,
                'held_by': 0,
            }

        # type of object - 0 plate, 1 food, 2 dish
        NUM_OBJECT_TYPES = len(ObservationHelpers.object_types)
        obj_type = [obj.is_plate(), obj.needs_chopped() or obj.is_chopped() or obj.is_merged(), is_dish(obj)].index(True)

        # object state
        NUM_OBJECT_CONTENTS = len(ObservationHelpers.object_names)
        obj_contents = [0] * NUM_OBJECT_CONTENTS
        object_parts = obj.full_name.split('-') # for example ['ChoppedTomato', 'Plate', 'FreshLettuce']
        for part_name in object_parts:
            # check for example if 'Tomato' in 'ChoppedTomato'
            c_index = [obj_name in part_name for obj_name in ObservationHelpers.object_names].index(True)
            obj_contents[c_index] = 1

        return {
            'location': obj.location,
            'type': obj_type + 1, # +1 because 0 when invalid object (doesn't exist or removed from world by being merged into something else)
            'distance': ObservationHelpers.get_distance(obj.location, agent_location) if agent_location is not None else 14, # distance to agent, 0 for holding, 1 for adjacent, 2 for 2 away, 3 for 3 away, 4 for 4 away
            'contents': obj_contents, # all 0s if object is invalid
            'state_index': obj.get_state_index(is_dish(obj)) + 1, # +1 because 0 if object is invalid,
            'held_by': ObservationHelpers.get_object_holder_type(obj, agent_location, possible_holders), # 0 for counter, 1 for prep, 2 for delivery_station, 3 for agent
        }

    @staticmethod
    def get_goal_state(agent_location: tuple[int, int], obj: Object, possible_goals: list[Union[Cutboard, Delivery, GridSquare, Object]], is_dish: Callable[[Object], bool]):

        if agent_location is None:
            return {
                'location': [-1, -1],
                'distance': 14,
                'type': 0,
            }
            
        # Holding nothing, goal is closest fresh object
        if obj is None:
            closest_food = sorted([o for o in possible_goals if ObservationHelpers.is_food(o)], 
                key=lambda _: ObservationHelpers.get_distance(agent_location, _.location))[0]
            closest_plate = sorted([o for o in possible_goals if ObservationHelpers.is_plate(o)],
                key=lambda _: ObservationHelpers.get_distance(agent_location, _.location))
            closest_plate = closest_plate[0] if any(closest_plate) else None
            closest = closest_food if not closest_food.is_held else closest_plate
            goal_type = 0 if closest is None else (1 + ObservationHelpers.object_goal_types.index('Food') if ObservationHelpers.is_food(closest) else ObservationHelpers.object_goal_types.index('Plate'))
        else: 
            # object goal state
            is_possible_goal = lambda g: (ObservationHelpers.needs_prep(obj) and ObservationHelpers.is_prep_station(g)) or (ObservationHelpers.needs_food(obj) and ObservationHelpers.is_food(g)) or (ObservationHelpers.needs_plate(obj) and ObservationHelpers.is_plate(g)) or (ObservationHelpers.needs_delivery(obj, is_dish) and ObservationHelpers.is_delivery_station(g))

            closest = sorted([pg for pg in possible_goals if is_possible_goal(pg)], key=lambda _: ObservationHelpers.get_distance(obj.location, _.location))
            closest = closest[0] if any(closest) else None
            goal_type = 0 if closest is None else (1 + ([ObservationHelpers.is_prep_station(closest), ObservationHelpers.is_food(closest), ObservationHelpers.is_plate(closest), ObservationHelpers.is_delivery_station(closest)].index(True)))

        return {
            'location': closest.location if closest is not None else [-1, -1],
            'distance': ObservationHelpers.get_distance(agent_location, closest.location) if closest is not None else 14,
            'type': goal_type,
        }

    @staticmethod
    def get_neighbor_types(agent_location: tuple[int, int], neighbors: list[Union[Object, Cutboard, Delivery, SimAgent]], is_dish: Callable[[Object], bool]):
        MAX_NUM_DIRECTIONS = 4
        # determine if down, up, left, right [0, 1, 2, 3] in relation to agent -- assumes 4 directions
        get_neighbor_direction = lambda x, y: [c == agent_location for c in [(x, y-1), (x, y+1), (x+1, y), (x-1, y)]].index(True)
        neighbor_type_indexes = [0] * MAX_NUM_DIRECTIONS # 4 neighbors, 0 means floor or empty counter

        # priority based type assignment
        for n in neighbors:
            # determine if down, up, left, right [0, 1, 2, 3] in relation to agent -- should be reworked for higher directions
            direction = get_neighbor_direction(*n.location)
            # if already determined neighbor type, skip
            if neighbor_type_indexes[direction] > 0:
                continue
            # check if neighbor is agent, dish, food, or plate -ignored if empty floor/counter/station
            if isinstance(n, SimAgent):
                neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('OtherAgent')
            elif isinstance(n, Object):
                if is_dish(n):
                    neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Dish')
                elif not n.is_plate():    
                    neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Food')
                elif n.is_plate():
                    neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Plate')
            elif isinstance(n, Cutboard):
                neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Cutboard')
            elif isinstance(n, Delivery):
                neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Delivery')
            elif isinstance(n, Counter):
                neighbor_type_indexes[direction] = ObservationHelpers.neighbor_types.index('Counter')

        return neighbor_type_indexes

    @staticmethod
    def get_object_holder_type(item: Object, self_agent_location: tuple[int, int], possible_holders: list[Union[Object, GridSquare, SimAgent]]):
        # spaces.Discrete(len(ObservationHelpers.holder_types))
        
        if self_agent_location is not None and item.location == self_agent_location:
            return ObservationHelpers.holder_types.index('Self')

        # should only be one or None which means object is on a counter
        holder = [h for h in possible_holders if h.holding == item]

        # not held by agent, delivery station, or cutboard means it is on a counter
        if not any(holder):
            return ObservationHelpers.holder_types.index('Counter')

        # determine if held by other_agent, delivery station, or cutboard
        choices = list(zip(['OtherAgent', 'Cutboard', 'Delivery'], [SimAgent, Cutboard, Delivery]))
        holder_type = [n for n, c in choices if isinstance(holder[0], c)][0]
        
        return ObservationHelpers.holder_types.index(holder_type)

    @staticmethod
    def get_station_state(station: Union[Cutboard, Delivery], self_agent: SimAgent, others:list[SimAgent]):
        # spaces.Dict({
        #     'location': location_space,
        #     'distance': distance_space, # 0 for adjacent, 1 for 1 away, 2 for 2 away, 3 for 3 away, 4 for 4 away
        #     'in_use_by': spaces.Discrete(len(ObservationHelpers.user_types)), # 0 for not in use, 1 for in use by self, 2 for in use by other agent
        # })
        
        # invalid station
        if station is None:
            return {
            'location': [-1, -1],
            'distance': 14,
            'in_use_by': 0, # not in use
        }
        
        # to determine what is next to agent
        is_direct_neighbor_of_station = lambda x, y: any(c == station.location for c in [(x, y+1), (x, y-1), (x-1, y), (x+1, y)]) 
        neighbors = [n for n in [self_agent, *others] if n is not None and is_direct_neighbor_of_station(*n.location)]
        any_user = any(neighbors)
        used_by_self = any_user and neighbors[0] == self_agent
        
        return {
            'location': [station.location[0], station.location[1]],
            'distance': ObservationHelpers.get_distance(station.location, self_agent.location) if self_agent is not None else 14,
            'in_use_by': [not any_user, used_by_self, not used_by_self].index(True), # 0 for no agent next to this station, 1 for self, 2 for other
        }

    @staticmethod
    @lru_cache(4096)
    def get_distance(loc1, loc2):
        """Get Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    @staticmethod
    def get_observation_space_structure_flattened():
        # dict of it all
        # obs_structure = {
        #     'agent_id': spaces.Discrete(4),
        #     'alive': spaces.Discrete(2),
        #     'layout_id': spaces.Discrete(5),
        #     'timestep': spaces.Discrete(200),
        #     'agents': spaces.Dict({
        #         'self': agent_space,
        #         'others': spaces.Tuple([agent_space] * (ObservationHelpers.MAX_NUM_AGENTS - 1)),
        #     }),
        #     'objects': spaces.Tuple([object_space] * ObservationHelpers.MAX_NUM_OBJECTS),
        #     'stations': spaces.Dict({
        #         'prep_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_PREP_STATIONS),
        #         'delivery_stations': spaces.Tuple([station_space] * ObservationHelpers.MAX_NUM_DELIVERY_STATIONS),
        #     }),
        # }

        if ObservationHelpers.flattened_obs_space_structure is not None:
            return ObservationHelpers.flattened_obs_space_structure

        timestep = spaces.Box(low=0, high=ObservationHelpers.MAX_NUM_TIMESTEPS-1, shape=(1,), dtype=np.int16)
        layout_id_space = spaces.Box(low=0, high=ObservationHelpers.MAX_NUM_LAYOUTS, shape=(1,), dtype=np.int16)
        num_agents_space = spaces.Box(low=0, high=ObservationHelpers.MAX_NUM_AGENTS-1, shape=(1,), dtype=np.int16)
        num_orders_space = spaces.Box(low=0, high=ObservationHelpers.MAX_NUM_ORDERS, shape=(1,), dtype=np.int16)
        agent_id_space = spaces.Box(low=0, high=ObservationHelpers.MAX_NUM_AGENTS-1, shape=(1,), dtype=np.int16)
        alive_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int16)
        
        self_space = ObservationHelpers.get_agent_space_flattened()
        others_space = ObservationHelpers.get_other_agents_space_flattened()

        agents_space = spaces.Box(low=np.concatenate([self_space.low, others_space.low]), high=np.concatenate([self_space.high, others_space.high]), shape=(spaces.flatdim(self_space) + spaces.flatdim(others_space),), dtype=np.int16)

        object_space = [ObservationHelpers.get_object_space_flattened()] * ObservationHelpers.MAX_NUM_OBJECTS
        objects_space = spaces.Box(low=np.concatenate([_.low for _ in object_space]), high=np.concatenate([_.high for _ in object_space]), shape=(sum(spaces.flatdim(_) for _ in object_space),),dtype=np.int16)
        
        station_space = [ObservationHelpers.get_station_space_flattened()] * (ObservationHelpers.MAX_NUM_PREP_STATIONS + ObservationHelpers.MAX_NUM_DELIVERY_STATIONS)
        stations_space = spaces.Box(low=np.concatenate([_.low for _ in station_space]), high=np.concatenate([_.high for _ in station_space]), shape=(sum(spaces.flatdim(_) for _ in station_space),), dtype=np.int16)

        order_space = [ObservationHelpers.get_order_space_flattened()]
        orders_space = spaces.Box(low=np.concatenate([_.low for _ in order_space]), high=np.concatenate([_.high for _ in order_space]), shape=(sum(spaces.flatdim(_) for _ in order_space),), dtype=np.int16)

        ObservationHelpers.flattened_obs_space_structure = spaces.Box(
            low=np.concatenate([timestep.low, layout_id_space.low, num_agents_space.low, num_orders_space.low, agent_id_space.low, alive_space.low, agents_space.low, objects_space.low, stations_space.low, orders_space.low]), 
            high=np.concatenate([timestep.high, layout_id_space.high, num_agents_space.high, num_orders_space.high, agent_id_space.high, alive_space.high, agents_space.high, objects_space.high, stations_space.high, orders_space.high]), 
            shape=(sum(spaces.flatdim(_) for _ in 
            [timestep, layout_id_space, num_agents_space, num_orders_space, agent_id_space, alive_space, agents_space, objects_space, stations_space, orders_space]),), 
            dtype=np.int16)
        
        return ObservationHelpers.flattened_obs_space_structure

    @staticmethod
    def get_location_space_flattened():
        return spaces.Box(low=np.array([-1, -1]), high=np.array([ObservationHelpers.MAX_WIDTH, ObservationHelpers.MAX_HEIGHT]), shape=(2,), dtype=np.int16)

    @staticmethod
    def get_goal_space_flattened():
        # goal_space = spaces.Dict({
        #     'location': location_space,
        #     'distance': distance_space, # 0 for holding, 1 for 1 away...
        #     'type':  spaces.Discrete(len(ObservationHelpers.object_goal_types) + 1), # 0, nothing, 1 cutboard, 2 food, 3 plate, 4 delivery
        # })
        
        l_f = ObservationHelpers.get_location_space_flattened()
        d_f = spaces.Box(low=0, high=ObservationHelpers.MAX_DISTANCE, shape=(1,), dtype=np.int16)
        t_f = spaces.Box(low=0, high=len(ObservationHelpers.object_goal_types), shape=(1,), dtype=np.int16)

        dim = spaces.flatdim(l_f) + spaces.flatdim(d_f) + spaces.flatdim(t_f)
        
        return spaces.Box(low=np.array([*l_f.low, *d_f.low, *t_f.low]), high=np.array([*l_f.high, *d_f.high, *t_f.high]), shape=(dim,), dtype=np.int16)

    @staticmethod
    def get_agent_space_flattened():
        # agent_space = spaces.Dict({
        #     'location': location_space,
        #     'distance': distance_space,
        #     'goal': goal_space,
        #     'next_to': spaces.MultiDiscrete([len(ObservationHelpers.neighbor_types) + 1] * 4), # can be 0 for floor, set of 4 for 4 directions
        # })

        l_f = ObservationHelpers.get_location_space_flattened()
        d_f = spaces.Box(low=0, high=ObservationHelpers.MAX_DISTANCE, shape=(1,), dtype=np.int16)
        g_f = ObservationHelpers.get_goal_space_flattened()
        nt_f = spaces.Box(low=0, high=len(ObservationHelpers.neighbor_types), shape=(4,), dtype=np.int16)

        flattened_spaces = [l_f, d_f, g_f, nt_f]
        
        dim = sum(spaces.flatdim(_) for _ in flattened_spaces)
        
        return spaces.Box(low=np.array([*l_f.low, *d_f.low, *g_f.low, *nt_f.low]), high=np.array([*l_f.high, *d_f.high, *g_f.high, *nt_f.high]), shape=(dim,), dtype=np.int16)

    @staticmethod
    def get_other_agents_space_flattened():
        others_space = [ObservationHelpers.get_agent_space_flattened()] * (ObservationHelpers.MAX_NUM_AGENTS - 1)
        
        return spaces.Box(low=np.concatenate([o.low for o in others_space]), high=np.concatenate([o.high for o in others_space]), shape=(sum(spaces.flatdim(o) for o in others_space),), dtype=np.int16)

    @staticmethod
    def get_object_space_flattened():
        # dynamic_object_space = spaces.Dict({
        #     'location': location_space,
        #     'type': spaces.Discrete(len(ObservationHelpers.object_types) + 1), # 0, nothing, 1 plate, 2 food, 3 dish
        #     'distance': spaces.Discrete(15),
        #     'contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # [0..0] for nothing
        #     'state_index': spaces.Discrete(len(ObservationHelpers.object_states) + 1), # +1 for nothing
        #     'held_by': spaces.Discrete(len(ObservationHelpers.holder_types)), # object is always held by someone, can't be nothing
        # })

        l_f = ObservationHelpers.get_location_space_flattened()
        t_f = spaces.Box(low=0, high=len(ObservationHelpers.object_types), shape=(1,), dtype=np.int16)
        d_f = spaces.Box(low=0, high=ObservationHelpers.MAX_DISTANCE, shape=(1,), dtype=np.int16)
        c_f = spaces.Box(low=0, high=1, shape=(len(ObservationHelpers.object_names),), dtype=np.int16)
        s_f = spaces.Box(low=0, high=len(ObservationHelpers.object_states), shape=(1,), dtype=np.int16)
        h_f = spaces.Box(low=0, high=len(ObservationHelpers.holder_types), shape=(1,), dtype=np.int16)

        dim = spaces.flatdim(l_f) + spaces.flatdim(t_f) + spaces.flatdim(d_f) + spaces.flatdim(c_f) + spaces.flatdim(s_f) + spaces.flatdim(h_f)

        return spaces.Box(low=np.array([*l_f.low, *t_f.low, *d_f.low, *c_f.low, *s_f.low, *h_f.low]), high=np.array([*l_f.high, *t_f.high, *d_f.high, *c_f.high, *s_f.high, *h_f.high]), shape=(dim,), dtype=np.int16)

    @staticmethod
    def get_station_space_flattened():
        # spaces.Dict({
        #     'location': location_space,
        #     'distance': distance_space,
        #     'in_use_by': spaces.Discrete(len(ObservationHelpers.user_types) + 1), # 0 for not in use, 1 for in use by self, 2 for in use by other agent
        # })
        l_f = ObservationHelpers.get_location_space_flattened()
        d_f = spaces.Box(low=0, high=ObservationHelpers.MAX_DISTANCE, shape=(1,), dtype=np.int16)
        iub_f = spaces.Box(low=0, high=len(ObservationHelpers.user_types) + 1, shape=(1,), dtype=np.int16)

        dim = spaces.flatdim(l_f) + spaces.flatdim(d_f) + spaces.flatdim(iub_f)
        
        return spaces.Box(low=np.array([*l_f.low, *d_f.low, *iub_f.low]), high=np.array([*l_f.high, *d_f.high, *iub_f.high]), shape=(dim,), dtype=np.int16)

    @staticmethod
    def get_order_space_flattened():
        # spaces.Dict({
        #     'recipe': spaces.Discrete(len(ObservationHelpers.recipes)), # order always has a recipe, can't be nothing
        #     'recipe_contents': spaces.MultiBinary(len(ObservationHelpers.object_names)), # 0 represents that object not in recipe, 
        #     'status': spaces.Discrete(2), # 0 for not delivered, 1 for delivered
        # }):

        r_f = spaces.Box(low=0, high=len(ObservationHelpers.recipes), shape=(1,), dtype=np.int16)
        rc_f = spaces.Box(low=0, high=1, shape=(len(ObservationHelpers.object_names),), dtype=np.int16)
        s_f = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int16)

        dim = spaces.flatdim(r_f) + spaces.flatdim(rc_f) + spaces.flatdim(s_f)
        
        return spaces.Box(low=np.array([*r_f.low, *rc_f.low, *s_f.low]), high=np.array([*r_f.high, *rc_f.high, *s_f.high]), shape=(dim,), dtype=np.int16)

    @staticmethod
    def normalize_flattened_obs(obs):
        flattened_struct = ObservationHelpers.get_observation_space_structure_flattened()
        limits = flattened_struct.low, flattened_struct.high

        return (obs - limits[0]) / (limits[1] - limits[0])