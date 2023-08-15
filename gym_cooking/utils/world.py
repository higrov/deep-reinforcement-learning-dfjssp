import numpy as np
from collections import defaultdict
from itertools import product
import networkx as nx
import copy
from functools import lru_cache
from utils.utils import timeit

import recipe_planner.utils as recipe

from navigation_planner.utils import manhattan_dist
from utils.core import Floor, Food, Object, GridSquare, Counter, Order, Plate, RepToClass


class World:
    """World class that hold all of the non-agent objects in the environment."""
    # DOWN UP LEFT RIGHT
    NAV_ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self):
        self.rep = []  # [row0, row1, ..., rown]
        self.objects = defaultdict(lambda: [])
        self.default_objects = defaultdict(lambda: [])
        self.distances = None
        self.reachability_graph = None

    def get_repr(self):
        return self.get_dynamic_objects()

    def __str__(self):
        _display = list(map(lambda x: "".join(map(lambda y: y + " ", x)), self.rep))
        return "\n".join(_display)

    def __copy__(self):
        new = World()
        new.__dict__ = self.__dict__.copy()
        new.objects = copy.deepcopy(self.objects)
        new.reachability_graph = self.reachability_graph
        new.distances = self.distances
        return new

    def update_display(self):
        # Reset the current display (self.rep).
        self.rep = [[" " for i in range(self.width)] for j in range(self.height)]
        objs = []
        for o in self.objects.values():
            objs += o
        for obj in objs:
            if isinstance(obj, Order):
                if not obj.delivered:
                    self.add_object(obj, obj.location)
            else:
                self.add_object(obj, obj.location)
        # for obj in self.objects["Tomato"]:
        #     self.add_object(obj, obj.location)
        return self.rep

    def print_objects(self):
        for k, v in self.objects.items():
            print(k, list(map(lambda o: o.location, v)))

    def make_loc_to_gridsquare(self):
        """Creates a mapping between object location and object."""
        self.loc_to_gridsquare = {}
        for obj in self.get_object_list():
            if isinstance(obj, GridSquare):
                self.loc_to_gridsquare[obj.location] = obj

    def make_reachability_graph(self):
        """Create a reachability graph between world objects."""
        # self.reachability_graph = nx.Graph()
        # for x in range(self.width):
        #     for y in range(self.height-1):
        #         location = (x, y)
        #         gs = self.loc_to_gridsquare[(x, y)]

        #         # If not collidable, add node with direction (0, 0).
        #         if not gs.collidable:
        #             self.reachability_graph.add_node((location, (0, 0)))

        #         # Add nodes for collidable gs + all edges.
        #         for nav_action in World.NAV_ACTIONS:
        #             new_location = self.inbounds(location=tuple(np.asarray(location) + np.asarray(nav_action)))
        #             new_gs = self.loc_to_gridsquare[new_location]

        #             # If collidable, add edges for adjacent noncollidables.
        #             if gs.collidable and not new_gs.collidable:
        #                 self.reachability_graph.add_node((location, nav_action))
        #                 if (new_location, (0, 0)) in self.reachability_graph:
        #                     self.reachability_graph.add_edge((location, nav_action),
        #                                                      (new_location, (0, 0)))
        #             # If not collidable and new_gs collidable, add edge.
        #             elif not gs.collidable and new_gs.collidable:
        #                 if (new_location, tuple(-np.asarray(nav_action))) in self.reachability_graph:
        #                     self.reachability_graph.add_edge((location, (0, 0)),
        #                                                      (new_location, tuple(-np.asarray(nav_action))))
        #             # If both not collidable, add direct edge.
        #             elif not gs.collidable and not new_gs.collidable:
        #                 if (new_location, (0, 0)) in self.reachability_graph:
        #                     self.reachability_graph.add_edge((location, (0, 0)), (new_location, (0, 0)))
        #             # If both collidable, add nothing.

        # If you want to visualize this graph, uncomment below.
        # plt.figure()
        # nx.draw(self.reachability_graph)
        # plt.show()
    
    def get_lower_bound_between(self, subtask_name, agent_locs, A_locs, B_locs):
        """Return distance lower bound between subtask-relevant locations."""
        lower_bound = self.perimeter + 1
        for A_loc, B_loc in product(A_locs, B_locs):
            bound = self.get_lower_bound_between_helper(
                    subtask_name=subtask_name,
                    agent_locs=agent_locs,
                    A_loc=A_loc,
                    B_loc=B_loc)
            if bound < lower_bound:
                lower_bound = bound
        return lower_bound

    @lru_cache(maxsize=50_000)
    def get_lower_bound_between_helper(self, subtask_name, agent_locs, A_loc, B_loc):
        # lower_bound = self.perimeter + 1
        # A = self.get_gridsquare_at(A_loc)
        # B = self.get_gridsquare_at(B_loc)
        # A_possible_na = [(0, 0)] if not A.collidable else World.NAV_ACTIONS
        # B_possible_na = [(0, 0)] if not B.collidable else World.NAV_ACTIONS

        # bound = None
        # for A_na, B_na in product(A_possible_na, B_possible_na):
        #     if len(agent_locs) == 1:
        #         try:
        #             bound_1 = nx.shortest_path_length(
        #                     self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
        #             bound_2 = nx.shortest_path_length(
        #                     self.reachability_graph, (A_loc, A_na), (B_loc, B_na))
        #         except:
        #             continue
        #         bound = bound_1 + bound_2 - 1

        #     elif len(agent_locs) == 2:
        #         # Try to calculate the distances between agents and Objects A and B.
        #         # Distance between Agent 1 <> Object A.
        #         try:
        #             bound_1_to_A = nx.shortest_path_length(
        #                     self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
        #         except:
        #             bound_1_to_A = self.perimeter
        #         # Distance between Agent 2 <> Object A.
        #         try:
        #             bound_2_to_A = nx.shortest_path_length(
        #                     self.reachability_graph, (agent_locs[1], (0, 0)), (A_loc, A_na))
        #         except:
        #             bound_2_to_A = self.perimeter

        #         # Take the agent that's the closest to Object A.
        #         min_bound_to_A = min(bound_1_to_A, bound_2_to_A)

        #         # Distance between the agents.
        #         bound_between_agents = manhattan_dist(A_loc, B_loc)

        #         # Distance between Agent 1 <> Object B.
        #         try:
        #             bound_1_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[0], (0, 0)), (B_loc, B_na))
        #         except:
        #             bound_1_to_B = self.perimeter

        #         # Distance between Agent 2 <> Object B.
        #         try:
        #             bound_2_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[1], (0, 0)), (B_loc, B_na))
        #         except:
        #             bound_2_to_B = self.perimeter

        #         # Take the agent that's the closest to Object B.
        #         min_bound_to_B = min(bound_1_to_B, bound_2_to_B)

        #         # For chop or deliver, must bring A to B.
        #         if subtask_name in ['Chop', 'Deliver']:
        #             bound = min_bound_to_A + bound_between_agents - 1
        #         # For merge, agents can separately go to A and B and then meet in the middle.
        #         elif subtask_name == 'Merge':
        #             min_bound_to_A, min_bound_to_B = self.check_bound(
        #                     min_bound_to_A=min_bound_to_A,
        #                     min_bound_to_B=min_bound_to_B,
        #                     bound_1_to_A=bound_1_to_A,
        #                     bound_2_to_A=bound_2_to_A,
        #                     bound_1_to_B=bound_1_to_B,
        #                     bound_2_to_B=bound_2_to_B
        #                     )
        #             bound = max(min_bound_to_A, min_bound_to_B) + (bound_between_agents - 1)/2

        #     if bound is not None and lower_bound > bound:
        #         lower_bound = bound

        # return max(1, lower_bound)
        return self.get_distance(A_loc, B_loc)

    def check_bound(
        self,
        min_bound_to_A,
        min_bound_to_B,
        bound_1_to_A,
        bound_2_to_A,
        bound_1_to_B,
        bound_2_to_B,
    ):
        # Checking for whether it's the same agent that does the subtask.
        if (bound_1_to_A == min_bound_to_A and bound_1_to_B == min_bound_to_B) or (
            bound_2_to_A == min_bound_to_A and bound_2_to_B == min_bound_to_B
        ):
            return 2 * min_bound_to_A, 2 * min_bound_to_B
        return min_bound_to_A, min_bound_to_B

    def is_occupied(self, location):
        o = list(
            filter(
                lambda obj: obj.location == location
                and isinstance(obj, Object)
                and not (obj.is_held),
                self.get_object_list(),
            )
        )
        if o:
            return True
        return False

    def clear_object(self, position):
        """Clears object @ position in self.rep and replaces it with an empty space"""
        x, y = position
        self.rep[y][x] = " "

    def clear_all(self):
        self.rep = []

    def add_object(self, object_, position):
        x, y = position
        self.rep[y][x] = str(object_)

    def insert(self, obj, toDefault=False):
        self.objects.setdefault(obj.name, []).append(obj)
        if toDefault:
            self.default_objects.setdefault(obj.name, []).append(obj)

    def remove(self, obj):
        num_objs = len(self.objects[obj.name])
        index = None
        for i in range(num_objs):
            if self.objects[obj.name][i].location == obj.location:
                index = i
        assert index is not None, f"Could not find {obj.name}!"
        self.objects[obj.name].pop(index)
        assert (
            len(self.objects[obj.name]) < num_objs
        ), f"Nothing from {obj.name} was removed from world.objects"

    def remove_order(self, delivered_name, t):
        """Marks an order of given recipe name as completed from the world"""
        # find the earliest added order for the given recipe and mark it as delivered
        delivered_obj = self.objects[delivered_name][0]
        pending_orders: Order = sorted(
            filter(
                lambda o: (not o.delivered)
                and o.recipe.full_plate_name == delivered_obj.name,
                self.objects.get("Order", []),
            ),
            key=lambda o: o.queued_at,
        )
        # assert pending_orders is not None and len(pending_orders) > 0, f"Could not find a pending {delivered_name} order !"
        if len(pending_orders):
            # mark this order as delivered, shift order queue coordinates to render from the left
            o: Order = pending_orders.pop(0)
            o.deliver(t)
            for order in self.objects["Order"]:
                if order == o: # delivered order pushed to end of list
                    o.location = len(self.objects["Order"]) - 1, o.location[1]
                else:
                    order.location = (order.location[0] - 1, order.location[1])

    def respawn_components(self, delivered_obj):
        """Respawns delivered components of the given recipe"""
        for component in delivered_obj.contents:
            original_locations = [o.spawn_location for o in self.default_objects[component.name]]
            counters = [
                c
                for c in self.get_object_list()
                if isinstance(c, Counter)
                and c.location in original_locations
            ][:1]
            for counter in counters:
                obj = Object(
                    location=counter.location, contents=RepToClass[component.rep]()
                )
                counter.acquire(obj)
                self.insert(obj)

    def get_object_list(self, names=[]):
        all_obs = []
        for o in self.objects.values():
            if len(names):
                all_obs += [_ for _ in o if _.name in names]
            else:
                all_obs += o
        return all_obs

    def get_default_dynamic_object_list(self):
        return self.get_dynamic_object_list(_from=self.default_objects)
    
    def get_dynamic_object_list(self, _from=None):
        all_obs = []
        if _from is None:
            _from = self.objects
            
        for key, val in _from.items():
            if key not in ("Agent-Counter", "Counter", "Floor", "Delivery", "Cutboard", "Order") and "Supply" not in key:
                all_obs.extend(tuple(_from[key]))
        return all_obs
    
    def get_dynamic_objects(self):
        """Get objects that can be moved (objects of same type are grouped on same index)."""
        objs = list()

        for key in sorted(self.objects.keys()):
            if key not in ("Agent-Counter", "Counter", "Floor", "Delivery", "Cutboard", "Order") and "Supply" not in key:
                objs.append(tuple(list(map(lambda o: o.get_repr(), self.objects[key]))))

        # Must return a tuple because this is going to get hashed.
        return tuple(objs)

    def get_dynamic_objects_flat(self):
        """Get objects that can be moved (objects of same type are ungrouped)."""
        objs = list()
        for items in self.get_dynamic_objects():
            objs.extend(items)
        # Must return a tuple because this is going to get hashed.
        return tuple(objs)
    
    def get_order_queue(self) -> list[Order]:
        """Get remaining orders in the queue."""
        return list(filter(lambda o: not o.delivered, self.objects["Order"]))

    def get_recipes(self):
        """Get all recipes in the level."""
        return set(map(lambda o: o.recipe, self.objects.get("Order")))
    
    def get_collidable_objects(self):
        return list(filter(lambda o: o.collidable, self.get_object_list()))

    def get_collidable_object_locations(self):
        return list(map(lambda o: o.location, self.get_collidable_objects()))

    def get_dynamic_object_locations(self):
        return list(map(lambda o: o.location, self.get_dynamic_objects_flat()))

    def is_collidable(self, location):
        return location in list(
            map(
                lambda o: o.location,
                list(filter(lambda o: o.collidable, self.get_object_list())),
            )
        )

    def get_object_locs(self, obj, is_held):
        if obj.name not in self.objects.keys():
            return []

        if isinstance(obj, Object):
            return list(
                map(
                    lambda o: o.location,
                    list(
                        filter(
                            lambda o: obj == o and o.is_held == is_held,
                            self.objects[obj.name],
                        )
                    ),
                )
            )
        else:
            return list(
                map(
                    lambda o: o.location,
                    list(filter(lambda o: obj == o, self.objects[obj.name])),
                )
            )

    def get_all_object_locs(self, obj):
        return list(
            set(
                self.get_object_locs(obj=obj, is_held=True)
                + self.get_object_locs(obj=obj, is_held=False)
            )
        )

    def get_object_at(self, location, desired_obj, find_held_objects):
        # Map obj => location => filter by location => return that object.
        all_objs = self.get_object_list()

        if desired_obj is None:
            objs = [obj for obj in all_objs if obj.location == location and isinstance(obj, Object) and (obj.is_held is find_held_objects)]
        else:
            objs = [obj for obj in all_objs if obj.name == desired_obj.name and obj.location == location and isinstance(obj, Object) and (obj.is_held is find_held_objects)]

        # assert (
        #     len(objs) == 1
        # ), f"looking for {desired_obj}, found {','.join(o.name for o in objs)} at {location}"

        return objs[0] if any(objs) else None
    
    def get_object_loc(self, desired_obj):
        all_objs = self.get_object_list()

        assert isinstance(desired_obj, Object)

        objs = [obj for obj in all_objs if obj.name == desired_obj.name and isinstance(obj, Object) and (not obj.is_held )]
        return objs[0] if any(objs) else None
    
    def get_object_at_loc(self, location): 
        all_objs = self.get_object_list()

        objs = [obj for obj in all_objs if obj.location == location]
        return objs[0] if any(objs) else None
    
    def get_objects_at_loc(self, location): 
        all_objs = self.get_object_list()

        objs = [obj for obj in all_objs if obj.location == location]
        return objs if any(objs) else None

    def get_gridsquare_at(self, location):
        gss = list(
            filter(
                lambda o: o.location == location and isinstance(o, GridSquare),
                self.get_object_list(),
            )
        )

        assert len(gss) == 1, f"{len(gss)} gridsquares at {location}: {gss}"
        return gss[0]

    def inbounds(self, location):
        """Correct location to be in bounds of world object."""
        x, y = location
        return min(max(x, 0), self.width - 1), min(
            max(y, 0), self.height - 2
        )  # -2 because last row is always orders queue.

    def get_neighbors(self, location):
        """Get all neighboring coordinates of a location."""
        x, y = location
        return [loc for loc in (
            (x - 1, y - 1),
            (x - 1, y),
            (x - 1, y + 1),
            (x, y - 1),
            (x, y + 1),
            (x + 1, y - 1),
            (x + 1, y),
            (x + 1, y + 1),
        ) if self.inbounds(loc)]

    def get_direct_neighbors(self, location):
        """Get all neighboring coordinates of a location."""
        x, y = location
        return [loc for loc in self.get_neighbors(location) if loc[0] == x or loc[1] == y]

    @lru_cache(4096)
    def get_distance(self, loc1, loc2):
        """Get Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
        
    def is_accessible(self, location):
        corners = (0, 0), (self.width - 1, 0), (0, self.height - 2), (self.width - 1, self.height - 2)
        # not a corner and is inbounds, check if blocked by neighbors
        if self.inbounds(location) and location not in corners:
            return any(isinstance(n, Floor) for n in filter(lambda o: o.location in self.get_direct_neighbors(location), self.get_object_list()))

        return False