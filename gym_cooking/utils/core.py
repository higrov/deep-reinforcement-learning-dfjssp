# recipe planning
import math
#import recipe_planner
from recipe_planner import Recipe
import recipe_planner.utils as recipe
import time

# helpers
import numpy as np
import copy
import random
from termcolor import colored as color
from itertools import combinations
from collections import namedtuple

# -----------------------------------------------------------
# GRIDSQUARES
# -----------------------------------------------------------
GridSquareRepr = namedtuple("GridSquareRepr", "name location holding")
OrderRepr = namedtuple("OrderRepr", "name recipe_name delivered queued_at delivered_at")

class Rep:
    FLOOR = " "
    COUNTER = "-"
    CUTBOARD = "/"
    DELIVERY = "*"
    TOMATO = "t"
    LETTUCE = "l"
    ONION = "o"
    PLATE = "p"
    ORDER = "O"


class GridSquare:
    def __init__(self, name, location):
        self.name = name
        self.location = location  # (x, y) tuple
        self.holding: Object = None
        self.color = "white"
        self.collidable = True  # cannot go through
        self.dynamic = False  # cannot move around
        self.hidden = False

    def __str__(self):
        return color(self.rep, self.color)

    def __eq__(self, o):
        return isinstance(o, GridSquare) and self.name == o.name

    def __copy__(self):
        gs = type(self)(self.location)
        gs.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            gs.holding = copy.copy(self.holding)
        return gs
    
    def acquire(self, obj):
        obj.location = self.location
        self.holding = obj

    def release(self):
        temp = self.holding
        self.holding = None
        return temp
    

class Floor(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Floor", location)
        self.color = None
        self.rep = Rep.FLOOR
        self.collidable = False

    def __eq__(self, other):
        return GridSquare.__eq__(self, other)

    def __hash__(self):
        return GridSquare.__hash__(self)


class Order(GridSquare):
    def __init__(self, recipe: Recipe, location, t):
        GridSquare.__init__(self, recipe.name, location)
        self.recipe = recipe
        self.rep = recipe.rep  # instead of Rep.ORDER
        self.full_name = recipe.delivery_name
        self.queued_at = t
        self.delivered_at = None
        self.delivered = False
        self.duration = -1
        self.dynamic = False
        self.collidable = False

    def reset(self, t):
        self.queued_at = t
        self.delievered_at = None
        self.duration = -1
        self.delivered = False

    def __eq__(self, other):
        return (
            GridSquare.__eq__(self, other)
            and self.name == other.name
            and self.rep == other.rep
            and self.delivered == other.delivered
            and self.queued_at == other.queued_at
            and self.delivered_at == other.delivered_at)

    def __hash__(self):
        return hash((self.full_name, self.delivered, self.queued_at, self.duration))

    def get_repr(self):
        return OrderRepr(self.name, self.full_name, self.delivered, self.queued_at, self.delivered_at)

    def deliver(self, t):
        self.delivered_at = t
        self.duration = self.delivered_at - self.queued_at
        self.delivered = True

    
class Counter(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Counter", location)
        self.rep = Rep.COUNTER

    def __eq__(self, other):
        return GridSquare.__eq__(self, other)

    def __hash__(self):
        return GridSquare.__hash__(self)

    def update_holding_location(self):
        ''' If counter location changed (for eg. by randomizer), update location of object that was on counter'''
        if self.holding:
            self.holding.location = self.location
            self.holding.spawn_location = self.location

    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding=self.holding)

    def swap_holding(self, other_counter: 'Counter'):
        ''' Swap the holding objects of two counters.'''
        holding = self.release()
        other = other_counter.release()
        if holding:
            other_counter.acquire(holding)
            other_counter.update_holding_location()
        if other:
            self.acquire(other)
            self.update_holding_location() # to update spawn location
            
class AgentCounter(Counter):
    def __init__(self, location):
        GridSquare.__init__(self, "Agent-Counter", location)
        self.rep = Rep.COUNTER
        self.collidable = True

    def __eq__(self, other):
        return Counter.__eq__(self, other)

    def __hash__(self):
        return Counter.__hash__(self)

    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding=None)


class Cutboard(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Cutboard", location)
        self.rep = Rep.CUTBOARD
        self.collidable = True

    def __eq__(self, other):
        return GridSquare.__eq__(self, other)

    def __hash__(self):
        return GridSquare.__hash__(self)

    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding=self.holding)


class Delivery(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Delivery", location)
        self.rep = Rep.DELIVERY
        self.holding = []

    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)

    def release(self):
        if self.holding:
            return self.holding.pop()
        else:
            return None

    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding=self.holding)

    def __eq__(self, other):
        return GridSquare.__eq__(self, other)

    def __hash__(self):
        return GridSquare.__hash__(self)


# -----------------------------------------------------------
# OBJECTS
# -----------------------------------------------------------
# Objects are wrappers around foods items, plates, and any combination of them

ObjectRepr = namedtuple("ObjectRepr", "name location spawn_location is_held")


class Object:
    def __init__(self, location, contents):
        self.spawn_location = location  # the original location, respawned here as well
        self.location = location
        self.contents = contents if isinstance(contents, list) else [contents]
        self.is_held = False
        self.update_names()
        self.collidable = False
        self.dynamic = False

    def __str__(self):
        res = "-".join(
            list(map(lambda x: str(x), sorted(self.contents, key=lambda i: i.name)))
        )
        return res

    def __eq__(self, other):
        # check that content is the same and in the same state(s)
        return (
            isinstance(other, Object)
            and self.name == other.name
            and len(self.contents) == len(other.contents)
            and self.full_name == other.full_name
        )
        # all([i == j for i, j in zip(sorted(self.contents, key=lambda x: x.name),
        #                             sorted(other.contents, key=lambda x: x.name))])

    def __copy__(self):
        new = Object(self.location, self.contents[0])
        new.__dict__ = self.__dict__.copy()
        new.contents = [copy.copy(c) for c in self.contents]
        return new

    def get_repr(self):
        return ObjectRepr(
            name=self.full_name,
            location=self.location,
            spawn_location=self.spawn_location,
            is_held=self.is_held,
        )

    def update_names(self):
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.name)
        self.name = "-".join([c.name for c in sorted_contents])
        self.full_name = "-".join([c.full_name for c in sorted_contents])
        
    def contains(self, c_name):
        return c_name in list(map(lambda c: c.name, self.contents))

    def needs_chopped(self):
        if len(self.contents) > 1:
            return False
        return self.contents[0].needs_chopped()

    def is_chopped(self):
        for c in self.contents:
            if isinstance(c, Plate) or c.get_state() != "Chopped":
                return False
        return True

    def chop(self):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.contents[0].update_state()
        assert not (self.needs_chopped())
        self.update_names()

    def merge(self, obj):
        if isinstance(obj, Object):
            # move obj's contents into this instance
            for i in obj.contents:
                self.contents.append(i)
        elif not (isinstance(obj, Food) or isinstance(obj, Plate)):
            raise ValueError("Incorrect merge object: {}".format(obj))
        else:
            self.contents.append(obj)
        self.update_names()

    def unmerge(self, full_name):
        # remove by full_name, assumming all unique contents
        matching = list(filter(lambda c: c.full_name == full_name, self.contents))
        self.contents.remove(matching[0])
        self.update_names()
        return matching[0]

    def is_merged(self):
        return len(self.contents) > 1

    def is_deliverable(self):
        # must be merged, and all contents must be Plates or Foods in done state
        for c in self.contents:
            if not (isinstance(c, Plate) or (isinstance(c, Food) and c.done())):
                return False
        return (
            any(c for c in self.contents if isinstance(c, Plate)) and self.is_merged()
        )
        
    def is_plate(self):
        return len(self.contents) == 1 and isinstance(self.contents[0], Plate)

    def has_plate(self):
        return any(isinstance(c, Plate) for c in self.contents)
    
    def get_state_index(self, is_ready=True):
        return [self.is_plate(), self.needs_chopped(), self.is_chopped(), self.is_merged(), not self.has_plate(), self.is_deliverable() and is_ready].index(True)
    
    
def mergeable(obj1, obj2):
    # query whether two objects are mergeable
    contents = obj1.contents + obj2.contents
    # check that there is at most one plate
    try:
        contents.remove(Plate())
    except:
        pass  # do nothing, 1 plate is ok
    finally:
        try:
            contents.remove(Plate())
        except:
            for c in contents:  # everything else must be in last state
                if not c.done():
                    return False
        else:
            return False  # more than 1 plate
    return True


# -----------------------------------------------------------


class FoodState:
    FRESH = globals()["recipe"].__dict__["Fresh"]
    CHOPPED = globals()["recipe"].__dict__["Chopped"]

class FoodSequence:
    FRESH = [FoodState.FRESH]
    FRESH_CHOPPED = [FoodState.FRESH, FoodState.CHOPPED]


# -----------------------------------------------------------
class Food:
    def __init__(self):
        self.state = self.state_seq[self.state_index]
        self.movable = False
        self.color = self._set_color()
        self.update_names()

    def __str__(self):
        return color(self.rep, self.color)

    # def __hash__(self):
    #     return hash((self.state, self.name))

    def __eq__(self, other):
        return isinstance(other, Food) and self.get_state() == other.get_state()

    def __len__(self):
        return 1  # one food unit

    def set_state(self, state):
        assert (
            state in self.state_seq
        ), f"Desired state {state} does not exist for the food with sequence {self.state_seq}"
        self.state_index = self.state_seq.index(state)
        self.state = state
        self.update_names()

    def get_state(self):
        return self.state.__name__

    def update_names(self):
        self.full_name = f"{self.get_state()}{self.name}"

    def needs_chopped(self):
        return (
            self.state_seq[(self.state_index + 1) % len(self.state_seq)]
            == FoodState.CHOPPED
        )

    def done(self):
        return (self.state_index % len(self.state_seq)) == len(self.state_seq) - 1

    def update_state(self):
        self.state_index += 1
        assert 0 <= self.state_index and self.state_index < len(
            self.state_seq
        ), "State index is out of bounds for its state sequence"
        self.state = self.state_seq[self.state_index]
        self.update_names()

    def _set_color(self):
        pass


class Tomato(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index  # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = "t"
        self.name = "Tomato"
        Food.__init__(self)

    def __hash__(self):
        return Food.__hash__(self)

    def __eq__(self, other):
        return Food.__eq__(self, other)


class Lettuce(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index  # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = "l"
        self.name = "Lettuce"
        Food.__init__(self)

    def __eq__(self, other):
        return Food.__eq__(self, other)

    def __hash__(self):
        return Food.__hash__(self)


class Onion(Food):
    def __init__(self, state_index=0):
        self.state_index = state_index  # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = "o"
        self.name = "Onion"
        Food.__init__(self)

    def __eq__(self, other):
        return Food.__eq__(self, other)

    def __hash__(self):
        return Food.__hash__(self)


# -----------------------------------------------------------


class Plate:
    def __init__(self):
        self.rep = "p"
        self.name = "Plate"
        self.full_name = "Plate"
        self.color = "white"

    def __str__(self):
        return color(self.rep, self.color)

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        return isinstance(other, Plate)

    def __copy__(self):
        return Plate()
    
    def needs_chopped(self):
        return False


# -----------------------------------------------------------
# PARSING
# -----------------------------------------------------------
RepToClass = {
    Rep.FLOOR: globals()["Floor"],
    Rep.COUNTER: globals()["Counter"],
    Rep.CUTBOARD: globals()["Cutboard"],
    Rep.DELIVERY: globals()["Delivery"],
    Rep.TOMATO: globals()["Tomato"],
    Rep.LETTUCE: globals()["Lettuce"],
    Rep.ONION: globals()["Onion"],
    Rep.PLATE: globals()["Plate"],
    Rep.ORDER: globals()["Order"],
}