from asyncio.log import logger
import logging
from utils.world import World
from utils.core import *
import numpy as np

logger = logging.getLogger(__name__)

ACTION_TYPE = ("Wait", "Move", "Get", "Chop", "Drop", "Merge", "Deliver","Teleport")
ActionRepr = namedtuple("ActionRepr", "agent agent_location action_type object")

is_dish = lambda recipes, x: any(r.full_plate_name == x.name for r in recipes)


def interact(agent, world: World, t=-1, play=False):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """
    # agent does nothing
    if agent.action == (0, 0):
        return ActionRepr(agent.name, agent.location, "Wait", agent.location, agent.holding)
    

    #if action is holding nothing
    if agent.holding is None: 
        # Action is Get Tomato
        if agent.action == (-1,0):
            pickFreshIngredient(agent,world,'Tomato')
            interaction = ActionRepr(agent.name, agent.location, "Get", agent.holding)

        elif agent.action == (1,0): 
            #Action is Get Lettuce
            pickFreshIngredient(agent,world,'Lettuce')
            interaction = ActionRepr(agent.name, agent.location, "Get", agent.holding)

    #if agent is holding something
    elif agent.holding is not None: 
        if agent.action == (0,-1) and agent.holding.needs_chopped():
            chopIngredient(agent,world)
            interaction = ActionRepr(agent.name, agent.location, "Chop", agent.holding)
        elif agent.action == (0,1) and agent.holding.is_chopped:
            mergeWithPlateAndIngredient(agent,world,agent.holding.name)
            interaction = ActionRepr(agent.name, agent.location, "Merge", agent.holding)
    
    if agent.action == (2,0):
        deliverOrder(agent,world)
        interaction = ActionRepr(agent.name, agent.location, "Deliver", agent.holding)


    return interaction

def pickFreshIngredient(agent,world, ingredient:str):
    test = [elm for elm in world.get_object_list() if isinstance(elm, Object)]
    test_list = [obj for obj in test if obj.name==ingredient and not obj.is_chopped() and not obj.is_held]
    tomato_location = test_list[0].location
    
    #interaction = ActionRepr(agent.name, agent.location, "Move", neighbour_floor, agent.holding)

    gs: GridSquare = world.get_gridsquare_at(tomato_location)
    obj = world.get_object_at(gs.location, None, find_held_objects = False)
    if obj is None:
        return
    
    if not obj.is_held:
        neighbours = world.get_direct_neighbors(tomato_location)
        floors = world.objects['Floor']
        list= [flr for flr in floors if flr.location in neighbours]
        neighbour_floor = [flr for flr in list if isinstance(flr, Floor)][0].location
        agent.move_to(neighbour_floor)
        gs.release()
        agent.acquire(obj)
    world.respawn_components(obj)
    interaction = ActionRepr(agent.name, agent.location, "Get", gs.location, agent.holding)

def chopIngredient(agent,world):
    test = [elm for elm in world.get_object_list() if (isinstance(elm, Cutboard) or elm.name == 'Cutboard') and elm.holding is None]
    neighbours = world.get_direct_neighbors(test[0].location)
    floors = world.objects['Floor']
    list= [flr for flr in floors if flr.location in neighbours]
    neighbour_floor = [flr for flr in list if isinstance(flr, Floor)]
    agent.move_to(neighbour_floor[0].location)
    obj = agent.holding
    gs: GridSquare = world.get_gridsquare_at(test[0].location)
    gs.acquire(obj= obj)
    agent.release()
    #time.sleep(2)
    obj.chop()
    gs.release()
    agent.acquire(obj)

def mergeWithPlate(agent,world):
    test = [elm for elm in world.get_object_list() if isinstance(elm, Object) and elm.is_plate()]
    plate_location = test[0].location

    gs: GridSquare = world.get_gridsquare_at(plate_location)
    obj = world.get_object_at(plate_location, None, find_held_objects = False)

    if mergeable(agent.holding, obj):
        neighbour_floor= get_direct_neighbour_floor(world,plate_location)
        agent.move_to(neighbour_floor)
        world.remove(obj)
        o = gs.release() # counter is holding object
        world.remove(agent.holding)
        agent.acquire(obj)
        world.insert(agent.holding)
        interaction = ActionRepr(agent.name, agent.location, "Merge", gs.location, agent.holding)
                # if playable version, merge onto counter first
        gs.acquire(agent.holding)
        agent.release()

def mergeWithPlateAndIngredient(agent, world, ingredient: str):
    plates = [elm for elm in world.get_object_list() if isinstance(elm, Object) and elm.name == 'Plate']
    empty_plate_location = plates[0].location

    gs: GridSquare = world.get_gridsquare_at(empty_plate_location)
    empty_plate = world.get_object_at(empty_plate_location, None, find_held_objects = False)

    if mergeable(agent.holding, empty_plate):
        # neighbours = world.get_direct_neighbors(empty_plate_location)
        # floors = world.objects['Floor']
        # list= [flr for flr in floors if flr.location in neighbours]
        # neighbour_floor = [flr for flr in list if isinstance(flr, Floor)][0].location
        neighbour_floor = get_direct_neighbour_floor(world,empty_plate_location)
        agent.move_to(neighbour_floor)

        world.remove(empty_plate)
        o = gs.release() # counter is holding object
        world.remove(agent.holding)
        agent.acquire(empty_plate)
        world.insert(agent.holding)
        interaction = ActionRepr(agent.name, agent.location, "Merge", gs.location, agent.holding)
                # if playable version, merge onto counter first

        useable_random_counter = get_useable_counter(world)
        useable_counter_floor_loc = get_direct_neighbour_floor(world,useable_random_counter.location)
        agent.move_to(useable_counter_floor_loc)
        gs_counter: GridSquare = world.get_gridsquare_at(useable_random_counter.location)
        gs_counter.acquire(agent.holding)
        agent.release()

def deliverOrder(agent,world):
        
    deliverable_orders = [obj for obj in world.get_dynamic_object_list() if obj.is_deliverable() and is_dish(world.get_recipes(), obj)]
    
    delivery_location = world.objects.get('Delivery')[0].location

    if(len(deliverable_orders) > 0):
        order_location = deliverable_orders[0].location
        gs: GridSquare = world.get_gridsquare_at(order_location)
        obj = world.get_object_at(order_location, None, find_held_objects = False)

        order_floor = get_direct_neighbour_floor(world= world, obj_location = order_location) 
        agent.move_to(order_floor)
        agent.acquire(obj)
        gs.release()

        delivery_floor= get_direct_neighbour_floor(world,delivery_location)

        agent.move_to(delivery_floor)
        gs_delivery: GridSquare = world.get_gridsquare_at(delivery_location)
        agent.release()
        gs_delivery.acquire(obj)

        world.remove_order(obj.name, 2)
        # update env, world and agent orders here, respawn used components and remove delivered object
        world.respawn_components(obj)
        world.remove(obj)

def get_direct_neighbour_floor(world,obj_location):
        neighbours = world.get_direct_neighbors(obj_location)
        floors = world.objects['Floor']
        list= [flr for flr in floors if flr.location in neighbours]
        neighbour_floor = None

        for flr in list :
            if isinstance(flr, Floor):
                neighbour_floor = flr.location
                break

        return neighbour_floor

def get_useable_counter(world):
    counters = world.objects['Counter']

    useable_counters = [counter for counter in counters if get_direct_neighbour_floor(world, counter.location) is not None and counter.holding is None]

    import random

    random_counter = random.choice(useable_counters)

    return random_counter



def simulated_interact(agent, world, play=False):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor): #and gs.holding is None:
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding
            if obj.is_deliverable():
                gs.acquire(obj)
                agent.release()
                #print('\nDelivered {}!'.format(obj.full_name))

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj):
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                if play:
                    gs.acquire(agent.holding)
                    agent.release()


        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not play:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
            else:
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                assert world.get_object_at(gs.location, obj, find_held_objects=False).is_held == False, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and play:
                obj.chop()
            else:
                gs.release()
                agent.acquire(obj)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass
