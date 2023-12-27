from asyncio.log import logger
import asyncio
import logging
from utils.world import World
from utils.core import *
import numpy as np
# from recipe_planner.utils import Get, Chop, Merge, Deliver, Grill

logger = logging.getLogger(__name__)

ACTION_TYPE = ("Get", "Chop", "Merge", "Deliver")
ActionRepr = namedtuple("ActionRepr", "agent agent_location action_type object")

is_dish = lambda recipes, x: any(r.full_plate_name == x.name for r in recipes)


def interact(agent, world: World, t=-1, play=False):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """
    interaction = ActionRepr(agent.name, agent.location, "Wait", agent.holding)

    interaction = concept_interact(agent,world,agent.action)

    return interaction


def concept_interact(agent, world: World, action, t=-1):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """
    # agent does nothing

    if(action.name == 'Get'):
        if action.args[0] == 'Tomato' or action.args[0] == 'Lettuce' or action.args[0] == 'Meat' or action.args[0] == 'Bun' : 
            pickFreshIngredient(agent, world, action.args[0])
    
    elif(action.name == 'Chop'):
        #assert agent.holding is not None
        ing = action.args[0]
        get_fresh_ingredient(agent,ing,world)
        chopIngredient(agent,world)
    
    elif(action.name == 'Grill'):
        #assert agent.holding is not None
        ing = action.args[0]
        get_fresh_ingredient(agent,ing,world)
        grillIngredient(agent,world)
        brkpt = True

    elif(action.name == 'Merge'):
        #assert agent.holding is None
        obj1 = action.args[0]
        obj2 = action.args[1]
        merge(obj1,obj2,agent,world)

    elif(action.name == 'Deliver'):
        deliverable_order = next((obj for obj in world.get_dynamic_object_list() 
                              if obj.is_deliverable() and is_dish(world.get_recipes(), obj) and obj.name == action.args[0]), None)
        
        order_location = deliverable_order.location
        gs: GridSquare = world.get_gridsquare_at(order_location)
        obj = world.get_object_at(order_location, None, find_held_objects = False)

        order_floor = get_direct_neighbour_floor(world= world, obj_location = order_location) 
        agent.move_to(order_floor)
        agent.acquire(obj)
        gs.release()

        deliverOrder(agent,world,obj)

    interaction = ActionRepr(agent.name, agent.location, "Wait", agent.holding)

    return interaction

def pickFreshIngredient(agent,world, ingredient:str):
    object_list = [elm for elm in world.get_object_list() if isinstance(elm, Object)]
    ingredient_list = [obj for obj in object_list if obj.name==ingredient and not obj.is_chopped() and not obj.is_held]
    ingredient_location = ingredient_list[0].location
    
    gs: GridSquare = world.get_gridsquare_at(ingredient_location)
    obj = world.get_object_at(gs.location, None, find_held_objects = False)
    if obj is None:
        return
    
    if not obj.is_held:
        neighbour_floor = get_direct_neighbour_floor(world,ingredient_location)
        agent.move_to(neighbour_floor)
        gs.release()
        agent.acquire(obj)
        world.respawn_components(obj)

        useable_counter = get_useable_counter(world)
        neighbour_floor = get_direct_neighbour_floor(world,useable_counter.location)
        agent.move_to(neighbour_floor)
        gs: GridSquare = world.get_gridsquare_at(useable_counter.location)
        gs.acquire(obj= obj)
        agent.release()

def get_fresh_ingredient(agent,ingredient,world):
    object_list = [elm for elm in world.get_object_list() if isinstance(elm, Object)]
    ingredient_list = [obj for obj in object_list if obj.name==ingredient and not obj.is_chopped() and not obj.is_held]
    ingredient_location = ingredient_list[0].location
    
    gs: GridSquare = world.get_gridsquare_at(ingredient_location)
    obj = world.get_object_at(gs.location, None, find_held_objects = False)
    if obj is None:
        return
    
    if not obj.is_held:
        neighbour_floor = get_direct_neighbour_floor(world,ingredient_location)
        agent.move_to(neighbour_floor)
        gs.release()
        agent.acquire(obj)

def chopIngredient(agent,world: World):
    cutboard = get_available_cutboards(world)
    if cutboard is not None:
        neighbour_floor = get_direct_neighbour_floor(world,cutboard.location)
        agent.move_to(neighbour_floor)
        obj = agent.holding
        gs: GridSquare = world.get_gridsquare_at(cutboard.location)
        gs.acquire(obj= obj)
        agent.release()
        obj.chop()

        breakpt = True

        #mergeWithPlate(obj,world,plate)
        # agent.move_to(neighbour_floor)
        # gs.acquire(agent.holding)
        # agent.release()

def grillIngredient(agent,world: World):
    cutboard = get_available_grill(world)
    if cutboard is not None:
        neighbour_floor = get_direct_neighbour_floor(world,cutboard.location)
        agent.move_to(neighbour_floor)
        obj = agent.holding
        gs: GridSquare = world.get_gridsquare_at(cutboard.location)
        gs.acquire(obj= obj)
        agent.release()
        obj.grill()

def pickChoppedIngredient(agent, world: World, ingredient:str):
    plate_with_ing = get_plate_with_ingredient(world,ingredient)
    if plate_with_ing is not None and not plate_with_ing.is_held:
        gs: GridSquare = world.get_gridsquare_at(plate_with_ing.location)
        neighbour_floor = get_direct_neighbour_floor(world,plate_with_ing.location)
        agent.move_to(neighbour_floor)
        gs.release()
        agent.acquire(plate_with_ing)

def get_available_cutboards(world):
    cutboards = world.objects['Cutboard']
    empty_cutboards = [elm for elm in cutboards if elm.holding is None]

    available_cutboard = None

    for cutbrd in empty_cutboards:
        if isinstance(cutbrd, Cutboard):
            available_cutboard = cutbrd
            break
    
    return available_cutboard

def get_available_grill(world):
    grills = world.objects['Grill']
    empty_grills = [elm for elm in grills if elm.holding is None]

    available_grill = None

    for grill in empty_grills:
        if isinstance(grill, Grill):
            available_grill = grill
            break
    
    return available_grill

def mergeWithPlate(obj,world:World,empty_plate):
    gs: GridSquare = world.get_gridsquare_at(empty_plate.location)

    gs_obj :GridSquare = world.get_gridsquare_at(obj.location)

    if empty_plate is not None and mergeable(obj, empty_plate):
        o = gs.release() # counter is holding object
        world.remove(o)
        world.remove(obj)
        gs_obj.acquire(empty_plate)
        world.insert(gs_obj.holding)
        world.respawn_components(empty_plate)

def mergeWithPlateAndIngredient(agent, world, ingredient: str):
    plate = get_plate_with_ingredient(world, ingredient)

    gs: GridSquare = world.get_gridsquare_at(plate.location)
    empty_plate = world.get_object_at(plate.location, None, find_held_objects = False)
    if mergeable(agent.holding, empty_plate):
        neighbour_floor = get_direct_neighbour_floor(world,plate.location)
        agent.move_to(neighbour_floor)

        world.remove(empty_plate)
        o = gs.release() # counter is holding object
        world.remove(agent.holding)
        agent.acquire(empty_plate)
        world.insert(agent.holding)

        useable_random_counter = get_useable_counter(world)
        useable_counter_floor_loc = get_direct_neighbour_floor(world,useable_random_counter.location)
        agent.move_to(useable_counter_floor_loc)
        gs_counter: GridSquare = world.get_gridsquare_at(useable_random_counter.location)
        gs_counter.acquire(agent.holding)
        agent.release()

def merge(ingredient,plate, agent,world):
    if ingredient == 'Bun':
        ing_obj = next(x for x in world.objects[ingredient])
    elif ingredient == 'Meat':
        ing_obj = next((x for x in world.objects[ingredient] if x.is_grilled()), None)
    else:    
        ing_obj = next((x for x in world.objects[ingredient] if x.is_chopped()), None)

    plate_obj = next(plt for plt in world.objects[plate] if plt.name == plate)

    mergeWithPlate(ing_obj,world,plate_obj)

def get_available_plate(world:World):
    plates = world.objects['Plate']
    empty_plate= None

    for plate in plates:
        if(isinstance(plate, Object) and any(isinstance(plt, Plate) for plt in plate.contents)):
            empty_plate= plate
            break

    return empty_plate

def get_plate_with_ingredient(world: World, ingredient:str):
    objects = world.get_dynamic_object_list()

    merged_plates = [elm for elm in objects if len(elm.contents)>1 and 
                     any(ing.name == ingredient for ing in elm.contents)]

    plate_with_ingredient = None

    if len(merged_plates)>0:
        for plate in merged_plates:
            if not plate.is_held:
                plate_with_ingredient = plate

    return plate_with_ingredient

def deliverOrder(agent,world: World,obj):
    delivery_location = world.objects.get('Delivery')[0].location

    delivery_floor= get_direct_neighbour_floor(world,delivery_location)

    agent.move_to(delivery_floor)
    gs_delivery: GridSquare = world.get_gridsquare_at(delivery_location)
    agent.release()
    gs_delivery.acquire(obj)
    world.remove_order(obj.name, 2)
    # update env, world and agent orders here, respawn used components and remove delivered object
    #world.respawn_components(obj)
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
