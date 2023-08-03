
from asyncio.log import logger
import logging

from functools import lru_cache

logger = logging.getLogger(__name__)

class RewardHelpers:

    TIME_PENALTY = 0.04
    
    def __init__(self):
        pass

    @staticmethod
    def compute_rewards(successful, failed, step, agents, stations, dynamic_objects, orders, current_state: dict, previous_state: dict, previous_previous_state: dict):
        if successful:
            return RewardHelpers.normalize_rewards(RewardHelpers.compute_success_reward(len(agents), step))

        if failed:
            return RewardHelpers.normalize_rewards(RewardHelpers.compute_failure_reward(len(agents)))
        
        rewards = []

        plates = {o.location for o in dynamic_objects if o.is_plate()}
        to_prep = {o.location for o in dynamic_objects if o.needs_chopped()}
        to_merge_or_deliver = {o.location for o in dynamic_objects if o.is_merged()}
        
        to_deliver = {o for o in orders if not o.delivered}
        prep_stations = {o.location for o in stations if o.name in ['Cutboard', 'CuttingBoard', 'Stove', 'Grill']}
        delivery_stations = {o.location for o in stations if o.name in ['Delivery']}
        
        get_closest = lambda loc, loc_list: min([1e4, *[RewardHelpers.get_distance(loc, x) for x in loc_list]])

        recipes = {_.recipe for _ in orders}
        # to check if an object is fully ready
        is_dish = lambda x: any(r.full_plate_name == x.name for r in recipes)
        
        is_held_by_another = lambda me, x: any(x == a.location for a in agents if a.name != me.name)

        is_deadlocked = not any(plates) and not any(is_dish(m) for m in dynamic_objects)  # all plates have items on them and no items are ready to be delivered
        
        for agent in agents:
            REWARD = 0
            PENALTY = step * (RewardHelpers.TIME_PENALTY / len(agents)) # time step decay

            curr = current_state[agent.name]
            prev = previous_state[agent.name]
            prev_prev = previous_previous_state[agent.name]
            
            closest_plate = get_closest(curr.location, plates)
            closest_to_prep = get_closest(curr.location, to_prep)
            closest_to_merge = get_closest(curr.location, to_merge_or_deliver)
            closest_dish = get_closest(curr.location, [dish for dish in to_merge_or_deliver if dish in plates])

            prev_closest_plate = get_closest(prev.location, plates)
            prev_closest_to_prep = get_closest(prev.location, to_prep)
            prev_closest_to_merge = get_closest(prev.location, to_merge_or_deliver)
            prev_closest_dish = get_closest(prev.location, [dish for dish in to_merge_or_deliver if dish in plates])

            prev_closest_prep_station = get_closest(prev.location, prep_stations)
            prev_closest_delivery_station = get_closest(prev.location, delivery_stations)
            
            action = curr.action_type

            # direct action rewards
            if action == 'Deliver':
                REWARD += 50
            elif action == 'Merge':
                REWARD += 20
            elif action == 'Chop':
                REWARD += 10
            elif action == 'Get' and curr.location != prev_prev.location: # or action == 'Drop':
                REWARD += 3
            elif action == 'Drop' and curr.location != prev_prev.location:
                REWARD += 1
            
            # if dropped then should've brought item closer to target location (dish, prep station, delivery station)
            # same if moving with an item
            elif action == 'Drop' or (action == 'Move' and curr.holding):
                # if holding get reward based on recipe progress ONLY first time per pickup
                if curr.holding and curr.holding != prev.holding and curr.holding != prev_prev.holding:
                    recipe_progress = 1 + agent.holding.get_state_index()
                    REWARD += (2 * recipe_progress)
                # else if dropped, get reward based on how close was dropped to correct location
                item = prev.holding if action == 'Drop' else curr.holding

                # plates should be brought closer to prepped food
                if item.is_plate():
                    not_plates = [_ for _ in to_merge_or_deliver if _ not in plates]
                    REWARD += (get_closest(item.location, not_plates) < prev_closest_to_merge)

                # fresh food should be brought closer to prep stations
                elif item.needs_chopped():
                    REWARD += (get_closest(item.location, prep_stations) < prev_closest_prep_station)

                # chopped food should be brought closer to other chopped food or plates
                elif item.is_chopped():
                    other_prepped = [_ for _ in to_merge_or_deliver if _ != item.location]
                    REWARD += sum([get_closest(item.location, other_prepped) < prev_closest_to_merge, get_closest(item.location, plates) < prev_closest_plate]) 

                # merged food should be brought closer to plate if missing plate or other mergable food if missing other mergable food
                elif item.is_merged() and not is_dish(item):
                    needs_plate = any(plates) and not item.has_plate()
                    needs_other_food = any(to_merge_or_deliver)
                    logger.debug(f"MERGED: {item.name, item.location, needs_plate, needs_other_food}")
                    REWARD += (get_closest(item.location, plates) < prev_closest_plate) if needs_plate else (get_closest(item.location, to_merge_or_deliver) < prev_closest_to_merge) if needs_other_food else 0
                    
                # merged food should be brought closer to delivery stations
                elif item.is_deliverable() or is_dish(item):
                    logger.debug(f"DELIVERABLE: {item.name, item.location, prev_closest_delivery_station, get_closest(item.location, delivery_stations)}")
                    REWARD += 3 * (get_closest(item.location, delivery_stations) < prev_closest_delivery_station)
            # if not holding, must be moving closer to dish, prepped food, fresh food or plate
            elif action == 'Move' and not agent.holding:
                any_dish = any(is_dish(_) for _ in dynamic_objects if not is_held_by_another(agent, _.location))
                any_needs_prep = any(_ for _ in to_prep if not is_held_by_another(agent, _))
                any_needs_merge = any(_ for _ in to_merge_or_deliver if not is_held_by_another(agent, _))
                any_plate = any(_ for _ in plates if not is_held_by_another(agent, _))
                logger.debug(f"MOVING: {any_dish, any_needs_prep, any_needs_merge, any_plate}")
                REWARD += 2 * ((closest_dish < prev_closest_dish) if any_dish else (closest_to_prep < prev_closest_to_prep) if any_needs_prep else (closest_to_merge < prev_closest_to_merge) if any_needs_merge else (closest_plate < prev_closest_plate) if any_plate else 0)

            # handover rewards
            REWARD += 0.5 * curr.handed_over


            # action cost
            PENALTY += 0.05 * ((action == 'Wait') + (action == 'Move' and not curr.holding))    
            # bad action penalties
            PENALTY += curr.invalid_actor
            PENALTY += 0.5 * curr.collided
            PENALTY += 0.5 * (curr.shuffled + curr.location_repeater + curr.holding_repeater)    

            # penalty for dropping on wrong station
            if action == 'Drop' and not is_dish(prev.holding):
                if prev.holding.location in prep_stations or prev.holding.location in delivery_stations:
                    logger.debug(f"WRONG STATION: {prev.holding.name, prev.holding.location, curr.location}")
                    PENALTY += 3
            elif action == 'Drop' and is_dish(prev.holding):
                if prev.holding.location in prep_stations:
                    logger.debug(f"WRONG STATION: {prev.holding.name, prev.holding.location, curr.location}")
                    PENALTY += 5

            if is_deadlocked:
                logger.debug(f"DEADLOCKED")
                PENALTY += 10


            rewards.append(REWARD - PENALTY)
            
            
        return RewardHelpers.normalize_rewards(rewards)

    @staticmethod
    def compute_success_reward(num_agents, timestep):
        return [50 - (timestep * (RewardHelpers.TIME_PENALTY / num_agents))] * num_agents # reward for completing all deliveries

    @staticmethod
    def compute_failure_reward(num_agents):
        return [-50] * num_agents

    @staticmethod
    def normalize_rewards(rewards, lower=-1, upper=1, min_val=-50, max_val=50):
        norm_value = lambda v: (v - min_val) / (max_val - min_val) * (upper - lower) + lower

        return [norm_value(r) for r in rewards]

    @staticmethod
    @lru_cache(4096)
    def get_distance(loc1, loc2):
        """Get Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])