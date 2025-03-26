import recipe_planner.utils as recipe

from itertools import combinations


class Recipe:
    def __init__(self, name, rep = 'R'):
        self.name = name
        self.rep = rep
        self.contents = []
        self.actions = []
        self.actions.append(recipe.Get('Plate'))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.rep == other.rep

    def __hash__(self):
        return hash(self.rep)
    
    def add_ingredient(self, item):
        from utils.core import FoodSequence
        self.contents.append(item)

        # always starts with FRESH
        self.actions.append(recipe.Get(item.name))

        if item.state_seq == FoodSequence.FRESH_CHOPPED:
            self.actions.append(recipe.Chop(item.name))
            self.actions.append(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))

    def add_goal(self):
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        self.contents_names = [c.name for c in self.contents]   # list of strings
        self.full_contents_names = [c.full_name for c in self.contents]   # list of strings
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        self.full_plate_name = '-'.join(sorted(self.contents_names + ['Plate'])) # string
        self.delivery_name = '-'.join(sorted(self.full_contents_names)) # name of graphic file
        self.goal = recipe.Delivered(self.full_plate_name)
        self.actions.append(recipe.Deliver(self.full_plate_name))

    def add_merge_actions(self):
        # should be general enough for any kind of salad / raw plated veggies

        # alphabetical, joined by dashes ex. Ingredient1-Ingredient2-Plate
        #self.full_name = '-'.join(sorted(self.contents + ['Plate']))

        # for any plural number of ingredients
        for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.append(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    if len(rem) == 1:
                        self.actions.append(recipe.Merge(item, rem_str,\
                            [recipe.Chopped(item), recipe.Chopped(rem_str)], None))
                        self.actions.append(recipe.Merge(rem_str, plate_str))
                        self.actions.append(recipe.Merge(item, rem_plate_str))
                    else:
                        self.actions.append(recipe.Merge(item, rem_str))
                        self.actions.append(recipe.Merge(plate_str, rem_str,\
                            [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                        self.actions.append(recipe.Merge(item, rem_plate_str))


    def takes_plate(self):
        return 'Plate' in self.full_plate_name

class SimpleTomato(Recipe):
    def __init__(self):
        from utils.core import  Tomato
        Recipe.__init__(self, 'Tomato', 'T')
        self.contents.append(Tomato(state_index=-1))
        self.actions.append(recipe.Get('Tomato'))
        self.actions.append(recipe.Chop('Tomato'))
        self.actions.append(recipe.Merge('Tomato','Plate'))
        self.add_goal()
        

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce', 'L')
        from utils.core import Lettuce
        self.contents.append(Lettuce(state_index=-1))
        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))
        self.add_goal()

class Salad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Salad', 'S')
        from utils.core import  Tomato,Lettuce
        self.contents.append(Tomato(state_index=-1))
        self.contents.append(Lettuce(state_index=-1))
        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))
        self.actions.append(recipe.Get('Tomato'))
        self.actions.append(recipe.Chop('Tomato'))
        self.actions.append(recipe.Merge('Tomato','Lettuce-Plate'))
        self.add_goal()

class OnionSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSalad', 'O')
        from utils.core import  Tomato,Lettuce,Onion
        self.contents.append(Tomato(state_index=-1))
        self.contents.append(Lettuce(state_index=-1))
        self.contents.append(Onion(state_index=-1))

        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))
        self.actions.append(recipe.Get('Tomato'))
        self.actions.append(recipe.Chop('Tomato'))
        self.actions.append(recipe.Merge('Tomato','Lettuce-Plate'))

        self.actions.append(recipe.Get('Onion'))
        self.actions.append(recipe.Chop('Onion'))
        self.actions.append(recipe.Merge('Onion','Lettuce-Plate-Tomato'))

        self.add_goal()

class SimpleBun(Recipe):
    def __init__(self):
        from utils.core import  Bun
        Recipe.__init__(self, 'Bun', 'b')
        self.contents.append(Bun(state_index=-1))
        self.actions.append(recipe.Get('Bun'))
        self.actions.append(recipe.Merge('Bun','Plate'))
        self.add_goal()

class BunLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'BunLettuce', 'bL')
        from utils.core import Lettuce, Bun
        self.contents.append(Bun(state_index=-1))
        self.contents.append(Lettuce(state_index=-1))

        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))
        self.actions.append(recipe.Get('Bun'))
        self.actions.append(recipe.Merge('Bun','Lettuce-Plate'))

        self.add_goal()

class BunLettuceTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'BunLettuceTomato', 'bLT')
        from utils.core import Lettuce, Bun, Tomato
        self.contents.append(Bun(state_index=-1))
        self.contents.append(Lettuce(state_index=-1))
        self.contents.append(Tomato(state_index=-1))

        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))

        self.actions.append(recipe.Get('Tomato'))
        self.actions.append(recipe.Chop('Tomato'))
        self.actions.append(recipe.Merge('Tomato','Lettuce-Plate'))

        self.actions.append(recipe.Get('Bun'))
        self.actions.append(recipe.Merge('Bun','Lettuce-Plate-Tomato'))
        
        self.add_goal()

class Burger(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Burger', 'B')
        from utils.core import Tomato,Lettuce, Bun, Meat
        self.contents.append(Bun(state_index=-1))
        self.contents.append(Tomato(state_index=-1))
        self.contents.append(Lettuce(state_index=-1))
        self.contents.append(Meat(state_index=-1))

        self.actions.append(recipe.Get('Lettuce'))
        self.actions.append(recipe.Chop('Lettuce'))
        self.actions.append(recipe.Merge('Lettuce','Plate'))

        self.actions.append(recipe.Get('Tomato'))
        self.actions.append(recipe.Chop('Tomato'))
        self.actions.append(recipe.Merge('Tomato','Lettuce-Plate'))
        
        self.actions.append(recipe.Get('Meat'))
        self.actions.append(recipe.Grill('Meat'))
        self.actions.append(recipe.Merge('Meat','Lettuce-Plate-Tomato'))

        self.actions.append(recipe.Get('Bun'))
        self.actions.append(recipe.Merge('Bun','Lettuce-Meat-Plate-Tomato'))

        self.add_goal()