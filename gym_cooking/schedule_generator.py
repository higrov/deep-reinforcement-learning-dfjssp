from pymongo import MongoClient
from recipe_planner.recipe import SimpleLettuce, SimpleTomato, OnionSalad, Salad
from utils.core import Order

class ScheduleGenerator: 
    mappings = {'C0': SimpleLettuce,
                'C1': SimpleTomato,
                'C2': Salad,
                'C3': OnionSalad}
    def __init__(self) -> None:

        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client['rcll']
        self.collection = self.db['game_report']

        self.data = self.collection.find({})

        self.rcll_schedule = dict()
        i = 0
        for doc in self.data:
            self.rcll_schedule[i] = [{'complexity': element['complexity'], 'activate_at': element['activate_at'], 'delivery_period': element['delivery_period'] }
              for element in doc['orders']]
            i +=1

        self.overcooked_schedule = dict()

        print(self.rcll_schedule[0])
    
    def generateSchedule(self):
        for key in self.rcll_schedule.keys():
            self.overcooked_schedule[key] = self.mapToOvercooked(self.rcll_schedule[key])
        return self.overcooked_schedule

    
    def mapToOvercooked(self, rcll_schedule):
        overcooked_schedule = []
        for element in rcll_schedule:
            overcooked_schedule.append(Order(recipe =self.mappings[element['complexity']](), 
                                             location=(0,7), 
                                             queued_at=element['activate_at'], 
                                             delivery_window= element['delivery_period']))            
        return overcooked_schedule


