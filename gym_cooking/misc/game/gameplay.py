# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact, concept_interact

from recipe_planner.utils import Get, Chop, Merge, Deliver,Grill

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.current_agent = self.sim_agents[0]
        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                ts = datetime.now().strftime('%m-%d-%y_%H-%M-%S')
                image_name = f'{self.filename}_{ts}.png'
                pygame.image.save(self.screen, f'{self.save_dir}/{image_name}')
                print(f'just saved image {image_name} to {self.save_dir}')
                return
            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
           
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                action = None
                if(len(test_actions)> 0):
                    action= test_actions.pop(0)
                    self.current_agent.action = action
                    concept_interact(self.current_agent, self.world, action )

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()


recipeActions = [Get('Plate'),Get('Plate'),Get('Lettuce'),Chop('Lettuce'),Get('Lettuce'),Get('Lettuce'),Get('Lettuce'),Get('Lettuce'),Get('Lettuce'),Get('Lettuce')]

recipeAct = [Get('Plate'),Get('Lettuce'),Chop('Lettuce'),Merge('Lettuce','Plate'),Get('Tomato'),Chop('Tomato'),Merge('Tomato','Lettuce-Plate'), 
             Get('Bun'),Merge('Bun','Lettuce-Plate-Tomato'), Deliver('Bun-Lettuce-Plate-Tomato')]

test_actions = [Get("Plate"),Get("Plate"),Get("Plate"),Get("Plate"),Get("Lettuce"),Get("Bun"),Get("Lettuce"),Get("Lettuce"),Chop("Lettuce"),Chop("Lettuce"),
                Merge("Bun", "Plate"),Merge("Lettuce", "Plate"),Chop("Lettuce"),Get("Plate"),Merge("Lettuce", "Plate"), Deliver("Bun-Plate"), Get("Tomato"),
                Chop("Tomato"),Merge("Lettuce", "Plate"),Get("Lettuce"),Get("Tomato"),Get("Plate"),Chop("Tomato"),Merge("Tomato", "Lettuce-Plate"),Get("Tomato"),
                Chop("Lettuce"),Get("Lettuce"),Get("Plate"),Get("Lettuce"),Merge("Tomato", "Lettuce-Plate"),Chop("Lettuce"),Get("Bun"),Chop("Tomato"),
                Merge("Lettuce", "Plate"),Get("Tomato"),Chop("Lettuce"),Chop("Tomato"),Get("Bun"),Merge("Lettuce", "Plate"),Merge("Bun", "Lettuce-Plate-Tomato"),
                Merge("Tomato", "Lettuce-Plate"),Get("Bun"),Merge("Lettuce", "Plate"),Merge("Tomato", "Lettuce-Plate"),Merge("Bun", "Lettuce-Plate-Tomato"),Get("Plate"),
                Get("Bun"),Get("Plate"),Deliver("Bun-Lettuce-Plate-Tomato"),Get("Bun"),Deliver("Bun-Lettuce-Plate-Tomato"),
                Merge("Bun", "Lettuce-Plate-Tomato"),Get("Plate"),Get("Tomato"),Deliver("Bun-Lettuce-Plate-Tomato"),Get("Bun"),Merge("Bun", "Plate"),Get("Lettuce")
]

