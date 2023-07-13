import pygame
import os
import numpy as np
from PIL import Image
from misc.game.game import Game
# from gym_cooking.misc.game.utils import *

class GameImage(Game):
    def __init__(self, filename, world, sim_agents, record=False):
        Game.__init__(self, world, sim_agents)
        self.game_record_dir = 'misc/game/record/{}/'.format(filename)
        self.record = record


    def on_init(self):
        super().on_init()

        if self.record:
            # Make game_record folder if doesn't already exist
            if not os.path.exists(self.game_record_dir):
                os.makedirs(self.game_record_dir)

            # Clear game_record folder
            for f in os.listdir(self.game_record_dir):
                os.remove(os.path.join(self.game_record_dir, f))

    def get_image_obs(self):
        self.on_render()
        img_rgb = pygame.surfarray.array3d(self.screen.copy()) # make a copy of the surface to avoid locking issues
        img_rgb = img_rgb.swapaxes(0, 1) # swap the height and width dimensions

        return img_rgb

    def save_image_obs(self, t):
        self.on_render()
        pygame.image.save(self.screen, '{}/t={:03d}.png'.format(self.game_record_dir, t))
