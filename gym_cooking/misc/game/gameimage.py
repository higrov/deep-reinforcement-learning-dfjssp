import pygame
import os
import numpy as np
from PIL import Image
from misc.game.game import Game
import time
import glob
# from misc.game.utils import *


class GameImage(Game):
    def __init__(self, filename, env_id, world, sim_agents, record=False):
        Game.__init__(self, world, sim_agents)
        self.game_record_dir = f'recordings/{filename}'
        self.record = record
        self.latest_suffix = ""
        self.is_env_prime = (env_id == 0) or ('eval' in env_id) or (env_id.split('_')[-1] in ['0', 'eval'])

    def on_init(self):
        super().on_init()

        if self.record and self.is_env_prime:
            # Make game_record folder if doesn't already exist
            if not os.path.exists(self.game_record_dir):
                os.makedirs(os.path.join(self.game_record_dir))

            # Clear game_record folder -- EXCEPT FOR *.gif
            for f in os.listdir(self.game_record_dir):
                if not f.endswith(".gif"):
                    os.remove(os.path.join(self.game_record_dir, f))

    def get_image_obs(self):
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb

    def save_image_obs(self, t):
        if self.record and self.is_env_prime:
            self.on_render()
            pygame.image.save(self.screen, f'{self.game_record_dir}/t={t:03d}.png')

    def get_animation_path(self):
        if not self.record or not self.is_env_prime:
            return ""
        
        return f"{self.game_record_dir}/run{self.latest_suffix}.gif"

    def generate_animation(self, suffix=""):
        if not self.record or not self.is_env_prime:
            return None
        
        fp_in = f'{self.game_record_dir}/t=*.png'
        fp_out = f"{self.game_record_dir}/run{suffix}.gif"
        
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)  # extract first image from iterator
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)
        self.latest_suffix = suffix
        return fp_out