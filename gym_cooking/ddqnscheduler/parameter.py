import os
from dataclasses import dataclass
import numpy as np
import random

# parameters
SINGLE_NODE = False  # False
OUTPUT_PORT = 2  # 2
SRCES = 8  # 8

# Reward
COMPLEX = False
BOUND = [0.5, 0.6]
W0 = [0.1, 0.03]
W1 = [0.01, 0.01]
W2 = [-0.6, -0.2]
W3 = -1
LM = 1.5
W = [0.1, 0.1]

A = 0
RRW = 3
# HOP_WEIGHT = 4
RANDOM_HOP = 0  # 0
RANDOM_CURRENT_DELAY_CC = 2  # originally 0 unit : T
RANDOM_CURRENT_DELAY_BE = [0, 3]  # originally [0,1] unit : T
PERIOD_CC = 2  # T
PERIOD_BE = 2
COMMAND_CONTROL = 60  # 40
BEST_EFFORT = 60  # 100
CC_DEADLINE = 7  # 5 (8 T), 10 least 5T, unit : T (if not, just multiply TIMESLOT_SIZE)
BE_DEADLINE = 7  # 50 ( 75 T ) 12
FIXED_SEQUENCE = False
FIRST_TRAIN = True
MAXSLOT_MODE = True
MAXSLOTS = 250  # 250
LEARNING_RATE = 0.001  # 0.0001
UPDATE = 400  # 500
EPSILON_DECAY = 0.9998  # 0.9998

# Save
DATE = '0429_2'
FILENAME = 'result/@0220/[15963]0.001464993692934513.h5'  # weight file name
WEIGHT_FILE = FILENAME

# RL agent
PRIORITY_QUEUE = 2
STATE = 2  # for simulation with different utilizations(periods), it has to be editted to 3
INPUT_SIZE = 4
# GCL_LENGTH = 3
OUTPUT_SIZE = 2
ALPHA = 0.1
INITIAL_ACTION = 0
ACTION_LIST = [0, 1]
ACTION_SIZE = len(ACTION_LIST)
BATCH = 64
EPSILON_MAX = 1
EPSILON_MIN = 0.01
DISCOUNT_FACTOR = 0.99

# Environment
MAX_EPISODE = 20000
# CC_PERIOD = 10
# AD_PERIOD = 6
# VD_PERIOD = 8
# BE_PERIOD = 4  # PERIOD는 Utilization을 위해 조절해야 할 듯

CC_BYTE = 1500
# AD_BYTE = 256
# VD_BYTE = 1500
BE_BYTE = 1500
TIMESLOT_SIZE = 0.6
BANDWIDTH = 20000  # bits per msec (20Mbps)
MAX_BURST = 12000
NODES = 9
MAX_REWARD = COMMAND_CONTROL * W[0] + BEST_EFFORT * W[1] + A * (COMMAND_CONTROL + BEST_EFFORT)