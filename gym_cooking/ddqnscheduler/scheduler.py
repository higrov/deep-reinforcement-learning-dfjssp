import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.regularizers import l2
import tensorflow as tf

#from rl.agents.dqn import DQNAgent
import rl

from collections import deque

from .behaviourpolicy import SoftEpsilonGreedyPolicy

from .processor import NormalizerProcessor



import numpy as np
import random
from .ddqn import DoubleDeepQNetwork
import warnings


def create_model(shape, nb_actions):
    model = Sequential()
    model.add(Input(input_shape=shape, name= 'l1'))
    model.add(Dense(30, activation='relu', name= 'l2'))
    model.add(Dense(30, activation='relu', name= 'l3'))
    model.add(Dense(30, activation='relu', name= 'l4'))
    model.add(Dense(30, activation='relu', name= 'l5'))
    model.add(Dense(30, activation='relu', name= 'l6'))
    model.add(Dense(nb_actions, activation='linear',  name= 'l7'))

    return model

from .parameter import *

warnings.filterwarnings('ignore')


class Agent:  # one node agent
    def __init__(self,nb_total_operations, nb_input_params, nb_actions):
        self.model = DoubleDeepQNetwork(nb_input_params, nb_actions)
        self.epsilon = EPSILON_MAX
        self.memory = deque(maxlen=50000)
        self.policy = SoftEpsilonGreedyPolicy(nb_total_operations,self.epsilon, nb_actions)
        self.prediction = np.zeros(nb_actions)

    def reset(self):
        self._epsilon_decay_()
        return self.epsilon

    def choose_action(self, state):
        prediction= self.model.predict_one(state)
        return self.policy.select_action(prediction)

    def _epsilon_decay_(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_MAX/MAX_EPISODE
        else:
            self.epsilon = EPSILON_MIN

    # agent memory
    def sample(self):
        sample_batch = random.sample(self.memory, BATCH)
        return sample_batch

    def observation(self, state, action_id, reward, next_state, done):
        sample = [state, action_id, reward, next_state, done]
        self.memory.append(sample)

    def state_target(self, batch):
        min_loss = 99999
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                action_t = np.argmax(self.model.predict(next_state)[0])
                target = reward + DISCOUNT_FACTOR*(self.model.predict(next_state,target=True)[0][action_t])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            state = np.array(state).reshape(1, 4)
            loss = self.model.train(state, target_f)
            if loss <= min_loss : 
                min_loss = loss

        return min_loss

    # build the replay buffer 
    def replay(self):

        if len(self.memory) < BATCH:
            return 99999

        batch = self.sample()
        min_loss = self.state_target(batch)

        return min_loss

    def update_target_model(self):
        self.model.update_target_model()

