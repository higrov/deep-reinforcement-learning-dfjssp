import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.regularizers import l2
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

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
        self.memory = SequentialMemory(limit=50000, window_length=1)
        self.policy = SoftEpsilonGreedyPolicy(nb_total_operations,self.epsilon, nb_actions)

    def reset(self):
        self._epsilon_decay_()
        return self.epsilon

    def choose_action(self, state):
        prediction= self.model.predict(state)
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
        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])  # next state

        p = self.model.predict(states)  # model predict with state
        p_ = self.model.predict(states_)
        pTarget_ = self.model.predict(states_, target=True)  # target_model predict with next state

        x = np.zeros((BATCH, INPUT_SIZE))
        y = np.zeros((BATCH, ACTION_SIZE))
        # errors = np.zeros(batch_len)

        for i in range(BATCH):
            o = batch[i]  # batch=[state, action, reward, next_state, done]
            s = o[0]
            a = o[1]
            # a = action_to_number(o[1])
            r = o[2]
            # ns = o[3]
            done = o[4]

            t = p[i]
            # old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + DISCOUNT_FACTOR * pTarget_[i][np.argmax(p_[i])]

            x[i] = s
            y[i] = t

        return [x, y]

    # build the replay buffer 
    def replay(self):

        if len(self.memory) < BATCH:
            return 99999

        batch = self.sample()
        x, y = self.state_target(batch)
        min_loss = self.model.train(x, y)

        return min_loss

    def update_target_model(self):
        self.model.update_target_model()

