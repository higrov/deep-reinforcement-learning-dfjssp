import sys
import numpy as np
from collections import deque

from .behaviourpolicy import SoftEpsilonGreedyPolicy

import numpy as np
import random
from .ddqn import DoubleDeepQNetwork

import warnings

from .parameter import EPSILON_MAX, EPSILON_MIN, BATCH, DISCOUNT_FACTOR, MAX_EPISODE

warnings.filterwarnings('ignore')

class SchedulingAgent:  # one node agent
    def __init__(self,nb_total_operations, nb_input_params, nb_actions, train = True, network_model_file = None):
        self.model = DoubleDeepQNetwork(nb_input_params, nb_actions, train=train,  model_file=network_model_file)
        self.epsilon = EPSILON_MAX
        self.min_loss = 99999
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
        self.min_loss = self.state_target(batch)

        return self.min_loss

    def update_target_model(self):
        self.model.update_target_model()

