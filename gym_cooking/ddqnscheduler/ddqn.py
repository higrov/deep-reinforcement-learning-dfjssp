#!/usr/bin/env python

import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.regularizers import l2
import tensorflow as tf
from .parameter import LEARNING_RATE
import warnings

tf.compat.v1.disable_eager_execution()

warnings.filterwarnings('ignore')

class DoubleDeepQNetwork():
    def __init__(self, nb_input_params, nb_actions, train= True, model_file= None):
        self.loss_history = []

        self.nb_input_params = nb_input_params
        self.nb_actions =nb_actions

        gpus = tf.config.list_physical_devices('GPU')
        if(gpus):
             tf.config.set_visible_devices(gpus[0], 'GPU')
             tf.device('/physical_device:GPU:1')
        else:
            tf.device('/physical_device:CPU:0')

        if train:
            self.model = self.create_model(shape= nb_input_params, nb_actions=nb_actions)
            self.target_model = self.create_model(shape= nb_input_params, nb_actions=nb_actions)
        else:
            self.model = tf.keras.models.load_model(model_file)
            self.target_model = tf.keras.models.load_model(model_file)

        self.update_target_model() # create the neural network to train the q function

    def create_model(self,shape, nb_actions):
        model = Sequential()

        model.add(Input(shape=(shape,)))
        model.add(Dense(30, activation='relu', name= 'l1'))
        model.add(Dense(30, activation='relu', name= 'l2'))
        model.add(Dense(30, activation='relu', name= 'l3'))
        model.add(Dense(30, activation='relu', name= 'l4'))
        model.add(Dense(30, activation='relu', name= 'l5'))
        model.add(Dense(nb_actions, activation='linear',  name= 'l6'))

        model.compile(optimizer= tf.keras.optimizers.legacy.Adam(lr= LEARNING_RATE), loss="mean_squared_error")
        #model.summary()

        return model
    
    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output
        loss = []
        history = self.model.fit(x, y, epochs=epochs, verbose=verbose)
        loss.append(history.history['loss'][0])
        return min(loss)

    def test(self, weight_file):
        self.model.load_weights(weight_file)

    def predict_one(self, state, target=False):
        return self.predict(np.array(state).reshape(self.nb_input_params,), target=target)

    def predict(self, state, target=False):
        state = np.array(state).reshape(1,self.nb_input_params)

        if target:  # get prediction from target network
            return self.target_model.predict(state)
        
        else:  # get prediction from local network
            return self.model.predict(state)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save our model 
    def save_model(self, filename):
        self.model.save(filename)