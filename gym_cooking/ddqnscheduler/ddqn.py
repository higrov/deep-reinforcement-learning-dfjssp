#!/usr/bin/env python

import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.regularizers import l2
import tensorflow as tf


tf.compat.v1.disable_eager_execution()


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


class DoubleDeepQNetwork:
    def __init__(self, nb_input_params, nb_actions, train= True, model_file= None):
        self.loss_history = []

        self.nb_input_params = nb_input_params
        self.nb_actions =nb_actions

        if train:
            self.model = create_model((1,nb_input_params), nb_actions)
            self.target_model = create_model((1,nb_input_params), nb_actions)
        else:
            self.model = tf.keras.models.load_model(model_file)
            self.target_model = tf.keras.models.load_model(model_file)

        self.update_target_model()
    # create the neural network to train the q function

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output
        loss = []
        history = self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        loss.append(history.history['loss'][0])  # loss 기록
        return min(loss)

    def test(self, weight_file):
        self.model.load_weights(weight_file)

    def predict_one(self, state, target=False):
        return self.predict(np.array(state).reshape(1, self.nb_input_params), target=target).flatten()

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.target_model.predict(state)
        else:  # get prediction from local network
            # print (state)
            return self.model.predict(state)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save our model 
    def save_model(self, filename):
        self.model.save(filename)