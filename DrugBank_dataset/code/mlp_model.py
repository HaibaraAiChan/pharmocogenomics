
import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution3D, MaxPooling3D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
# For reproductivity
seed = 12306
np.random.seed(seed)


class MLP_Builder(object):

    @staticmethod
    def build(input_shape, hidden_units, num_labels):
        model = Sequential()
        # this is 3-layer MLP with ReLU after each layer
        model = Sequential()
        model.add(Dense(hidden_units, input_dim=input_shape))
        model.add(Activation('relu'))
        model.add(Dense(hidden_units))
        model.add(Activation('relu'))
        model.add(Dense(int(hidden_units/3)))
        model.add(Activation('relu'))
        model.add(Dense(num_labels))
        # this is the output for one-hot vector
        model.add(Activation('softmax'))
        model.summary()
        return model