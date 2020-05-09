'''
neural network architecture
'''

__author__ = 'Oguzhan Gencoglu'

import tensorflow.keras as keras
from tensorflow.keras import layers

from configs import config as cf


def get_dense_model():
    '''
    define dense model architecture
    '''
    model = keras.Sequential()
    model.add(layers.Dense(512, activation='relu',
                           input_shape=(cf.bert_embed_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    return model
