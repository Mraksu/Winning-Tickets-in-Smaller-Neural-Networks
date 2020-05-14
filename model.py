#!/usr/bin/env python
# coding: utf-8

# In[1]:

'''This file is for building the model for the experiments.
Same model is used for every experiment for consistency.'''

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras import models, layers, datasets
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model


# In[2]:


def pruned_nn(pruning_params,first_layer=300,second_layer=100):
    """
    Function to define a 300 100 dense
    fully connected architecture for MNIST
    classification
    Arguments:
    pruning_params=dictionary for the desired pruning rate.
    see the used_func.py for detailed function
    first_layer= Desired unit size in first hidden layer
    second_layer= Desired unit size in second hidden layer

    returns a model

    """
    #Using 2 different pruning rate, because the layer connected
    #to the output pruned as half of the rest of the network
    prun=pruning_params[0]
    prun_out=pruning_params[1]

    pruned_model = Sequential()

    pruned_model.add(
        sparsity.prune_low_magnitude(
            Dense(units = first_layer, activation = 'relu',
                  kernel_initializer = tf.keras.initializers.glorot_normal(),
                  input_shape = (784,)
                 ),
            **prun)
    )

    pruned_model.add(
        sparsity.prune_low_magnitude(
            Dense(
                units = second_layer, activation = 'relu',
                kernel_initializer = tf.keras.initializers.glorot_normal()
            ),
            **prun))

    pruned_model.add(
        sparsity.prune_low_magnitude(
            Dense(
                units = 10, activation = 'softmax'
            ),
            **prun_out))


    pruned_model.compile(
        loss = tf.keras.losses.categorical_crossentropy,
        # 0.0012 learning rate with Adam optimizer is used in the original paper
        optimizer=tf.keras.optimizers.Adam(0.0012),
        metrics = ['accuracy'])

    return pruned_model
