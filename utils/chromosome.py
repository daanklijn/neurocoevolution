from itertools import chain
from random import random

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Concatenate, \
    BatchNormalization
import numpy as np


class VBNChromosome:
    """ Class that wraps the neural network. Includes functionality
    for Virtual Batch Normalization and the mutation of weights."""

    def __init__(self, number_actions=6, input_channels=4):
        self.input_channels = input_channels
        self.number_actions = number_actions
        inputs, outputs = self.construct_layers()
        self.model = Model(inputs=inputs, outputs=outputs)

    def construct_layers(self):
        """ Construct the Mnih et al. DQN architecture."""
        inputs = Input(shape=(84, 84, self.input_channels))
        layer1 = Conv2D(32, [8, 8], strides=(4, 4), activation="relu")(inputs)
        layer1 = BatchNormalization(momentum=0.95, center=False, scale=False)(layer1)
        layer2 = Conv2D(64, [4, 4], strides=(2, 2), activation="relu")(layer1)
        layer2 = BatchNormalization(momentum=0.95, center=False, scale=False)(layer2)
        layer3 = Conv2D(64, [3, 3], strides=(1, 1), activation="relu")(layer2)
        layer3 = BatchNormalization(momentum=0.95, center=False, scale=False)(layer3)
        layer4 = Flatten()(layer3)
        layer5 = Dense(512, activation="relu")(layer4)
        layer5 = BatchNormalization(momentum=0.95, center=False, scale=False)(layer5)
        action = Dense(self.number_actions, activation="softmax")(layer5)
        return [inputs], action

    def virtual_batch_norm(self, samples):
        """ We apply Batch Normalization on a number of samples. By setting the learning
        rate to 0 we make sure that the weights and biases are not affected. This method
        is only ment to be used at the start of training."""
        optimizer = tf.keras.optimizers.SGD(learning_rate=0)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(loss=loss, optimizer=optimizer)
        fake_y = np.zeros((len(samples), self.number_actions))
        self.model.fit(np.array(samples), fake_y)

    def get_weights(self, layers=None):
        """ Retrieve all the weights of the network. """
        layers = layers if layers else self.model.layers
        layer_weights = chain(*[layer.get_weights() for layer in layers])
        flat_weights = [weights.flatten() for weights in layer_weights]
        return np.concatenate(flat_weights)

    def set_weights(self, flat_weights, layers=None):
        """ Set all the weights of the network. """
        i = 0
        layers = layers if layers else self.model.layers
        for layer in layers:
            new_weights = []
            for sub_layer in layer.get_weights():
                reshaped = flat_weights[i: i + sub_layer.size].reshape(sub_layer.shape)
                new_weights.append(reshaped)
                i += sub_layer.size
            layer.set_weights(new_weights)

    def get_perturbable_layers(self):
        """ Get all the perturbable layers of the network. This excludes the
        BatchNorm layers. """
        return [layer for layer in self.model.layers if
                not isinstance(layer, BatchNormalization)]

    def get_perturbable_weights(self):
        """ Get all the perturbable weights of the network. This excludes the
        BatchNorm weights. """
        return self.get_weights(self.get_perturbable_layers())

    def set_perturbable_weights(self, flat_weights):
        """ Set all the perturbable weights of the network. This excludes setting
         the BatchNorm weights. """
        self.set_weights(flat_weights, self.get_perturbable_layers())

    def mutate(self, mutation_power):
        """ Mutate the current weights by adding a normally distributed vector of
        noise to the current weights. """
        weights = self.get_perturbable_weights()
        noise = np.random.normal(loc=0.0, scale=mutation_power, size=weights.shape)
        self.set_perturbable_weights(weights + noise)
        return noise

    def determine_actions(self, inputs):
        """ Choose an action based on the pixel inputs. We do this by simply
        selecting the action with the highest outputted value. """
        actions = self.model(inputs)
        return [np.argmax(action_set) for action_set in actions]
