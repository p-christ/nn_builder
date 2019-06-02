from tensorflow.keras.layers import BatchNormalization
from nn_builder.Overall_Base_Network import Overall_Base_Network
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as initializers
import numpy as np
import random
import tensorflow as tf
from abc import ABC, abstractmethod

class Base_Network(Overall_Base_Network, ABC):
    """Base class for TensorFlow neural network classes"""
    def __init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser, batch_norm, y_range,
                 random_seed, input_dim):
        if input_dim is not None: print("You don't need to provide input_dim for a tensorflow network")
        super().__init__(None, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)

    @abstractmethod
    def call(self, x, training=True):
        """Runs a forward pass of the tensorflow model"""
        raise NotImplementedError

    @abstractmethod
    def create_and_append_layer(self, layer, list_to_append_layer_to, activation, output_layer=False):
        """Creates a layer and appends it to the provided list"""
        raise NotImplementedError

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        random.seed(random_seed)

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": activations.elu, "exponential": activations.exponential,
                                        "hard_sigmoid": activations.hard_sigmoid, "linear": activations.linear,
                                        "relu": activations.relu, "selu": activations.selu, "sigmoid": activations.sigmoid,
                                        "softmax": activations.softmax, "softplus": activations.softplus,
                                        "softsign": activations.softsign, "tanh": activations.tanh, "none": activations.linear}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"glorot_normal": initializers.glorot_normal, "glorot_uniform": initializers.glorot_uniform,
                                        "xavier_normal": initializers.glorot_normal, "xavier_uniform": initializers.glorot_uniform,
                                        "xavier": initializers.glorot_uniform, "he_normal": initializers.he_normal(),
                                        "he_uniform": initializers.he_uniform(), "lecun_normal": initializers.lecun_normal(),
                                        "lecun_uniform": initializers.lecun_uniform(), "truncated_normal": initializers.TruncatedNormal,
                                        "variance_scaling": initializers.VarianceScaling, "default": initializers.glorot_uniform}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return tf.keras.layers.Dropout(rate=self.dropout)

    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        hidden_layers = []
        for layer_ix, layer in enumerate(self.layers_info[:-1]):
            activation = self.get_activation(self.hidden_activations, layer_ix)
            self.create_and_append_layer(layer, hidden_layers, activation, output_layer=False)
        return hidden_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = []
        network_type = type(self).__name__
        if network_type in ["CNN", "RNN"]:
            if not isinstance(self.layers_info[-1][0], list): self.layers_info[-1] = [self.layers_info[-1]]
        elif network_type == "NN":
            if isinstance(self.layers_info[-1], int): self.layers_info[-1] = [self.layers_info[-1]]
        else:
            raise ValueError("Network type not recognised")
        for output_layer_ix, output_layer in enumerate(self.layers_info[-1]):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            self.create_and_append_layer(output_layer, output_layers, activation, output_layer=True)
        return output_layers

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = []
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([tf.keras.layers.Embedding(input_dim, output_dim)])
        return embedding_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = []
        for layer in self.layers_info[:-1]:
            batch_norm_layers.extend([BatchNormalization()])
        return batch_norm_layers

    def print_model_summary(self, input_shape=None):
        assert input_shape is not None, "Must provide the input_shape parameter as a tuple"
        self.build(input_shape=input_shape)
        self.dropout_layer.build(input_shape)
        self.summary()