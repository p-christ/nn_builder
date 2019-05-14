from nn_builder.Base_Network import Base_Network
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as initializers
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class TensorFlow_Base_Network(Base_Network, ABC):
    """Base class for TensorFlow neural network classes"""
    def __init__(self, input_dim, layers, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary):
        super().__init__(input_dim, layers, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary)



    @abstractmethod
    def create_model(self, input_data):
        """Creates and returns a tensorflow model"""
        raise NotImplementedError


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
                                        "xavier": initializers.glorot_uniform,
                                        "he_normal": initializers.he_normal, "he_uniform": initializers.he_uniform,
                                        "identity": initializers.identity, "lecun_normal": initializers.lecun_normal,
                                        "lecun_uniform": initializers.lecun_uniform, "truncated_normal": initializers.TruncatedNormal,
                                        "variance_scaling": initializers.VarianceScaling, "default": initializers.glorot_uniform}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return tf.keras.layers.Dropout(rate=self.dropout)

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)


