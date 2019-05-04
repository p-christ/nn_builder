import torch
import random
import numpy as np
import torch.nn as nn
from nn_builder.pytorch.Base_Network import Base_Network


class CNN(nn.Module, Base_Network):

    def __init__(self, input_dim, hidden_layers, output_dim, output_activation=None, hidden_activations="relu",
                 dropout: float = 0.0, initialiser: str = "default", batch_norm: bool = False, y_range: tuple = (),
                 random_seed=0, print_model_summary: bool =False):
        self.set_all_random_seeds(random_seed)
        nn.Module.__init__(self)
        Base_Network.__init__(self)
        self.input_dim = input_dim
        self.cnn_hidden_layers = hidden_layers
        self.hidden_activations = hidden_activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range
        self.print_model_summary = print_model_summary
        self.valid_cnn_hidden_layer_types = set(['conv', 'maxpool', 'avgpool', 'adaptivemaxpool', 'adaptiveavgpool'])

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_input_and_output_dim_valid()
        # self.check_linear_hidden_units_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def check_cnn_hidden_layers_valid(self):
        """Checks that the user inputs for cnn_hidden_layers were valid"""
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_cnn_hidden_layer_types)
        error_msg_conv_layer = """Conv layer must be of form ['conv', channels, kernel_size, stride, padding] where the 
                               final 4 elements are non-negative integers"""
        error_msg_maxpool_layer = """Maxpool layer must be of form ['maxpool', kernel_size, stride] where the 
                                       final 2 elements are non-negative integers"""
        error_msg_avgpool_layer = """Avgpool layer must be of form ['avgpool', kernel_size, stride] where the 
                                               final 2 elements are non-negative integers"""
        error_msg_adaptivemaxpool_layer = """Adaptivemaxpool layer must be of form ['adaptivemaxpool']"""
        error_msg_adaptiveavgpool_layer = """Adaptiveavgpool layer must be of form ['adaptiveavgpool']"""

        for layer in self.cnn_hidden_layers:
            assert isinstance(layer[0], str), error_msg_layer_type
            layer_type_name = layer[0].lower()
            assert layer_type_name in self.valid_cnn_hidden_layer_types, error_msg_layer_type
            if layer_type_name == "conv":
                assert len(layer) == 4, error_msg_conv_layer
                for ix in range(4): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0, error_msg_conv_layer
            elif layer_type_name == "maxpool":
                assert len(layer) == 3, error_msg_maxpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_maxpool_layer
            elif layer_type_name == "avgpool":
                assert len(layer) == 3, error_msg_avgpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_avgpool_layer
            elif layer_type_name == "adaptivemaxpool":
                assert len(layer) == 1, error_msg_adaptivemaxpool_layer
            elif layer_type_name == "adaptiveavgpool":
                assert len(layer) == 1, error_msg_adaptiveavgpool_layer

