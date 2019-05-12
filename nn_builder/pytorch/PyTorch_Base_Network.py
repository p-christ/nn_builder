import random
import numpy as np
import torch

import torch.nn as nn

from nn_builder.Base_Network import Base_Network


class PyTorch_Base_Network(Base_Network):
    """Base class for PyTorch neural network classes"""
    def __init__(self, input_dim, layers, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary):
        super().__init__(input_dim, layers, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary)

        # self.set_all_random_seeds(random_seed)
        # self.str_to_activations_converter = self.create_str_to_activations_converter()
        # self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        # self.input_dim = input_dim
        # self.layers = layers
        # self.hidden_activations = hidden_activations
        # self.output_activation = output_activation
        # self.dropout = dropout
        # self.initialiser = initialiser
        # self.batch_norm = batch_norm
        # self.y_range = y_range
        # if print_model_summary: self.print_model_summary()
        #
        # self.check_all_user_inputs_valid()
        #
        # self.hidden_layers = self.create_hidden_layers()
        # self.output_layers = self.create_output_layers()
        #
        #
        # if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()
        # self.dropout_layer = nn.Dropout(p=dropout)
        # self.initialise_all_parameters()
        #
        # # Flag we use to run checks on the input data into forward the first time it is entered
        # self.checked_forward_input_data_once = False

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return nn.Dropout(p=self.dropout)

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": nn.ELU(), "hardshrink": nn.Hardshrink(), "hardtanh": nn.Hardtanh(),
                                        "leakyrelu": nn.LeakyReLU(), "logsigmoid": nn.LogSigmoid(), "prelu": nn.PReLU(),
                                        "relu": nn.ReLU(), "relu6": nn.ReLU6(), "rrelu": nn.RReLU(), "selu": nn.SELU(),
                                        "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(), "logsoftmax": nn.LogSoftmax(),
                                        "softshrink": nn.Softshrink(), "softsign": nn.Softsign(), "tanh": nn.Tanh(),
                                        "tanhshrink": nn.Tanhshrink(), "softmin": nn.Softmin(), "softmax": nn.Softmax(dim=1),
                                        "softmax2d": nn.Softmax2d(), "none": None}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": nn.init.uniform_, "normal": nn.init.normal_,
                                        "constant": nn.init.constant_,
                                        "eye": nn.init.eye_, "dirac": nn.init.dirac_,
                                        "xavier_uniform": nn.init.xavier_uniform_, "xavier": nn.init.xavier_uniform_,
                                        "xavier_normal": nn.init.xavier_normal_,
                                        "kaiming_uniform": nn.init.kaiming_uniform_, "kaiming": nn.init.kaiming_uniform_,
                                        "kaiming_normal": nn.init.kaiming_normal_, "he": nn.init.kaiming_normal_,
                                        "orthogonal": nn.init.orthogonal_, "sparse": nn.init.sparse_, "default": "use_default"}
        return str_to_initialiser_converter

    def check_NN_input_dim_valid(self):
        """Checks that user input for input_dim is valid"""
        assert isinstance(self.input_dim, int), "input_dim must be an integer"
        assert self.input_dim > 0, "input_dim must be 1 or higher"

    def check_activations_valid(self):
        """Checks that user input for hidden_activations and output_activation is valid"""
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        if isinstance(self.output_activation, list):
            for activation in self.output_activation:
                if activation is not None:
                    assert activation.lower() in set(valid_activations_strings), "Output activations must be string from list {}".format(valid_activations_strings)
        else:
            assert self.output_activation.lower() in set(valid_activations_strings), "Output activation must be string from list {}".format(valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations, list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(self.layers), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
            for activation in self.hidden_activations:
                assert isinstance(activation, str), "hidden_activations must be a string or list of strings"
                assert activation.lower() in set(valid_activations_strings), "each element in hidden_activations must be from list {}".format(valid_activations_strings)

    def check_embedding_dimensions_valid(self):
        """Checks that user input for embedding_dimensions is valid"""
        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"
        for embedding_dim in self.embedding_dimensions:
            assert len(embedding_dim) == 2 and isinstance(embedding_dim, list), \
                "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

    def check_initialiser_valid(self):
        """Checks that user input for initialiser is valid"""
        valid_initialisers = set(self.str_to_initialiser_converter.keys())
        assert isinstance(self.initialiser, str), "initialiser must be a string from list {}".format(valid_initialisers)
        assert self.initialiser.lower() in valid_initialisers, "initialiser must be from list {}".format(valid_initialisers)

    def check_y_range_values_valid(self):
        """Checks that user input for y_range is valid"""
        if self.y_range:
            assert isinstance(self.y_range, tuple) and len(self.y_range) == 2, "y_range must be a tuple of 2 floats or integers"
            for elem in range(2):
                assert isinstance(self.y_range[elem], float) or isinstance(self.y_range[elem], int), "y_range must be a tuple of 2 floats or integers"
            assert self.y_range[0] <= self.y_range[1], "y_range's first element must be smaller than the second element"

    def check_timesteps_to_output_valid(self):
        """Checks that user input for timesteps_to_output is valid"""
        assert self.timesteps_to_output in ["all", "last"]

    def check_rnn_hidden_layers_valid(self):
        """Checks that the layers given by user for an RNN are valid choices"""
        error_msg = "Layer must be of form ['gru', hidden_size], ['lstm', hidden_size] or ['linear', hidden_size]"
        seen_linear_layer = False
        for layer in self.layers:
            assert len(layer) == 2
            assert isinstance(layer[0], str) and layer[0].lower() in ["gru", "lstm", "linear"], error_msg
            assert isinstance(layer[1], int) and layer[1] > 0, error_msg
            if layer[0].lower() == "linear":
                assert not seen_linear_layer, "After the first linear layers all subsequent layers must be linear too"
                seen_linear_layer = True

        #
        #
        # -
        # -
        # -

    def get_activation(self, activations, ix=None):
        """Gets the activation function"""
        if isinstance(activations, list):
            return self.str_to_activations_converter[str(activations[ix]).lower()]
        return self.str_to_activations_converter[str(activations).lower()]

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)


    def initialise_parameters(self, parameters_list: nn.ModuleList):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                initialiser(parameters.weight)

    def flatten_tensor(self, tensor):
        """Flattens a tensor of shape (a, b, c, d, ...) into (a, b * c * d * .. )"""
        return tensor.reshape(tensor.shape[0], -1)
