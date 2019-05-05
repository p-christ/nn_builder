import random
import numpy as np
import torch

import torch.nn as nn

class Base_Network(object):
    """Base class for PyTorch neural network classes"""
    def __init__(self, input_dim, hidden_layers_info, output_dim, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary):
        self.set_all_random_seeds(random_seed)
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        self.input_dim = input_dim
        self.hidden_layers_info = hidden_layers_info
        self.hidden_activations = hidden_activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range
        if print_model_summary: self.print_model_summary()

        self.check_all_user_inputs_valid()

        self.hidden_layers = self.create_hidden_layers()
        self.output_layers = self.create_output_layers()
        if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.initialise_all_parameters()

        # Flag we use to run checks on the input data into forward the first time it is entered
        self.checked_forward_input_data_once = False

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        raise ValueError("Must be implemented")

    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        raise ValueError("Must be implemented")

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        raise ValueError("Must be implemented")

    def create_output_layers(self):
        """Creates the hidden layers in the network"""
        raise ValueError("Must be implemented")

    def forward(self, input_data):
        """Runs a forward pass of the network"""
        raise ValueError("Must be implemented")

    def check_input_data_into_forward_once(self, input_data):
        """Checks the input data into the network is of the right form. Only runs the first time data is provided
        otherwise would slow down training too much"""
        raise ValueError("Must be implemented")

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

    def check_input_dim_valid(self):
        """Checks that user input for input_dim is valid"""
        assert isinstance(self.input_dim, int), "input_dim must be an integer"
        assert self.input_dim > 0, "input_dim must be 1 or higher"

    def check_output_dim_valid(self):
        """Checks that user input for output_dim is valid"""
        if not isinstance(self.output_dim, list):
            out_dimensions = [self.output_dim]
        else:
            assert isinstance(self.output_activation, list), "Output activation must be a list if output_dim is a list"
            out_dimensions = self.output_dim
            assert len(self.output_dim) == len(self.output_activation), "If one is a list then output_dim and output_activation must be list of same length"
        for dim in out_dimensions:
            assert isinstance(dim, int), "output_dim must be integers"
            assert dim > 0, "output_dim must be 1 or higher"

    def check_linear_hidden_units_valid(self):
        """Checks that user input for hidden_units is valid"""
        assert isinstance(self.hidden_layers_info, list), "hidden_units must be a list"
        for hidden_unit in self.hidden_layers_info:
            assert isinstance(hidden_unit, int), "hidden_units must be a list of integers"
            assert hidden_unit > 0, "Every element of hidden_units must be 1 or higher"

    def check_cnn_hidden_layers_valid(self):
        """Checks that the user inputs for cnn_hidden_layers were valid. cnn_hidden_layers must be a list of layers where
        each layer must be of one of these forms:
        - ["conv", channels, kernel_size, stride, padding]
        - ["maxpool", kernel_size, stride, padding]
        - ["avgpool", kernel_size, stride, padding]
        - ["adaptivemaxpool", output height, output width]
        - ["adaptiveavgpool", output height, output width]
        - ["linear", in, out]
        """
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_cnn_hidden_layer_types)
        error_msg_conv_layer = """Conv layer must be of form ['conv', channels, kernel_size, stride, padding] where the 
                               final 4 elements are non-negative integers"""
        error_msg_maxpool_layer = """Maxpool layer must be of form ['maxpool', kernel_size, stride, padding] where the 
                                       final 2 elements are non-negative integers"""
        error_msg_avgpool_layer = """Avgpool layer must be of form ['avgpool', kernel_size, stride, padding] where the 
                                               final 2 elements are non-negative integers"""
        error_msg_adaptivemaxpool_layer = """Adaptivemaxpool layer must be of form ['adaptivemaxpool', output height, output width]"""
        error_msg_adaptiveavgpool_layer = """Adaptiveavgpool layer must be of form ['adaptiveavgpool', output height, output width]"""
        error_msg_linear_layer = """Linear layer must be of form ['linear', in, out] where in and out are non-negative integers"""

        assert isinstance(self.hidden_layers_info, list), "hidden_layers must be a list"

        for layer in self.hidden_layers_info:
            assert isinstance(layer, list), "Each layer must be a list"
            assert isinstance(layer[0], str), error_msg_layer_type
            layer_type_name = layer[0].lower()
            assert layer_type_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(layer_type_name, self.valid_cnn_hidden_layer_types)
            if layer_type_name == "conv":
                assert len(layer) == 5, error_msg_conv_layer
                for ix in range(3): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0, error_msg_conv_layer
                assert isinstance(layer[4], int) and layer[4] >= 0, error_msg_conv_layer
            elif layer_type_name == "maxpool":
                assert len(layer) == 4, error_msg_maxpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_maxpool_layer
                if layer[1] != layer[2]: print("NOTE that your maxpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
                assert isinstance(layer[3], int) and layer[3] >= 0, error_msg_conv_layer
            elif layer_type_name == "avgpool":
                assert len(layer) == 4, error_msg_avgpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_avgpool_layer
                assert isinstance(layer[3], int) and layer[3] >= 0, error_msg_conv_layer
                if layer[1] != layer[2]:print("NOTE that your avgpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
            elif layer_type_name == "adaptivemaxpool":
                assert len(layer) == 3, error_msg_adaptivemaxpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_adaptivemaxpool_layer
            elif layer_type_name == "adaptiveavgpool":
                assert len(layer) == 3, error_msg_adaptiveavgpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[
                    ix + 1] > 0, error_msg_adaptiveavgpool_layer
            elif layer_type_name == "linear":
                assert len(layer) == 3, error_msg_linear_layer
                for ix in range(2): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0
            else:
                raise ValueError("Invalid layer name")

        rest_must_be_linear = False
        for ix, layer in enumerate(self.hidden_layers_info):
            if rest_must_be_linear: assert layer[0].lower() == "linear", "If have linear layers then they must come at end"
            if layer[0].lower() == "linear":
                rest_must_be_linear = True

    def check_activations_valid(self):
        """Checks that user input for hidden_activations and output_activation is valid"""
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        if isinstance(self.output_activation, list):
            for activation in self.output_activation:
                if activation is not None:
                    assert activation.lower() in set(valid_activations_strings), "Output activations must be string from list {}".format(valid_activations_strings)
            assert len(self.output_activation) == len(self.output_dim), "Must be same amount of output activations as output dimensions"
        else:
            assert self.output_activation.lower() in set(valid_activations_strings), "Output activation must be string from list {}".format(valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations, list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(self.hidden_layers_info), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
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

    def initialise_parameters(self, parameters_list: nn.ModuleList):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                initialiser(parameters.weight)

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

    def get_activation(self, activations, ix=None):
        """Gets the activation function"""
        if isinstance(activations, list):
            return self.str_to_activations_converter[str(activations[ix]).lower()]
        return self.str_to_activations_converter[str(activations).lower()]

    def flatten_tensor(self, tensor):
        """Flattens a tensor of shape (a, b, c, d, ...) into (a, b * c * d * .. )"""
        return tensor.reshape(tensor.shape[0], -1)