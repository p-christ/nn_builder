import random
import numpy as np
import torch
import torch.nn as nn
from nn_builder.Overall_Base_Network import Overall_Base_Network
from abc import ABC, abstractmethod

class Base_Network(Overall_Base_Network, ABC):
    """Base class for PyTorch neural network classes"""
    def __init__(self, input_dim, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed):
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        super().__init__(input_dim, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)
        self.initialise_all_parameters()
        # Flag we use to run checks on the input data into forward the first time it is entered
        self.checked_forward_input_data_once = False

    @abstractmethod
    def initialise_all_parameters(self):
        """Initialises all the parameters of the network"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data):
        """Runs a forward pass of the network"""
        raise NotImplementedError

    @abstractmethod
    def check_input_data_into_forward_once(self, input_data):
        """Checks the input data into the network is of the right form. Only runs the first time data is provided
        otherwise would slow down training too much"""
        raise NotImplementedError

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": nn.ELU(), "hardshrink": nn.Hardshrink(), "hardtanh": nn.Hardtanh(),
                                        "leakyrelu": nn.LeakyReLU(), "logsigmoid": nn.LogSigmoid(), "prelu": nn.PReLU(),
                                        "relu": nn.ReLU(), "relu6": nn.ReLU6(), "rrelu": nn.RReLU(), "selu": nn.SELU(),
                                        "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(), "logsoftmax": nn.LogSoftmax(),
                                        "softshrink": nn.Softshrink(), "softsign": nn.Softsign(), "tanh": nn.Tanh(),
                                        "tanhshrink": nn.Tanhshrink(), "softmin": nn.Softmin(), "softmax": nn.Softmax(dim=1),
                                         "none": None}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": nn.init.uniform_, "normal": nn.init.normal_,
                                        "eye": nn.init.eye_,
                                        "xavier_uniform": nn.init.xavier_uniform_, "xavier": nn.init.xavier_uniform_,
                                        "xavier_normal": nn.init.xavier_normal_,
                                        "kaiming_uniform": nn.init.kaiming_uniform_, "kaiming": nn.init.kaiming_uniform_,
                                        "kaiming_normal": nn.init.kaiming_normal_, "he": nn.init.kaiming_normal_,
                                        "orthogonal": nn.init.orthogonal_,  "default": "use_default"}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return nn.Dropout(p=self.dropout)

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = nn.ModuleList([])
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([nn.Embedding(input_dim, output_dim)])
        return embedding_layers

    def initialise_parameters(self, parameters_list):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                if type(parameters) == nn.Linear:
                    initialiser(parameters.weight)
                elif type(parameters) in [nn.LSTM, nn.RNN, nn.GRU]:
                    initialiser(parameters.weight_hh_l0)
                    initialiser(parameters.weight_ih_l0)

    def flatten_tensor(self, tensor):
        """Flattens a tensor of shape (a, b, c, d, ...) into shape (a, b * c * d * .. )"""
        return tensor.reshape(tensor.shape[0], -1)

    def print_model_summary(self):
        print(self)

