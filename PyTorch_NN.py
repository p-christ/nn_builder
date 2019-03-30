import torch.nn as nn
from PyTorch_Base_Network import PyTorch_Base_Network

class Neural_Network(nn.Module, PyTorch_Base_Network):

    def __init__(self, input_dim, linear_hidden_units, hidden_activations, output_dim, output_activation,
                 initialiser, batch_norm, cols_to_embed, embedding_dimensions=[]):
        super(Neural_Network, self).__init__()

        self.input_dim = input_dim
        self.linear_hidden_units = linear_hidden_units
        self.hidden_activations = hidden_activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.initialiser = initialiser
        self.batch_norm = batch_norm

        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()

        self.cols_to_embed = cols_to_embed
        self.embedding_dimensions = embedding_dimensions

        self.check_all_user_inputs_valid()

        self.linear_layers = nn.ModuleList([])
        if self.batch_norm: self.batch_norm_layers = nn.ModuleList([])
        self.embedding_layers = nn.ModuleList([])

        self.create_linear_and_batch_norm_layers()
        self.create_embedding_layers()

        self.initialise_all_parameters()

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_linear_hidden_units_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()

    def create_linear_and_batch_norm_layers(self):
        """Creates the linear and batch norm layers in the network"""
        input_dim = self.input_dim
        for hidden_unit in self.linear_hidden_units:
            self.linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            if self.batch_norm: self.batch_norm_layers.extend([nn.BatchNorm1d(num_features=hidden_unit)])
            input_dim = hidden_unit
        self.linear_layers.extend([nn.Linear(hidden_unit, self.output_dim)])

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            self.embedding_layers.extend([nn.Embedding(input_dim, output_dim)])

    def initialise_all_parameters(self):
        """Initialises all the parameters of the network"""
        for parameters in self.linear_layers + self.embedding_layers:
            if self.initialiser == "Xavier":
                nn.init.xavier_normal_(parameters.weight)
            elif self.initialiser == "He":
                nn.init.kaiming_normal_(parameters.weight)
            elif self.initialiser == "Default":
                continue
            else:
                raise NotImplementedError("There is only support for activations 'Xavier', 'He', 'Default'")

    def forward(self, x):

        pass

