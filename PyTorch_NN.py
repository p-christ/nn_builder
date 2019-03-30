import torch.nn as nn
from PyTorch_Base_Network import PyTorch_Base_Network

class Neural_Network(nn.Module, PyTorch_Base_Network):
    """Creates a PyTorch neural network """

    def __init__(self, input_dim: int, linear_hidden_units: list, hidden_activations, output_dim: int, output_activation: str,
                 initialiser: str ="default", batch_norm: bool =False, cols_to_embed=None, embedding_dimensions: list =[]):
        super(Neural_Network, self).__init__()
        PyTorch_Base_Network.__init__(self)

        self.input_dim = input_dim
        self.linear_hidden_units = linear_hidden_units
        self.hidden_activations = hidden_activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.initialiser = initialiser
        self.batch_norm = batch_norm

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
        self.check_input_and_output_dim_valid()
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
        self.initialise_parameters(self.linear_layers)
        self.initialise_parameters(self.embedding_layers)

    def forward(self, x):

        pass

# Neural_Network(input_dim = 2, linear_hidden_units = [2, 2], hidden_activations="relu", output_dim=3, output_activation="relu", initialiser="her")