import torch
import numpy as np
import torch.nn as nn
from PyTorch_Base_Network import PyTorch_Base_Network

class Neural_Network(nn.Module, PyTorch_Base_Network):
    """Creates a PyTorch neural network """

    def __init__(self, input_dim: int, linear_hidden_units: list, hidden_activations, output_dim: int, output_activation: str,
                 dropout: float = 0.0,
                 initialiser: str ="default", batch_norm: bool =False, cols_to_embed: list =[], embedding_dimensions: list =[]):
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
        self.dropout_layer = nn.Dropout(p=dropout)

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
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum([output_dims[1] for output_dims in self.embedding_dimensions]))
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

    def get_activation(self, activations, ix=None):
        if isinstance(activations, list):
            return self.str_to_activations_converter[activations[ix]]
        return self.str_to_activations_converter[activations]

    def forward(self, x):
        print(x.shape)

        if len(self.embedding_dimensions) > 0:

            data_to_be_embedded = x[:, self.cols_to_embed]

            all_embedded_data = []

            for embedding_var in range(data_to_be_embedded.shape[1]):
                data = data_to_be_embedded[:, embedding_var]
                data = data.long()
                print(data)
                embedded_data = self.embedding_layers[embedding_var](data)
                all_embedded_data.append(embedded_data)

            all_embedded_data = torch.cat(tuple(all_embedded_data), dim=1)

            print("All embedded data shape ", all_embedded_data.shape)
            data_not_to_be_embedded = x[:, [col for col in range(x.shape[1]) if col not in self.cols_to_embed]]

            print(" data_not_to_be_embedded data shape ", data_not_to_be_embedded.shape)

            x = torch.cat((all_embedded_data, data_not_to_be_embedded), dim=1)

        for layer_ix in range(len(self.linear_hidden_units)):
            linear_layer = self.linear_layers[layer_ix]
            activation = self.get_activation(self.hidden_activations, layer_ix)
            print(x.shape)
            x = activation(linear_layer(x))
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            x = self.dropout_layer(x)

        final_activation = self.get_activation(self.output_activation)
        final_layer = self.linear_layers[-1]

        x = final_activation(final_layer(x))

        return x

#
# z = []
# print( np.sum([x[1] for x in z]))
#
# print(len([0, 5]))
# print(len([ [2, 4], [3, 9]]))
#
# nn = Neural_Network(input_dim = 12, linear_hidden_units = [2, 2], hidden_activations="relu",
#                     output_dim=3, output_activation="relu", initialiser="he", cols_to_embed=[0, 5],
#                     embedding_dimensions=[ [20, 4], [30, 9]]
#                     )
#
# x = torch.randn((25, 12)) + 5.0
#
# print(nn.forward(x).shape)
#


# nn.