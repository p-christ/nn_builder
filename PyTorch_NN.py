import torch
import numpy as np
import torch.nn as nn
from PyTorch_Base_Network import PyTorch_Base_Network


#TODO
# 1) Allow different dropout values per layer
# 2) Allow embedding layer dropout
# 3) Introduce a y_range set of values
# 4) Allow batch norm for input layer

class Neural_Network(nn.Module, PyTorch_Base_Network):
    """Creates a PyTorch neural network """

    def __init__(self, input_dim: int, linear_hidden_units: list, output_dim: int, output_activation: str ="None", hidden_activations="relu",
                 dropout: float =0.0, initialiser: str ="default", batch_norm: bool =False, cols_to_embed: list =[],
                 embedding_dimensions: list =[], print_model_summary=True):

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

        self.linear_layers = self.create_linear_layers()
        self.batch_norm_layers = self.create_batch_norm_layers()
        self.embedding_layers = self.create_embedding_layers()

        self.dropout_layer = nn.Dropout(p=dropout)
        self.initialise_all_parameters()

        if print_model_summary: self.print_model_summary()

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_input_and_output_dim_valid()
        self.check_linear_hidden_units_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()

    def create_linear_layers(self):
        """Creates the linear layers in the network"""
        linear_layers = nn.ModuleList([])
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum([output_dims[1] for output_dims in self.embedding_dimensions]))
        for hidden_unit in self.linear_hidden_units:
            linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            input_dim = hidden_unit
        linear_layers.extend([nn.Linear(hidden_unit, self.output_dim)])
        return linear_layers


    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_unit) for hidden_unit in self.linear_hidden_units])
        return batch_norm_layers

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = nn.ModuleList([])
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([nn.Embedding(input_dim, output_dim)])
        return embedding_layers

    def initialise_all_parameters(self):
        self.initialise_parameters(self.linear_layers)
        self.initialise_parameters(self.embedding_layers)

    def get_activation(self, activations, ix=None):
        if isinstance(activations, list):
            return self.str_to_activations_converter[activations[ix].lower()]
        return self.str_to_activations_converter[activations.lower()]

    def print_model_summary(self):
        print("Embedding layers")
        print("-------------")
        print(self.embedding_layers)
        print(" ")

        print("Linear layers")
        print("-------------")
        for layer_ix in range(len(self.linear_layers)):
            print(self.linear_layers[layer_ix])
            if self.batch_norm and layer_ix != len(self.linear_layers) - 1: print(self.batch_norm_layers[layer_ix])

    def forward(self, x):

        if len(self.embedding_dimensions) > 0:
            x = self.incorporate_embeddings(x)

        for layer_ix in range(len(self.linear_hidden_units)):
            linear_layer = self.linear_layers[layer_ix]
            activation = self.get_activation(self.hidden_activations, layer_ix)
            x = activation(linear_layer(x))
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            x = self.dropout_layer(x)

        final_activation = self.get_activation(self.output_activation)
        final_layer = self.linear_layers[-1]
        x = final_layer(x)

        if final_activation is not None: x = final_activation(x)

        return x

    def incorporate_embeddings(self, x):

        data_to_be_embedded = x[:, self.cols_to_embed]

        all_embedded_data = []

        for embedding_var in range(data_to_be_embedded.shape[1]):
            data = data_to_be_embedded[:, embedding_var]
            data = data.long()
            embedded_data = self.embedding_layers[embedding_var](data)
            all_embedded_data.append(embedded_data)

        all_embedded_data = torch.cat(tuple(all_embedded_data), dim=1)
        data_not_to_be_embedded = x[:, [col for col in range(x.shape[1]) if col not in self.cols_to_embed]]
        x = torch.cat((all_embedded_data, data_not_to_be_embedded), dim=1)
        return x




# batch norm on first layer option...


#
# z = []
#
#
# nn1 = Neural_Network(input_dim = 12, linear_hidden_units = [2, 2], hidden_activations="relu",
#                     output_dim=3, output_activation="relu", initialiser="he", cols_to_embed=[0, 5],
#                     embedding_dimensions=[ [20, 4], [30, 9]], batch_norm=True
#                     )
#
# x = torch.randn((25, 12)) + 5.0

#


# nn.