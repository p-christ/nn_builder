import torch
import random
import numpy as np
import torch.nn as nn
from nn_builder.pytorch.Base_Network import Base_Network

#TODO
# 1) Allow different dropout values per layer
# 2) Allow embedding layer dropout
# 4) Allow batch norm for input layer

# change it so it infers the input dimension... and then dimensions of all layers


class NN(nn.Module, Base_Network):
    """Creates a PyTorch neural network
    Args:
        - input_dim: Integer to indicate the dimension of the input into the network
        - hidden_layers_info: List of integers to indicate the width and number of linear hidden layers you want in your network
        - output_dim: Integer to indicate the dimension of the output of the network if you want 1 output head. Provide a list of integers
                      if you want multiple output heads
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                              (not including the output layer). Default is ReLU.
        - dropout: Float to indicate what dropout probability you want applied after each hidden layer
        - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                       initialisers are supported. PyTorch's default initialisation is the default.
        - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
        - columns_of_data_to_be_embedded: List to indicate the columns numbers of the data that you want to be put through an embedding layer
                                          before being fed through the other layers of the network. Default option is no embeddings
        - embedding_dimensions: If you have categorical variables you want embedded before flowing through the network then
                                you specify the embedding dimensions here with a list like so: [ [embedding_input_dim_1, embedding_output_dim_1],
                                [embedding_input_dim_2, embedding_output_dim_2] ...]. Default is no embeddings
        - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                   output values to in regression tasks. Default is no range restriction
        - print_model_summary: Boolean to indicate whether you want a model summary printed after model is created. Default is False.
    """
    def __init__(self, input_dim: int, hidden_layers_info: list, output_dim, output_activation=None,
                 hidden_activations="relu", dropout: float =0.0, initialiser: str ="default", batch_norm: bool =False,
                 columns_of_data_to_be_embedded: list =[], embedding_dimensions: list =[], y_range: tuple = (),
                 random_seed=0, print_model_summary: bool =False):
        nn.Module.__init__(self)
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        Base_Network.__init__(self, input_dim, hidden_layers_info, output_dim, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_input_dim_valid()
        self.check_output_dim_valid()
        self.check_linear_hidden_units_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def create_hidden_layers(self):
        """Creates the linear layers in the network"""
        linear_layers = nn.ModuleList([])
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum([output_dims[1] for output_dims in self.embedding_dimensions]))
        for hidden_unit in self.hidden_layers_info:
            linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            input_dim = hidden_unit
        return linear_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        input_dim = self.hidden_layers_info[-1]
        if not isinstance(self.output_dim, list): self.output_dim = [self.output_dim]
        for output_dim in self.output_dim:
            output_layers.extend([nn.Linear(input_dim, output_dim)])
        return output_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_unit) for hidden_unit in self.hidden_layers_info])
        return batch_norm_layers

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = nn.ModuleList([])
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([nn.Embedding(input_dim, output_dim)])
        return embedding_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        self.initialise_parameters(self.hidden_layers)
        self.initialise_parameters(self.output_layers)
        self.initialise_parameters(self.embedding_layers)

    def print_model_summary(self):
        if len(self.embedding_layers) > 0:
            print("Embedding layers")
            print("-------------")
            print(self.embedding_layers)
            print(" ")
        print("-------------")
        print("Linear layers")
        print("-------------")
        for layer_ix in range(len(self.hidden_layers)):
            print(self.hidden_layers[layer_ix])
            if self.batch_norm: print(self.batch_norm_layers[layer_ix])
        print("-------------")
        print("Output Layers")
        print("-------------")
        for layer_ix in range(len(self.output_layers)):
            print(self.output_layers[layer_ix])

    def forward(self, x):
        """Forward pass for the network"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x)
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        for layer_ix, linear_layer in enumerate(self.hidden_layers):
            x = self.get_activation(self.hidden_activations, layer_ix)(linear_layer(x))
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            x = self.dropout_layer(x)

        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            temp_output = output_layer(x)
            if activation is not None: temp_output = activation(temp_output)
            if out is None:
                out = temp_output
            else:
                out = torch.cat((out, temp_output), dim=1)

        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(out)
        return out

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        for embedding_dim in self.columns_of_data_to_be_embedded:
            data = x[:, embedding_dim]
            data_long = data.long()
            assert all(data_long >= 0), "All data to be embedded must be integers 0 and above -- {}".format(data_long)
            assert torch.sum(abs(data.float() - data_long.float())) < 0.0001, """Data columns to be embedded should be integer 
                                                                                values 0 and above to represent the different 
                                                                                classes"""
        if self.input_dim > len(self.columns_of_data_to_be_embedded): assert isinstance(x, torch.FloatTensor)
        assert len(x.shape) == 2, "X should be a 2-dimensional tensor"
        self.checked_forward_input_data_once = True #So that it doesn't check again

    def incorporate_embeddings(self, x):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the linear layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, embedding_var].long()
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            all_embedded_data.append(embedded_data)
        all_embedded_data = torch.cat(tuple(all_embedded_data), dim=1)
        x = torch.cat((x[:, [col for col in range(x.shape[1]) if col not in self.columns_of_data_to_be_embedded]].float(), all_embedded_data), dim=1)
        return x
