import torch
import torch.nn as nn
import numpy as np
from nn_builder.pytorch.Base_Network import Base_Network

class RNN(nn.Module, Base_Network):
    """Creates a PyTorch recurrent neural network
    Args:
        - input_dim: Integer to indicate the dimension of the input into the network
        - layers_info: List of layer specifications to specify the hidden layers of the network. Each element of the list must be
                         one of these 3 forms:
                         - ["lstm", hidden_units]
                         - ["gru", hidden_units]
                         - ["linear", hidden_units]
        - hidden_activations: String or list of string to indicate the activations you want used on the output of linear hidden layers
                              (not including the output layer). Default is ReLU.
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
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
        - return_final_seq_only: Boolean to indicate whether you only want to return the output for the final timestep (True)
                                 or if you want to return the output for all timesteps (False)
        - random_seed: Integer to indicate the random seed you want to use

    NOTE that this class' forward method expects input data in the form: (batch, sequence length, features)
    """

    def __init__(self, input_dim, layers_info, output_activation=None,
                 hidden_activations="relu", dropout=0.0, initialiser="default", batch_norm=False,
                 columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range= (),
                 return_final_seq_only=True, random_seed=0):
        nn.Module.__init__(self)
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        self.return_final_seq_only = return_final_seq_only
        self.valid_RNN_hidden_layer_types = {"linear", "gru", "lstm"}
        Base_Network.__init__(self, input_dim, layers_info, output_activation,
                              hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_NN_input_dim_valid()
        self.check_RNN_layers_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()
        self.check_return_final_seq_only_valid()

    def check_RNN_layers_valid(self):
        """Checks that layers provided by user are valid"""
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_RNN_hidden_layer_types)
        error_msg_layer_form = "Layer must be of form [layer_name, hidden_units]"
        error_msg_layer_list = "Layers must be provided as a list"
        error_msg_output_heads = "Number of output activations must equal number of output heads"

        assert isinstance(self.layers_info, list), error_msg_layer_list

        all_layers = self.layers_info[:-1]
        output_layer = self.layers_info[-1]
        assert isinstance(output_layer, list), error_msg_layer_list
        if isinstance(output_layer[0], list):
            assert len(output_layer) == len(
                self.output_activation), error_msg_output_heads
            for layer in output_layer:
                all_layers.append(layer)
        else:
            assert not isinstance(self.output_activation, list) or len(
                self.output_activation) == 1, error_msg_output_heads
            all_layers.append(output_layer)

        rest_must_be_linear = False
        for layer in all_layers:
            assert isinstance(layer, list), "Each layer must be a list"
            assert isinstance(layer[0], str), error_msg_layer_type
            layer_type_name = layer[0].lower()
            assert layer_type_name in self.valid_RNN_hidden_layer_types, "Layer name {} not valid, use one of {}".format(
                layer_type_name, self.valid_RNN_hidden_layer_types)

            assert isinstance(layer[1], int), error_msg_layer_form
            assert layer[1] > 0, "Must have hidden_units >= 1"
            assert len(layer) == 2, error_msg_layer_form

            if rest_must_be_linear: assert layer[0].lower() == "linear", "If have linear layers then they must come at end"
            if layer_type_name == "linear": rest_must_be_linear = True

    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        RNN_hidden_layers = nn.ModuleList([])
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum(
            [output_dims[1] for output_dims in self.embedding_dimensions]))
        for layer in self.layers_info[:-1]:
            input_dim = self.create_and_append_layer(input_dim, layer, RNN_hidden_layers)
        self.input_dim_into_final_layer = input_dim
        return RNN_hidden_layers

    def create_and_append_layer(self, input_dim, layer, RNN_hidden_layers):
        layer_type_name = layer[0].lower()
        hidden_size = layer[1]
        if layer_type_name == "lstm":
            RNN_hidden_layers.extend([nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)])
        elif layer_type_name == "gru":
            RNN_hidden_layers.extend(
                [nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)])
        elif layer_type_name == "linear":
            RNN_hidden_layers.extend([nn.Linear(input_dim, hidden_size)])
        else:
            raise ValueError("Wrong layer names")
        input_dim = hidden_size
        return input_dim

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        input_dim = self.input_dim_into_final_layer
        if not isinstance(self.layers_info[-1][0], list): self.layers_info[-1] = [self.layers_info[-1]]
        for output_layer in self.layers_info[-1]:
            self.create_and_append_layer(input_dim, output_layer, output_layers)
        return output_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        self.initialise_parameters(self.hidden_layers)
        self.initialise_parameters(self.output_layers)
        self.initialise_parameters(self.embedding_layers)

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=layer[1]) for layer in self.layers_info[:-1]])
        return batch_norm_layers

    def get_activation(self, activations, ix=None):
        """Gets the activation function"""
        if isinstance(activations, list):
            activation = self.str_to_activations_converter[str(activations[ix]).lower()]
        else:
            activation = self.str_to_activations_converter[str(activations).lower()]
        return activation

    def forward(self, x):
        """Forward pass for the network. Note that it expects input data in the form (batch, seq length, features)"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x)
        batch_size, seq_length, data_dimension = x.shape
        if self.embedding_to_occur: x = self.incorporate_embeddings(x, batch_size, seq_length)
        x = self.process_hidden_layers(x, batch_size, seq_length)
        out = self.process_output_layers(x, batch_size, seq_length)
        if self.return_final_seq_only: out = out[:, -1, :]
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(out)
        return out

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        assert len(x.shape) == 3, "x should have the shape (batch_size, sequence_length, dimension)"
        assert x.shape[2] == self.input_dim, "x must have the same dimension as the input_dim you provided"
        for embedding_dim in self.columns_of_data_to_be_embedded:
            data = x[:, :, embedding_dim]
            data = data.contiguous().view(-1, 1)
            data_long = data.long()
            assert all(data_long >= 0), "All data to be embedded must be integers 0 and above -- {}".format(data_long)
            assert torch.sum(abs(data.float() - data_long.float())) < 0.0001, """Data columns to be embedded should be integer 
                                                                                values 0 and above to represent the different 
                                                                                classes"""
        if self.input_dim > len(self.columns_of_data_to_be_embedded): assert isinstance(x, torch.FloatTensor), f'Incorrect Tensor type x is {type(x)} is {x}'
        self.checked_forward_input_data_once = True #So that it doesn't check again

    def incorporate_embeddings(self, x, batch_size, seq_length):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the hidden layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, :, embedding_var].long()
            data = data.contiguous().view(batch_size * seq_length, -1)
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            embedded_data = embedded_data.view(batch_size, seq_length, -1)
            all_embedded_data.append(embedded_data)
        all_embedded_data = torch.cat(tuple(all_embedded_data), dim=2)
        non_embedded_data = x[:, :, [col for col in range(x.shape[2]) if col not in self.columns_of_data_to_be_embedded]].float()
        x = torch.cat((non_embedded_data, all_embedded_data), dim=2)
        return x

    def process_hidden_layers(self, x, batch_size, seq_length):
        """Puts the data x through all the hidden layers"""
        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) == nn.Linear:
                x = x.contiguous().view(batch_size * seq_length, -1)
                activation = self.get_activation(self.hidden_activations, layer_ix)
                x = activation(layer(x))
                x = x.view(batch_size, seq_length, layer.out_features)
            else:
                x = layer(x)
                x = x[0] #because we only want to keep the output and not the hidden states
            if self.batch_norm:
                x.transpose_(1, 2)
                x = self.batch_norm_layers[layer_ix](x)
                x.transpose_(1, 2)
            if self.dropout != 0.0: x = self.dropout_layer(x)
        return x

    def process_output_layers(self, x, batch_size, seq_length):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)

            if type(output_layer) == nn.Linear:
                x = x.contiguous().view(batch_size * seq_length, -1)
                temp_output = output_layer(x)
                if activation is not None:
                    temp_output = activation(temp_output)
                temp_output = temp_output.view(batch_size, seq_length, -1)
                x = x.view(batch_size, seq_length, -1)
            else:
                temp_output = output_layer(x)
                temp_output = temp_output[0]
                if activation is not None:
                    if type(activation) == nn.Softmax:
                        temp_output = temp_output.contiguous().view(batch_size * seq_length, -1)
                        temp_output = activation(temp_output)
                        temp_output = temp_output.view(batch_size, seq_length, -1)
                    else:
                        temp_output = activation(temp_output)
            if out is None: out = temp_output
            else: out = torch.cat((out, temp_output), dim=2)
        return out