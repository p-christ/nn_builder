import torch
import torch.nn as nn
import numpy as np
from nn_builder.pytorch.Base_Network import Base_Network

class CNN(nn.Module, Base_Network):
    """Creates a PyTorch convolutional neural network
    Args:
        - input_dim: Tuple of integers to indicate the (channels, height, width) dimension of the input
        - layers_info: List of layer specifications to specify the hidden layers of the network. Each element of the list must be
                         one of these 6 forms:
                         - ["conv", channels, kernel_size, stride, padding]
                         - ["maxpool", kernel_size, stride, padding]
                         - ["avgpool", kernel_size, stride, padding]
                         - ["adaptivemaxpool", output height, output width]
                         - ["adaptiveavgpool", output height, output width]
                         - ["linear", out]
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                              (not including the output layer). Default is ReLU.
        - dropout: Float to indicate what dropout probability you want applied after each hidden layer
        - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                       initialisers are supported. PyTorch's default initialisation is the default.
        - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
        - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                   output values to in regression tasks. Default is no range restriction
        - random_seed: Integer to indicate the random seed you want to use

    NOTE that this class' forward method expects input data in the form: (batch, channels, height, width)
    """
    def __init__(self, input_dim, layers_info, output_activation=None,
                 hidden_activations="relu", dropout=0.0, initialiser="default", batch_norm=False,
                 y_range= (), random_seed=0, converted_from_tf_model=False):
        nn.Module.__init__(self)
        self.valid_cnn_hidden_layer_types = {'conv', 'maxpool', 'avgpool', 'adaptivemaxpool', 'adaptiveavgpool', 'linear'}
        self.valid_layer_types_with_no_parameters = [nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d]
        Base_Network.__init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed)
        self.converted_from_tf_model = converted_from_tf_model

    def flatten_tensor(self, tensor):
        """Flattens a tensor of shape (a, b, c, d, ...) into shape (a, b * c * d * .. )"""
        if self.converted_from_tf_model:
            tensor = tensor.permute(0, 2, 3, 1).contiguous()
            tensor = tensor.view(tensor.size(0), -1)
        else:
            tensor = tensor.reshape(tensor.shape[0], -1)
        return tensor

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_CNN_input_dim_valid()
        self.check_CNN_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def check_CNN_input_dim_valid(self):
        """Checks that the CNN input dim valid"""
        error_msg = "input_dim must be a tuple of 3 integers indicating (channels, height, width)"
        assert isinstance(self.input_dim, tuple), error_msg
        for element in self.input_dim: assert isinstance(element, int) and element > 0

    def check_CNN_layers_valid(self):
        """Checks that the user inputs for cnn_hidden_layers were valid. cnn_hidden_layers must be a list of layers where
        each layer must be of one of these forms:
        - ["conv", channels, kernel_size, stride, padding]
        - ["maxpool", kernel_size, stride, padding]
        - ["avgpool", kernel_size, stride, padding]
        - ["adaptivemaxpool", output height, output width]
        - ["adaptiveavgpool", output height, output width]
        - ["linear", out]
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
        error_msg_linear_layer = """Linear layer must be of form ['linear', out] where out is a non-negative integers"""
        assert isinstance(self.layers_info, list), "layers must be a list"

        all_layers = self.layers_info[:-1]
        output_layer = self.layers_info[-1]
        assert isinstance(output_layer, list), "layers must be a list"
        if isinstance(output_layer[0], list):
            assert len(output_layer) == len(
                self.output_activation), "Number of output activations must equal number of output heads"
            for layer in output_layer:
                all_layers.append(layer)
                assert layer[0].lower() == "linear", "Final layer must be linear"
        else:
            all_layers.append(output_layer)
            assert isinstance(output_layer[0], str), error_msg_layer_type
            assert output_layer[0].lower() == "linear", "Final layer must be linear"

        for layer in all_layers:
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
                assert len(layer) == 2, error_msg_linear_layer
                for ix in range(1): assert isinstance(layer[ix+1], int) and layer[ix+1] > 0
            else:
                raise ValueError("Invalid layer name")

        rest_must_be_linear = False
        for ix, layer in enumerate(all_layers):
            if rest_must_be_linear: assert layer[0].lower() == "linear", "If have linear layers then they must come at end"
            if layer[0].lower() == "linear":
                rest_must_be_linear = True

    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        cnn_hidden_layers = nn.ModuleList([])
        input_dim = self.input_dim
        for layer in self.layers_info[:-1]:
            input_dim = self.create_and_append_layer(input_dim, layer, cnn_hidden_layers)
        self.input_dim_into_final_layer = input_dim
        return cnn_hidden_layers

    def create_and_append_layer(self, input_dim, layer, list_to_append_layer_to):
        """Creates and appends a layer to the list provided"""
        layer_name = layer[0].lower()
        assert layer_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(
            layer_name, self.valid_cnn_hidden_layer_types)
        if layer_name == "conv":
            list_to_append_layer_to.extend([nn.Conv2d(in_channels=input_dim[0], out_channels=layer[1], kernel_size=layer[2],
                                                stride=layer[3], padding=layer[4])])
        elif layer_name == "maxpool":
            list_to_append_layer_to.extend([nn.MaxPool2d(kernel_size=layer[1],
                                                   stride=layer[2], padding=layer[3])])
        elif layer_name == "avgpool":
            list_to_append_layer_to.extend([nn.AvgPool2d(kernel_size=layer[1],
                                                   stride=layer[2], padding=layer[3])])
        elif layer_name == "adaptivemaxpool":
            list_to_append_layer_to.extend([nn.AdaptiveMaxPool2d(output_size=(layer[1], layer[2]))])
        elif layer_name == "adaptiveavgpool":
            list_to_append_layer_to.extend([nn.AdaptiveAvgPool2d(output_size=(layer[1], layer[2]))])
        elif layer_name == "linear":
            if isinstance(input_dim, tuple): input_dim = np.prod(np.array(input_dim))
            list_to_append_layer_to.extend([nn.Linear(in_features=input_dim, out_features=layer[1])])
        else:
            raise ValueError("Wrong layer name")
        input_dim = self.calculate_new_dimensions(input_dim, layer)
        return input_dim

    def calculate_new_dimensions(self, input_dim, layer):
        """Calculates the new dimensions of the data after passing through a type of layer"""
        layer_name = layer[0].lower()
        if layer_name == "conv":
            new_channels = layer[1]
            kernel, stride, padding = layer[2], layer[3], layer[4]
            new_height = int((input_dim[1] - kernel + 2*padding)/stride) + 1
            new_width = int((input_dim[2] - kernel + 2 * padding) / stride) + 1
            output_dim = (new_channels, new_height, new_width)
        elif layer_name in ["maxpool", "avgpool"]:
            new_channels = input_dim[0]
            kernel, stride, padding = layer[1], layer[2], layer[3]
            new_height = int((input_dim[1] - kernel + 2*padding)/stride) + 1
            new_width = int((input_dim[2] - kernel + 2 * padding) / stride) + 1
            output_dim = (new_channels, new_height, new_width)
        elif layer_name in ["adaptivemaxpool", "adaptiveavgpool"]:
            new_channels = input_dim[0]
            output_dim = (new_channels, layer[1], layer[2])
        elif layer_name == "linear":
            output_dim = layer[1]
        return output_dim

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        input_dim = self.input_dim_into_final_layer
        if not isinstance(self.layers_info[-1][0], list)  : self.layers_info[-1] = [self.layers_info[-1]]
        for output_layer in self.layers_info[-1]:
            self.create_and_append_layer(input_dim, output_layer, output_layers)
        return output_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        initialisable_layers = [layer for layer in self.hidden_layers if not type(layer) in self.valid_layer_types_with_no_parameters]
        self.initialise_parameters(nn.ModuleList(initialisable_layers))
        output_initialisable_layers = [layer for layer in self.output_layers if
                                not type(layer) in self.valid_layer_types_with_no_parameters]
        self.initialise_parameters(output_initialisable_layers)

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([])
        for layer in self.layers_info[:-1]:
            layer_type = layer[0].lower()
            if layer_type == "conv":
                batch_norm_layers.extend([nn.BatchNorm2d(num_features=layer[1])])
            elif layer_type == "linear":
                batch_norm_layers.extend([nn.BatchNorm1d(num_features=layer[1])])
        return batch_norm_layers

    def forward(self, x):
        """Forward pass for the network. Note that it expects input data in the form (Batch, Channels, Height, Width)"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x)
        x = self.process_hidden_layers(x)
        out = self.process_output_layers(x)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(out)
        return out

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        assert len(x.shape) == 4, "x should have the shape (batch_size, channel, height, width)"
        assert x.shape[1:] == self.input_dim, "Input data must be of shape (channels, height, width) that you provided, not of shape {}".format(x.shape[1:])
        self.checked_forward_input_data_once = True #So that it doesn't check again

    def process_hidden_layers(self, x):
        """Puts the data x through all the hidden layers"""
        flattened=False
        valid_batch_norm_layer_ix = 0
        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) in self.valid_layer_types_with_no_parameters:
                x = layer(x)
            else:
                if type(layer) == nn.Linear and not flattened:
                    x = self.flatten_tensor(x)
                    flattened = True
                x = self.get_activation(self.hidden_activations, layer_ix)(layer(x))
                if self.batch_norm:
                    x = self.batch_norm_layers[valid_batch_norm_layer_ix](x)
                    valid_batch_norm_layer_ix += 1
                if self.dropout != 0.0: x = self.dropout_layer(x)
        if not flattened: x = self.flatten_tensor(x)
        return x

    def process_output_layers(self, x):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            temp_output = output_layer(x)
            if activation is not None: temp_output = activation(temp_output)
            if out is None: out = temp_output
            else: out = torch.cat((out, temp_output), dim=1)
        return out