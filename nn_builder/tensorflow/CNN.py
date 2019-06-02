import numpy as np
from tensorflow.keras import Model, activations
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, BatchNormalization, MaxPool2D, AveragePooling2D
from nn_builder.tensorflow.Base_Network import Base_Network
import tensorflow as tf

class CNN(Model, Base_Network):
    """Creates a PyTorch convolutional neural network
    Args:
        - layers_info: List of layer specifications to specify the hidden layers of the network. Each element of the list must be
                         one of these 4 forms:
                         - ["conv", channels, kernel_size, stride, padding]
                         - ["maxpool", kernel_size, stride, padding]
                         - ["avgpool", kernel_size, stride, padding]
                         - ["linear", out]
                        where all variables are integers except for padding which must be either "same" or "valid"
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

    NOTE that this class' call method expects input data in the form: (batch, channels, height, width)
    """
    def __init__(self, layers_info, output_activation=None, hidden_activations="relu", dropout= 0.0, initialiser="default",
                 batch_norm=False, y_range=(), random_seed=0, input_dim=None):
        Model.__init__(self)
        self.valid_cnn_hidden_layer_types = {'conv', 'maxpool', 'avgpool', 'linear'}
        self.valid_layer_types_with_no_parameters = (MaxPool2D, AveragePooling2D)
        Base_Network.__init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed, input_dim)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_CNN_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def check_CNN_layers_valid(self):
        """Checks that the user inputs for cnn_hidden_layers were valid. cnn_hidden_layers must be a list of layers where
        each layer must be of one of these forms:
        - ["conv", channels, kernel_size, stride, padding]
        - ["maxpool", kernel_size, stride, padding]
        - ["avgpool", kernel_size, stride, padding]
        - ["linear", out]

        where all variables must be integers except padding which must be "valid" or "same"
        """
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_cnn_hidden_layer_types)
        error_msg_conv_layer = """Conv layer must be of form ['conv', channels, kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_maxpool_layer = """Maxpool layer must be of form ['maxpool', kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_avgpool_layer = """Avgpool layer must be of form ['avgpool', kernel_size, stride, padding] where the 
                               variables are all non-negative integers except padding which must be either "valid" or "same"""
        error_msg_linear_layer = """Linear layer must be of form ['linear', out] where out is a non-negative integers"""
        assert isinstance(self.layers_info, list), "layers must be a list"

        all_layers = self.layers_info[:-1]
        output_layer = self.layers_info[-1]
        assert isinstance(output_layer, list), "layers must be a list"
        if isinstance(output_layer[0], list):
            assert len(output_layer) == len(self.output_activation), "Number of output activations must equal number of output heads"
            for layer in output_layer:
                all_layers.append(layer)
                assert isinstance(layer[0], str), error_msg_layer_type
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
                assert isinstance(layer[4], str) and layer[4].lower() in ["valid", "same"], error_msg_conv_layer
            elif layer_type_name == "maxpool":
                assert len(layer) == 4, error_msg_maxpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_maxpool_layer
                if layer[1] != layer[2]: print("NOTE that your maxpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
                assert isinstance(layer[3], str) and layer[3].lower() in ["valid", "same"], error_msg_maxpool_layer
            elif layer_type_name == "avgpool":
                assert len(layer) == 4, error_msg_avgpool_layer
                for ix in range(2): assert isinstance(layer[ix + 1], int) and layer[ix + 1] > 0, error_msg_avgpool_layer
                assert isinstance(layer[3], str) and layer[3].lower() in ["valid", "same"], error_msg_avgpool_layer
                if layer[1] != layer[2]:print("NOTE that your avgpool kernel size {} isn't the same as your stride {}".format(layer[1], layer[2]))
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

    def create_and_append_layer(self, layer, list_to_append_layer_to, activation=None, output_layer=False):
        """Creates and appends a layer to the list provided"""
        layer_name = layer[0].lower()
        assert layer_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(
            layer_name, self.valid_cnn_hidden_layer_types)
        if layer_name == "conv":
            list_to_append_layer_to.extend([Conv2D(filters=layer[1], kernel_size=layer[2],
                                                strides=layer[3], padding=layer[4], activation=activation,
                                                   kernel_initializer=self.initialiser_function)])
        elif layer_name == "maxpool":
            list_to_append_layer_to.extend([MaxPool2D(pool_size=(layer[1], layer[1]),
                                                   strides=(layer[2], layer[2]), padding=layer[3])])
        elif layer_name == "avgpool":
            list_to_append_layer_to.extend([AveragePooling2D(pool_size=(layer[1], layer[1]),
                                                   strides=(layer[2], layer[2]), padding=layer[3])])
        elif layer_name == "linear":
            list_to_append_layer_to.extend([Dense(layer[1], activation=activation, kernel_initializer=self.initialiser_function)])
        else:
            raise ValueError("Wrong layer name")

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = []
        for layer in self.layers_info[:-1]:
            layer_type = layer[0].lower()
            if layer_type in ["conv", "linear"]:
                batch_norm_layers.extend([BatchNormalization()])
        return batch_norm_layers

    def call(self, x, training=True):
        """Forward pass for the network. Note that it expects input data in the form (Batch, Height, Width, Channels)"""
        x = self.process_hidden_layers(x, training)
        out = self.process_output_layers(x)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * activations.sigmoid(out)
        return out

    def process_hidden_layers(self, x, training):
        """Puts the data x through all the hidden layers"""
        flattened=False
        training = training or training is None
        valid_batch_norm_layer_ix = 0
        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) in self.valid_layer_types_with_no_parameters:
                x = layer(x)
            else:
                if type(layer) == Dense and not flattened:
                    x = Flatten()(x)
                    flattened = True
                x = layer(x)
                if self.batch_norm:
                    x = self.batch_norm_layers[valid_batch_norm_layer_ix](x, training=False)
                    valid_batch_norm_layer_ix += 1
                if self.dropout != 0.0 and training: x = self.dropout_layer(x)
        if not flattened: x = Flatten()(x)
        return x

    def process_output_layers(self, x):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            temp_output = output_layer(x)
            if out is None: out = temp_output
            else: out = Concatenate(axis=1)([out, temp_output])
        return out
