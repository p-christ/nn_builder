import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model
import tensorflow.python.keras.activations as activations
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Concatenate, BatchNormalization, MaxPool2D,\
                                            AveragePooling2D, GRU, LSTM
from nn_builder.tensorflow.TensorFlow_Base_Network import TensorFlow_Base_Network

# TODO add embedding layers

class RNN(Model, TensorFlow_Base_Network):
    """Creates a TensorFlow recurrent neural network
    Args:
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
        - print_model_summary: Boolean to indicate whether you want a model summary printed after model is created. Default is False.
    """
    def __init__(self, layers_info: list, output_activation=None,
                 hidden_activations="relu", dropout: float =0.0, initialiser: str ="default", batch_norm: bool =False,
                 columns_of_data_to_be_embedded: list =[], embedding_dimensions: list =[], y_range: tuple = (),
                 random_seed=0, print_model_summary: bool =False):
        Model.__init__(self)
        # self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        # self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        # self.embedding_dimensions = embedding_dimensions
        # self.embedding_layers = self.create_embedding_layers()
        self.valid_RNN_hidden_layer_types = {"linear", "gru", "lstm"}
        TensorFlow_Base_Network.__init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser,
                                         batch_norm, y_range, random_seed, print_model_summary)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_RNN_layers_valid()
        self.check_activations_valid()
        # self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def check_RNN_layers_valid(self):
        """Checks that layers provided by user are valid"""
        error_msg_layer_type = "First element in a layer specification must be one of {}".format(self.valid_RNN_hidden_layer_types)
        error_msg_layer_form = "Layer must be of form [layer_name, hidden_units]"
        error_msg_layer_list = "Layers must be provided as a list"

        assert isinstance(self.layers_info, list), error_msg_layer_list

        all_layers = self.layers_info[:-1]
        output_layer = self.layers_info[-1]
        assert isinstance(output_layer, list), error_msg_layer_list
        if isinstance(output_layer[0], list):
            for layer in output_layer:
                all_layers.append(layer)
        else:
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
        rnn_hidden_layers = []
        for layer_ix, layer in enumerate(self.layers_info[:-1]):
            activation = self.get_activation(self.hidden_activations, layer_ix)
            self.create_and_append_layer(layer, rnn_hidden_layers, activation)
        return rnn_hidden_layers

    def create_and_append_layer(self, layer, rnn_hidden_layers, activation):
        layer_type_name = layer[0].lower()
        hidden_size = layer[1]
        if layer_type_name == "lstm":
            rnn_hidden_layers.extend([LSTM(units=hidden_size, kernel_initializer=self.initialiser_function)])
        elif layer_type_name == "gru":
            rnn_hidden_layers.extend([GRU(units=hidden_size, kernel_initializer=self.initialiser_function)])
        elif layer_type_name == "linear":
            rnn_hidden_layers.extend(
                [Dense(units=hidden_size, activation=activation, kernel_initializer=self.initialiser_function)])
        else:
            raise ValueError("Wrong layer names")
        input_dim = hidden_size
        return input_dim

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = []
        if not isinstance(self.layers_info[-1][0], list): self.layers_info[-1] = [self.layers_info[-1]]
        for output_layer_ix, output_layer in enumerate(self.layers_info[-1]):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            self.create_and_append_layer(output_layer, output_layers, activation)
        return output_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = []
        for layer in self.layers_info[:-1]:
            layer_type = layer[0].lower()
            batch_norm_layers.extend([BatchNormalization()])
        return batch_norm_layers

    def call(self, x, training=True):
        """Forward pass for the network"""
        batch_size, seq_length, data_dimension = x.shape

        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) == type(Dense):
                x = tf.reshape(x, [batch_size*seq_length, -1])
                x = layer(x)
                x = tf.reshape(x, [batch_size, seq_length, -1])
            else:
                x = layer(x)
                print(x.shape)
                assert 1 == 0
                x = x[0]
            if self.batch_norm:
                x = tf.transpose(x, perm=[0, 2, 1])
                # x.transpose_(1, 2)
                x = self.batch_norm_layers[layer_ix](x, training=training)
                x = tf.transpose(x, perm=[0, 2, 1])
                # x.transpose_(1, 2)
            if self.dropout != 0.0 and (training or training is None): x = self.dropout_layer(x)

        out = None
        x = tf.reshape(x, [batch_size * seq_length, -1])
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            temp_output = output_layer(x)
            temp_output = tf.reshape(temp_output, [batch_size, seq_length, -1])
            if out is None: out = temp_output
            else: out = Concatenate(axis=2)([out, temp_output])  # out = torch.cat((out, temp_output), dim=2)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * activations.sigmoid(out)
        return out