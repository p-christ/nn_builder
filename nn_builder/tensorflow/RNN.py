import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, activations
from tensorflow.keras.layers import Dense, Concatenate, GRU, LSTM
from nn_builder.tensorflow.Base_Network import Base_Network

class RNN(Model, Base_Network):
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
        - return_final_seq_only: Boolean to indicate whether you only want to return the output for the final timestep (True)
                                 or if you want to return the output for all timesteps (False)
        - random_seed: Integer to indicate the random seed you want to use

    NOTE that this class' call method expects input data in the form: (batch, sequence length, features)
    """
    def __init__(self, layers_info, output_activation=None, hidden_activations="relu", dropout=0.0, initialiser="default",
                 batch_norm=False, columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range= (),
                 return_final_seq_only=True, random_seed=0, input_dim=None):
        Model.__init__(self)
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        self.return_final_seq_only = return_final_seq_only
        self.valid_RNN_hidden_layer_types = {"linear", "gru", "lstm"}
        Base_Network.__init__(self, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed, input_dim)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
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
            assert not isinstance(self.output_activation, list) or len(self.output_activation) == 1, error_msg_output_heads
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

    def create_and_append_layer(self, layer, rnn_hidden_layers, activation, output_layer=False):
        layer_type_name = layer[0].lower()
        hidden_size = layer[1]
        if output_layer and self.return_final_seq_only: return_sequences = False
        else: return_sequences = True
        if layer_type_name == "lstm":
            rnn_hidden_layers.extend([LSTM(units=hidden_size, kernel_initializer=self.initialiser_function,
                                           return_sequences=return_sequences)])
        elif layer_type_name == "gru":
            rnn_hidden_layers.extend([GRU(units=hidden_size, kernel_initializer=self.initialiser_function,
                                          return_sequences=return_sequences)])
        elif layer_type_name == "linear":
            rnn_hidden_layers.extend(
                [Dense(units=hidden_size, activation=activation, kernel_initializer=self.initialiser_function)])
        else:
            raise ValueError("Wrong layer names")
        input_dim = hidden_size
        return input_dim

    def call(self, x, training=True):
        """Forward pass for the network. Note that it expects input data in the form (batch, seq length, features)"""
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        training = training or training is None
        x, restricted_to_final_seq = self.process_hidden_layers(x, training)
        out = self.process_output_layers(x, restricted_to_final_seq)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * activations.sigmoid(out)
        return out

    def incorporate_embeddings(self, x):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the hidden layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, :, embedding_var]
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            all_embedded_data.append(embedded_data)
        if len(all_embedded_data) > 1: all_embedded_data = Concatenate(axis=2)(all_embedded_data)
        else: all_embedded_data = all_embedded_data[0]
        non_embedded_columns = [col for col in range(x.shape[2]) if col not in self.columns_of_data_to_be_embedded]
        if len(non_embedded_columns) > 0:
            x = tf.gather(x, non_embedded_columns, axis=2)
            x = Concatenate(axis=2)([tf.dtypes.cast(x, float), all_embedded_data])
        else: x = all_embedded_data
        return x

    def process_hidden_layers(self, x, training):
        """Puts the data x through all the hidden layers"""
        restricted_to_final_seq = False
        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) == Dense:
                if self.return_final_seq_only and not restricted_to_final_seq:
                    x = x[:, -1, :]
                    restricted_to_final_seq = True
                x = layer(x)
            else:
                x = layer(x)
            if self.batch_norm:
                x = self.batch_norm_layers[layer_ix](x, training=False)
            if self.dropout != 0.0 and training: x = self.dropout_layer(x)
        return x, restricted_to_final_seq

    def process_output_layers(self, x, restricted_to_final_seq):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            if type(output_layer) == Dense:
                if self.return_final_seq_only and not restricted_to_final_seq:
                    x = x[:, -1, :]
                    restricted_to_final_seq = True
                temp_output = output_layer(x)
            else:
                temp_output = output_layer(x)
                activation = self.get_activation(self.output_activation, output_layer_ix)
                temp_output = activation(temp_output)
            if out is None: out = temp_output
            else:
                if restricted_to_final_seq: dim = 1
                else: dim = 2
                out = Concatenate(axis=dim)([out, temp_output])
        return out
