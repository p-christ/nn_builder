from tensorflow.keras import Model
import numpy as np
import tensorflow.keras.activations as activations

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, BatchNormalization
from nn_builder.tensorflow.TensorFlow_Base_Network import TensorFlow_Base_Network


class NN(Model, TensorFlow_Base_Network):
    """Creates a PyTorch neural network
    Args:
        - layers_info: List of integers to indicate the width and number of linear layers you want in your network
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
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
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        TensorFlow_Base_Network.__init__(self, None, layers_info, output_activation,
                                         hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed,
                                         print_model_summary)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_NN_layers_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def create_hidden_layers(self):
        """Creates the linear layers in the network"""
        hidden_layers = []
        for layer_ix, hidden_unit in enumerate(self.layers_info[:-1]):
            activation = self.get_activation(self.hidden_activations, layer_ix)
            hidden_layers.extend([Dense(hidden_unit, activation=activation, kernel_initializer=self.initialiser_function)])

        return hidden_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = []
        for _ in range(len(self.hidden_layers)):
            batch_norm_layers.append(BatchNormalization())
        return batch_norm_layers

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = []
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([tf.keras.layers.Embedding(input_dim, output_dim)])
        return embedding_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = []
        # if len(self.layers) >= 2: input_dim = self.layers[-2]
        # else: input_dim = self.input_dim
        if not isinstance(self.layers_info[-1], list): output_layer = [self.layers_info[-1]]
        else: output_layer = self.layers_info[-1]
        for output_layer_ix, output_dim in enumerate(output_layer):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            output_layers.extend([tf.keras.layers.Dense(output_dim, activation=activation,
                                                        kernel_initializer=self.initialiser_function)])
        return output_layers

    def call(self, x, training=True):
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        for layer_ix, linear_layer in enumerate(self.hidden_layers):
            x = linear_layer(x)
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            if self.dropout != 0.0 and (training or training is None):
                x = self.dropout_layer(x)
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            temp_output = output_layer(x)
            if out is None: out = temp_output
            else: out = Concatenate(axis=1)((out, temp_output))
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*activations.sigmoid(out)
        return out

    def incorporate_embeddings(self, x):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the hidden layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, embedding_var]  #.long()
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            all_embedded_data.append(embedded_data)
        all_embedded_data = Concatenate(axis=1)(all_embedded_data)
        non_embedded_columns = [col for col in range(x.shape[1]) if col not in self.columns_of_data_to_be_embedded]
        rest_of_data = tf.gather(x, non_embedded_columns, axis=1)
        x = Concatenate(axis=1)([tf.dtypes.cast(rest_of_data, float), all_embedded_data])
        return x


