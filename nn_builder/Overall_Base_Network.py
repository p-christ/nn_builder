from abc import ABC, abstractmethod

class Overall_Base_Network(ABC):

    def __init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser, batch_norm,
                 y_range, random_seed):

        self.set_all_random_seeds(random_seed)
        self.input_dim = input_dim
        self.layers_info = layers_info

        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range

        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()

        self.check_all_user_inputs_valid()

        self.initialiser_function = self.str_to_initialiser_converter[initialiser.lower()]

        self.hidden_layers = self.create_hidden_layers()
        self.output_layers = self.create_output_layers()
        self.dropout_layer = self.create_dropout_layer()
        if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()

    @abstractmethod
    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        raise NotImplementedError

    @abstractmethod
    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_output_layers(self):
        """Creates the output layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_dropout_layer(self):
        """Creates the dropout layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def print_model_summary(self):
        """Prints a summary of the model"""
        raise NotImplementedError

    @abstractmethod
    def set_all_random_seeds(self, random_seed):
        """Sets all random seeds"""
        raise NotImplementedError

    def check_NN_layers_valid(self):
        """Checks that user input for hidden_units is valid"""
        assert isinstance(self.layers_info, list), "hidden_units must be a list"
        list_error_msg = "neurons must be a list of integers"
        integer_error_msg = "Every element of hidden_units must be 1 or higher"
        activation_error_msg = "The number of output activations provided should match the number of output layers"
        for neurons in self.layers_info[:-1]:
            assert isinstance(neurons, int), list_error_msg
            assert neurons > 0, integer_error_msg
        output_layer = self.layers_info[-1]
        if isinstance(output_layer, list):
            assert len(output_layer) == len(self.output_activation), activation_error_msg
            for output_dim in output_layer:
                assert isinstance(output_dim, int), list_error_msg
                assert output_dim > 0, integer_error_msg
        else:
            assert isinstance(self.output_activation, str) or self.output_activation is None, activation_error_msg
            assert isinstance(output_layer, int), list_error_msg
            assert output_layer > 0, integer_error_msg

    def check_NN_input_dim_valid(self):
        """Checks that user input for input_dim is valid"""
        assert isinstance(self.input_dim, int), "input_dim must be an integer"
        assert self.input_dim > 0, "input_dim must be 1 or higher"

    def check_activations_valid(self):
        """Checks that user input for hidden_activations and output_activation is valid"""
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        if isinstance(self.output_activation, list):
            for activation in self.output_activation:
                if activation is not None:
                    assert activation.lower() in set(valid_activations_strings), "Output activations must be string from list {}".format(valid_activations_strings)
        else:
            assert self.output_activation.lower() in set(valid_activations_strings), "Output activation must be string from list {}".format(valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations, list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(self.layers_info), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
            for activation in self.hidden_activations:
                assert isinstance(activation, str), "hidden_activations must be a string or list of strings"
                assert activation.lower() in set(valid_activations_strings), "each element in hidden_activations must be from list {}".format(valid_activations_strings)

    def check_embedding_dimensions_valid(self):
        """Checks that user input for embedding_dimensions is valid"""
        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"
        for embedding_dim in self.embedding_dimensions:
            assert len(embedding_dim) == 2 and isinstance(embedding_dim, list), \
                "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

    def check_y_range_values_valid(self):
        """Checks that user input for y_range is valid"""
        if self.y_range:
            assert isinstance(self.y_range, tuple) and len(self.y_range) == 2, "y_range must be a tuple of 2 floats or integers"
            for elem in range(2):
                assert isinstance(self.y_range[elem], float) or isinstance(self.y_range[elem], int), "y_range must be a tuple of 2 floats or integers"
            assert self.y_range[0] <= self.y_range[1], "y_range's first element must be smaller than the second element"

    def check_timesteps_to_output_valid(self):
        """Checks that user input for timesteps_to_output is valid"""
        assert self.timesteps_to_output in ["all", "last"]

    def check_initialiser_valid(self):
        """Checks that user input for initialiser is valid"""
        valid_initialisers = set(self.str_to_initialiser_converter.keys())
        assert isinstance(self.initialiser, str), "initialiser must be a string from list {}".format(valid_initialisers)
        assert self.initialiser.lower() in valid_initialisers, "initialiser must be from list {}".format(valid_initialisers)

    def check_return_final_seq_only_valid(self):
        """Checks whether user input for return_final_seq_only is a boolean and therefore valid. Only relevant for RNNs"""
        assert isinstance(self.return_final_seq_only, bool)

    def get_activation(self, activations, ix=None):
        """Gets the activation function"""
        if isinstance(activations, list):
            return self.str_to_activations_converter[str(activations[ix]).lower()]
        return self.str_to_activations_converter[str(activations).lower()]
