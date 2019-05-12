from abc import ABC, abstractmethod

class Base_Network(ABC):

    def __init__(self, input_dim, layers, output_activation, hidden_activations, dropout, initialiser, batch_norm,
                 y_range, random_seed, print_model_summary):
        self.set_all_random_seeds(random_seed)
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range
        if print_model_summary: self.print_model_summary()

        self.check_all_user_inputs_valid()

        self.hidden_layers = self.create_hidden_layers()
        self.output_layers = self.create_output_layers()


        if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()
        self.dropout_layer = self.create_dropout_layer()
        self.initialise_all_parameters()

        # Flag we use to run checks on the input data into forward the first time it is entered
        self.checked_forward_input_data_once = False

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
    def create_dropout_layer(self):
        """Creates the dropout layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def initialise_all_parameters(self):
        """Initialises all the parameters of the network"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data):
        """Runs a forward pass of the network"""
        raise NotImplementedError

    @abstractmethod
    def check_input_data_into_forward_once(self, input_data):
        """Checks the input data into the network is of the right form. Only runs the first time data is provided
        otherwise would slow down training too much"""
        raise NotImplementedError

    @abstractmethod
    def print_model_summary(self):
        """Prints a summary of the model"""
        raise NotImplementedError

    @abstractmethod
    def set_all_random_seeds(self, random_seed):
        """Sets all random seeds"""
        raise NotImplementedError