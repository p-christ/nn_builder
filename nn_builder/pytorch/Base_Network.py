import torch.nn as nn

class Base_Network(object):
    """Base class for PyTorch neural network classes"""
    def __init__(self):
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": nn.ELU(), "hardshrink": nn.Hardshrink(), "hardtanh": nn.Hardtanh(),
                                        "leakyrelu": nn.LeakyReLU(), "logsigmoid": nn.LogSigmoid(), "prelu": nn.PReLU(),
                                        "relu": nn.ReLU(), "relu6": nn.ReLU6(), "rrelu": nn.RReLU(), "selu": nn.SELU(),
                                        "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(), "logsoftmax": nn.LogSoftmax(),
                                        "softshrink": nn.Softshrink(), "softsign": nn.Softsign(), "tanh": nn.Tanh(),
                                        "tanhshrink": nn.Tanhshrink(), "softmin": nn.Softmin(), "softmax": nn.Softmax(dim=1),
                                        "softmax2d": nn.Softmax2d(), "none": None}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": nn.init.uniform_, "normal": nn.init.normal_,
                                        "constant": nn.init.constant_,
                                        "eye": nn.init.eye_, "dirac": nn.init.dirac_,
                                        "xavier_uniform": nn.init.xavier_uniform_, "xavier": nn.init.xavier_uniform_,
                                        "xavier_normal": nn.init.xavier_normal_,
                                        "kaiming_uniform": nn.init.kaiming_uniform_, "kaiming": nn.init.kaiming_uniform_,
                                        "kaiming_normal": nn.init.kaiming_normal_, "he": nn.init.kaiming_normal_,
                                        "orthogonal": nn.init.orthogonal_, "sparse": nn.init.sparse_, "default": "use_default"}
        return str_to_initialiser_converter

    def check_input_and_output_dim_valid(self):
        """Checks that user input for input_dim and output_dim is valid"""
        for dim in [self.input_dim, self.output_dim]:
            assert isinstance(dim, int), "input_dim and output_dim must be integers"
            assert dim > 0, "input_dim and output_dim must be 1 or higher"

    def check_linear_hidden_units_valid(self):
        """Checks that user input for hidden_units is valid"""
        assert isinstance(self.linear_hidden_units, list), "hidden_units must be a list"
        for hidden_unit in self.linear_hidden_units:
            assert isinstance(hidden_unit, int), "hidden_units must be a list of integers"
            assert hidden_unit > 0, "Every element of hidden_units must be 1 or higher"

    def check_activations_valid(self):
        """Checks that user input for hidden_activations and output_activation is valid"""
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        assert self.output_activation.lower() in set(valid_activations_strings), "Output activation must be string from list {}".format(valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations, list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(self.linear_hidden_units), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
            for activation in self.hidden_activations:
                assert isinstance(activation, str), "hidden_activations must be a string or list of strings"
                assert activation.lower() in set(valid_activations_strings), "each element in hidden_activations must be from list {}".format(valid_activations_strings)

    def check_embedding_dimensions_valid(self):
        """Checks that user input for embedding_dimensions is valid"""
        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"
        for embedding_dim in self.embedding_dimensions:
            assert len(embedding_dim) == 2 and isinstance(embedding_dim, list), \
                "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

    def check_initialiser_valid(self):
        """Checks that user input for initialiser is valid"""
        valid_initialisers = set(self.str_to_initialiser_converter.keys())
        assert isinstance(self.initialiser, str), "initialiser must be a string from list {}".format(valid_initialisers)
        assert self.initialiser.lower() in valid_initialisers, "initialiser must be from list {}".format(valid_initialisers)

    def check_y_range_values_valid(self):
        """Checks that user input for y_range is valid"""
        if self.y_range:
            assert isinstance(self.y_range, tuple) and len(self.y_range) == 2, "y_range must be a tuple of 2 floats or integers"
            for elem in range(2):
                assert isinstance(self.y_range[elem], float) or isinstance(self.y_range[elem], int), "y_range must be a tuple of 2 floats or integers"
            assert self.y_range[0] <= self.y_range[1], "y_range's first element must be smaller than the second element"

    def initialise_parameters(self, parameters_list: nn.ModuleList):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                initialiser(parameters.weight)