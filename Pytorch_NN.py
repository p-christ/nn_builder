import torch.nn as nn

class Neural_Network(nn.Module):

    def __init__(self, input_dim, hidden_units, hidden_activations, output_dim, output_activation,
                 initialiser, batch_norm, cols_to_embed, embedding_dimensions=[]):
        super(Neural_Network, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_activations = hidden_activations
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.initialiser = initialiser
        self.batch_norm = batch_norm

        self.str_to_activations_converter = self.create_str_to_activations_converter()

        self.cols_to_embed = cols_to_embed
        self.embedding_dimensions = embedding_dimensions

        self.check_all_user_inputs_valid()

        self.linear_layers = nn.ModuleList([])
        if self.batch_norm: self.batch_norm_layers = nn.ModuleList([])
        self.embedding_layers = nn.ModuleList([])

        self.create_linear_and_batch_norm_layers()
        self.create_embedding_layers()

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_hidden_units_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"ELU": nn.ELU(), "Hardshrink": nn.Hardshrink(), "Hardtanh": nn.Hardtanh(),
                                        "LeakyReLU": nn.LeakyReLU(), "LogSigmoid": nn.LogSigmoid(), "PReLU": nn.PReLU(),
                                        "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6(),"RReLU": nn.RReLU(), "SELU": nn.SELU(),
                                        "CELU": nn.CELU(), "Sigmoid": nn.Sigmoid(), "Softplus": nn.Softplus(),"Softshrink": nn.Softshrink(),
                                        "Softsign": nn.Softsign(), "Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(),
                                        "Threshold": nn.Threshold(), "Softmin": nn.Softmin(), "Softmax": nn.Softmax(),
                                        "Softmax2d": nn.Softmax2d(), "LogSoftmax": nn.LogSoftmax(),
                                        "AdaptiveLogSoftmaxWithLoss": nn.AdaptiveLogSoftmaxWithLoss()}
        return str_to_activations_converter


    def check_activations_valid(self):


        assert self.output_activation, str), "Output activation must "

        for activations in [self.hidden_activations, self.output_activation]:
            assert isinstance(activations, list) or isinstance(activations, str), "Activations must be a list or a string"




    def check_hidden_units_valid(self):
        """Checks that user input for hidden_units is valid"""
        assert isinstance(self.hidden_units, list), "hidden_units must be a list"
        for hidden_unit in self.hidden_units:
            assert isinstance(hidden_unit, int), "hidden_units must be a list of integers"

    def check_embedding_dimensions_valid(self):
        """Checks that user input for embedding_dimensions is valid"""
        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"
        for embedding_dim in self.embedding_dimensions:
            assert len(embedding_dim) == 2 and isinstance(embedding_dim, set), \
                "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

    def create_linear_and_batch_norm_layers(self):
        """Creates the linear and batch norm layers in the network"""
        input_dim = self.input_dim
        for hidden_unit in self.hidden_units:
            self.linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            if self.batch_norm: self.batch_norm_layers.extend([nn.BatchNorm1d(num_features=hidden_unit)])
            input_dim = hidden_unit
        self.linear_layers.extend([nn.Linear(hidden_unit, self.output_dim)])

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            self.embedding_layers.extend([nn.Embedding(input_dim, output_dim)])

    def initialise_parameters(self):
        """Initialises all the parameters of the network"""
        for parameters in self.linear_layers + self.embedding_layers:
            if self.initialiser == "Xavier":
                nn.init.xavier_normal_(parameters.weight)
            elif self.initialiser == "He":
                nn.init.kaiming_normal_(parameters.weight)
            elif self.initialiser == "Default":
                continue
            else:
                raise NotImplementedError



    def forward(self, x):

        pass

