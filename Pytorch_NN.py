import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

class Neural_Network(nn.Module):

    def __init__(self, input_dim, hidden_units, output_dim, initialiser, cols_to_embed, embedding_dimensions):

        super(Neural_Network, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.initialiser = initialiser

        self.cols_to_embed = cols_to_embed
        self.embedding_dimensions = embedding_dimensions

        self.check_embedding_inputs_valid()


        self.linear_layers = nn.ModuleList([])
        self.batch_norm_layers = nn.ModuleList([])
        self.embedding_layers = nn.ModuleList([])

        self.create_linear_and_batch_norm_layers()
        self.create_embedding_layers()

    def check_embedding_inputs_valid(self):

        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"

        if len(self.embedding_dimensions) > 0:
            for embedding_dim in self.embedding_dimensions:
                assert len(embedding_dim) == 2 and isinstance(embedding_dim, set), \
                    "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

                


    def create_linear_and_batch_norm_layers(self):

        input_dim = self.input_dim

        for hidden_unit in self.hidden_units:
            self.linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            self.batch_norm_layers.extend([nn.BatchNorm1d(num_features=hidden_unit)])
            input_dim = hidden_unit
        self.linear_layers.extend([nn.Linear(hidden_unit, self.output_dim)])

    def create_embedding_layers(self):

        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            self.embedding_layers.extend([nn.Embedding(input_dim, output_dim)])

    def initialise_parameters(self):

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

