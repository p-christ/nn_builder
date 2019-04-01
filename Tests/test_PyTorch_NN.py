import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from PyTorch_NN import Neural_Network

def test_linear_hidden_units_user_input():
    """Tests whether network rejects an invalid linear_hidden_units input from user"""
    inputs_that_should_fail = ["a", ["a", "b"], [2, 4, "ss"], [-2], 2]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=input_value, hidden_activations="relu", output_dim=2, output_activation="relu")

def test_input_dim_output_dim_user_input():
    """Tests whether network rejects an invalid input_dim or output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2])]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=input_value, linear_hidden_units=[2], hidden_activations="relu", output_dim=2, output_activation="relu")
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=input_value, output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2]), "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations=input_value, output_dim=2,
                           output_activation="relu")
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                           output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2]), "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                           output_activation="relu", initialiser=input_value)

    Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                   output_activation="relu", initialiser="xavier")

def test_output_shape_correct():
    """Tests whether network returns output of the right shape"""
    input_dims = [x for x in range(1, 3)]
    output_dims = [x for x in range(4, 6)]
    linear_hidden_units_options = [ [2, 3, 4], [2, 9, 1], [55, 55, 55, 234, 15]]
    for input_dim, output_dim, linear_hidden_units in zip(input_dims, output_dims, linear_hidden_units_options):
        nn_instance = Neural_Network(input_dim=input_dim, linear_hidden_units=linear_hidden_units, hidden_activations="relu",
                                     output_dim=output_dim, output_activation="relu", initialiser="xavier")
        data = torch.randn((25, input_dim))
        output = nn_instance.forward(data)
        assert output.shape == (25, output_dim)

def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = torch.randn((1, 100))
        nn_instance = Neural_Network(input_dim=100, linear_hidden_units=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="relu", initialiser="xavier")
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)

        nn_instance = Neural_Network(input_dim=100, linear_hidden_units=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="sigmoid", initialiser="xavier")
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)

        nn_instance = Neural_Network(input_dim=100, linear_hidden_units=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="softmax", initialiser="xavier")
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) == 1.0

        nn_instance = Neural_Network(input_dim=100, linear_hidden_units=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=25)
        out = nn_instance.forward(data)
        assert not all(out.squeeze() >= 0)
        assert not round(torch.sum(out.squeeze()).item(), 3) == 1.0

def test_linear_layers():
    """Tests whether create_linear_layers method works correctly"""
    for input_dim, output_dim, hidden_units in zip( range(5, 8), range(9, 12), [[2, 9, 2], [3, 5, 6], [9, 12, 2]]):
        nn_instance = Neural_Network(input_dim=input_dim, linear_hidden_units=hidden_units,
                                     hidden_activations="relu",
                                     output_dim=output_dim, output_activation="relu", initialiser="xavier", print_model_summary=False)

        for layer in nn_instance.linear_layers:
            assert isinstance(layer, nn.Linear)
        assert len(nn_instance.linear_layers) == len(hidden_units) + 1

        assert nn_instance.linear_layers[0].in_features == input_dim
        assert nn_instance.linear_layers[0].out_features == hidden_units[0]
        assert nn_instance.linear_layers[1].in_features == hidden_units[0]
        assert nn_instance.linear_layers[1].out_features == hidden_units[1]
        assert nn_instance.linear_layers[2].in_features == hidden_units[1]
        assert nn_instance.linear_layers[2].out_features == hidden_units[2]
        assert nn_instance.linear_layers[3].in_features == hidden_units[2]
        assert nn_instance.linear_layers[3].out_features == output_dim

def test_embedding_layers():
    """Tests whether create_embedding_layers method works correctly"""
    for embedding_in_dim_1, embedding_out_dim_1, embedding_in_dim_2, embedding_out_dim_2 in zip(range(5, 8), range(3, 6), range(1, 4), range(24, 27)):
        nn_instance = Neural_Network(input_dim=5, linear_hidden_units=[5],
                                     output_dim=5, cols_to_embed = [4, 5],
                                     embedding_dimensions =[[embedding_in_dim_1, embedding_out_dim_1], [embedding_in_dim_2, embedding_out_dim_2]])

        for layer in nn_instance.embedding_layers:
            assert isinstance(layer, nn.Embedding)
        assert len(nn_instance.embedding_layers) == 2

        assert nn_instance.embedding_layers[0].num_embeddings == embedding_in_dim_1
        assert nn_instance.embedding_layers[0].embedding_dim == embedding_out_dim_1
        assert nn_instance.embedding_layers[1].num_embeddings == embedding_in_dim_2
        assert nn_instance.embedding_layers[1].embedding_dim == embedding_out_dim_2

def test_batch_norm_layers():
    """Tests whether batch_norm_layers method works correctly"""
    for input_dim, output_dim, hidden_units in zip( range(5, 8), range(9, 12), [[2, 9, 2], [3, 5, 6], [9, 12, 2]]):
        nn_instance = Neural_Network(input_dim=input_dim, linear_hidden_units=hidden_units,
                                     hidden_activations="relu",
                                     output_dim=output_dim, output_activation="relu", initialiser="xavier", print_model_summary=False)

        for layer in nn_instance.batch_norm_layers:
            assert isinstance(layer, nn.BatchNorm1d)
        assert len(nn_instance.batch_norm_layers) == len(hidden_units)

        assert nn_instance.batch_norm_layers[0].num_features == hidden_units[0]
        assert nn_instance.batch_norm_layers[1].num_features == hidden_units[1]
        assert nn_instance.batch_norm_layers[2].num_features == hidden_units[2]


def test_model_trains():

    N = 25
    X = torch.randn((N, 3))
    y = X[:, 0] > 0
    y = y.float()

    nn_instance = Neural_Network(input_dim=3, linear_hidden_units=[10, 10, 10], output_dim=1, output_activation="sigmoid")
    optimizer = optim.Adam(nn_instance.parameters(), lr=0.1)
    for ix in range(1000):
        out = nn_instance.forward(X)
        loss = torch.sum((out - y)**2)
        # print(out)
        # print(y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)


test_model_trains()