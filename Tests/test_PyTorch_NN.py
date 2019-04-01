import pytest
import torch

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
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2]), "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations=input_value, output_dim=2,
                           output_activation="relu")
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                           output_activation=input_value)

def test_initialiser_user_input():
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2]), "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                           output_activation="relu", initialiser=input_value)

    Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                   output_activation="relu", initialiser="xavier")

def test_output_shape_correct():


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
    nn_instance = Neural_Network(input_dim=5, linear_hidden_units=[5, 5, 5],
                                 hidden_activations="relu",
                                 output_dim=5, output_activation="relu", initialiser="xavier")

    for _ in range(20):
        data = torch.randn((1, 5))
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)

    nn_instance = Neural_Network(input_dim=5, linear_hidden_units=[5, 5, 5],
                                 hidden_activations="relu",
                                 output_dim=5, output_activation="sigmoid", initialiser="xavier")

    for _ in range(20):
        data = torch.randn((1, 5))
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)

    nn_instance = Neural_Network(input_dim=5, linear_hidden_units=[5, 5, 5],
                                 hidden_activations="relu",
                                 output_dim=5, output_activation="softmax", initialiser="xavier")

    for _ in range(20):
        data = torch.randn((1, 5))
        out = nn_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) == 1.0

    nn_instance = Neural_Network(input_dim=100, linear_hidden_units=[5, 5, 5],
                                 hidden_activations="relu",
                                 output_dim=100)

    for _ in range(20):
        data = torch.randn((1, 100))
        out = nn_instance.forward(data)
        assert not all(out.squeeze() >= 0)
        assert not round(torch.sum(out.squeeze()).item(), 3) == 1.0


