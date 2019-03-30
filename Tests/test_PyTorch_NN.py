import pytest
from PyTorch_NN import Neural_Network

def test_linear_hidden_units_user_input():
    """Tests whether network rejects an invalid linear_hidden_units input from user"""
    inputs_that_should_fail = ["a", ["a", "b"], [2, 4, "ss"], [-2], 2]
    for input in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=input, hidden_activations="relu", output_dim=2, output_activation="relu")

def test_input_dim_output_dim_user_input():
    """Tests whether network rejects an invalid input_dim or output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2])]
    for input in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=input, linear_hidden_units=[2], hidden_activations="relu", output_dim=2, output_activation="relu")
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=input, output_activation="relu")

def test_activations_user_input():
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, set([2]), "Xavier_"]
    for input in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations=input, output_dim=2,
                           output_activation="relu")
            Neural_Network(input_dim=2, linear_hidden_units=[2], hidden_activations="relu", output_dim=2,
                           output_activation=input)
