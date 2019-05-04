# Run from home directory with python -m pytest tests
import pytest
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from nn_builder.pytorch.CNN import CNN


def test_input_dim_output_dim_user_input():
    """Tests whether network rejects an invalid input_dim or output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(input_dim=input_value, hidden_layers=[2], hidden_activations="relu", output_dim=2, output_activation="relu")
        with pytest.raises(AssertionError):
            CNN(input_dim=2, hidden_layers=[2], hidden_activations="relu", output_dim=input_value, output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(input_dim=2, hidden_layers=[2], hidden_activations=input_value, output_dim=2,
                           output_activation="relu")
            CNN(input_dim=2, hidden_layers=[2], hidden_activations="relu", output_dim=2,
                           output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(input_dim=2, hidden_layers=[2], hidden_activations="relu", output_dim=2,
                           output_activation="relu", initialiser=input_value)

        CNN(input_dim=2, hidden_layers=[2], hidden_activations="relu", output_dim=2,
                   output_activation="relu", initialiser="xavier")
def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = torch.randn((1, 100))
        CNN_instance = CNN(input_dim=100, hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="relu", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)

        CNN_instance = CNN(input_dim=100, hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="sigmoid", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)

        CNN_instance = CNN(input_dim=100, hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="softmax", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) == 1.0

        CNN_instance = CNN(input_dim=100, hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=25)
        out = CNN_instance.forward(data)
        assert not all(out.squeeze() >= 0)
        assert not round(torch.sum(out.squeeze()).item(), 3) == 1.0
