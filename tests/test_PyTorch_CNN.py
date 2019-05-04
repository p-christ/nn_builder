# Run from home directory with python -m pytest tests
import pytest
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from nn_builder.pytorch.CNN import CNN


def test_user_hidden_layers_input_rejections():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_fail = [['maxpool', 33, 22, 33], [['a']], [[222, 222, 222, 222]], [["conv", 2, 2, -1]], [["conv", 2, 2]], [["conv", 2, 2, 55, 999, 33]],
                                [["maxpool", 33, 33]], [["maxpool", -1, 33]], [["maxpool", 33]], [["maxpoolX", 1, 33]],
                                [["cosnv", 2, 2]], [["avgpool", 33, 33, 333, 99]], [["avgpool", -1, 33]], [["avgpool", 33]], [["avgpoolX", 1, 33]],
                                [["adaptivemaxpool", 33, 33, 333, 33]], [["adaptivemaxpool", 2]], [["adaptivemaxpool", 33]], [["adaptivemaxpoolX"]],
                                [["adaptiveavgpool", 33, 33, 333, 11]], [["adaptiveavgpool", 2]], [["adaptiveavgpool", 33]],
                                [["adaptiveavgpoolX"]]]
    for input in inputs_that_should_fail:
        print(input)
        with pytest.raises(AssertionError):
            CNN(hidden_layers=input, hidden_activations="relu", output_dim=2,
                output_activation="relu")

def test_user_hidden_layers_input_acceptances():
    """Tests whether network rejects invalid hidden_layers inputted from user"""

    inputs_that_should_work = [[["conv", 2, 2, 3331, 2]], [["CONV", 2, 2, 3331, 22]], [["ConV", 2, 2, 3331, 1]],
                               [["maxpool", 2, 2, 3331]], [["MAXPOOL", 2, 2, 3331]], [["MaXpOOL", 2, 2, 3331]],
                               [["avgpool", 2, 2, 3331]], [["AVGPOOL", 2, 2, 3331]], [["avGpOOL", 2, 2, 3331]],
                               [["adaptiveavgpool"]], [["ADAPTIVEAVGpOOL"]], [["ADAPTIVEaVGPOOL"]],
                               [["adaptivemaxpool"]], [["ADAPTIVEMAXpOOL"]], [["ADAPTIVEmaXPOOL"]],
                               [["adaptivemaxpool"], ["ADAPTIVEMAXpOOL"]]]

    for ix, input in enumerate(inputs_that_should_work):
        CNN(hidden_layers=input, hidden_activations="relu", output_dim=2,
            output_activation="relu")
        CNN(hidden_layers=inputs_that_should_work[ix] + inputs_that_should_work[min(ix+1, len(inputs_that_should_work)-1)],
            hidden_activations="relu", output_dim=2, output_activation="relu")

def test_output_dim_user_input():
    """Tests whether network rejects an invalid output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(hidden_layers=[2], hidden_activations="relu", output_dim=input_value, output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(hidden_layers=[2], hidden_activations=input_value, output_dim=2,
                           output_activation="relu")
            CNN(hidden_layers=[2], hidden_activations="relu", output_dim=2,
                           output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(hidden_layers=[2], hidden_activations="relu", output_dim=2,
                           output_activation="relu", initialiser=input_value)

        CNN(hidden_layers=[2], hidden_activations="relu", output_dim=2,
                   output_activation="relu", initialiser="xavier")
def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = torch.randn((1, 100))
        CNN_instance = CNN(hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="relu", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)

        CNN_instance = CNN(hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="sigmoid", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)

        CNN_instance = CNN(hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=5, output_activation="softmax", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) == 1.0

        CNN_instance = CNN(hidden_layers=[5, 5, 5],
                                     hidden_activations="relu",
                                     output_dim=25)
        out = CNN_instance.forward(data)
        assert not all(out.squeeze() >= 0)
        assert not round(torch.sum(out.squeeze()).item(), 3) == 1.0
