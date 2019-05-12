# Run from home directory with python -m pytest tests
import shutil
import pytest
import torch
import random
import numpy as np
import torch.nn as nn

from nn_builder.pytorch.RNN import RNN
import torch.optim as optim
from torchvision import datasets, transforms


def test_user_hidden_layers_input_rejections():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_fail = [[["linearr", 33]], [["linear", 12, 33]], [["gru", 2, 33]], [["lstm", 2, 33]], [["lstmr", 33]],
                               [["gruu", 33]], [["gru", 33], ["xxx", 33]], [["linear", 33], ["gru", 12], ["gru", 33]] ]
    for input in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(input_dim=1, layers=input, hidden_activations="relu",
                output_activation="relu")

def test_user_hidden_layers_input_acceptances():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_work = [[["linear", 33]], [["linear", 12]], [["gru", 2]], [["lstm", 2]], [["lstm", 1]],
                               [["gru", 330]], [["gru", 33], ["linear", 2]] ]
    for input in inputs_that_should_work:
        assert  RNN(input_dim=1, layers=input, hidden_activations="relu",
                output_activation="relu")


def test_hidden_layers_created_correctly():
    """Tests that create_hidden_layers works correctly"""
    layers = [["gru", 25], ["lstm", 23], ["linear", 5], ["linear", 10]]

    rnn = RNN(input_dim=5, layers=layers, hidden_activations="relu",
              output_activation="relu")

    assert type(rnn.hidden_layers[0]) == nn.GRU
    assert rnn.hidden_layers[0].input_size == 5
    assert rnn.hidden_layers[0].hidden_size == 25

    assert type(rnn.hidden_layers[1]) == nn.LSTM
    assert rnn.hidden_layers[1].input_size == 25
    assert rnn.hidden_layers[1].hidden_size == 23

    assert type(rnn.hidden_layers[2]) == nn.Linear
    assert rnn.hidden_layers[2].in_features == 23
    assert rnn.hidden_layers[2].out_features == 5

    assert type(rnn.output_layers[0]) == nn.Linear
    assert rnn.output_layers[0].in_features == 5
    assert rnn.output_layers[0].out_features == 10


def test_output_layers_created_correctly():
    """Tests that create_output_layers works correctly"""
    layers = [["gru", 25], ["lstm", 23], ["linear", 5], ["linear", 10]]

    rnn = RNN(input_dim=5, layers=layers, hidden_activations="relu", output_activation="relu")

    assert rnn.output_layers[0].in_features == 5
    assert rnn.output_layers[0].out_features == 10

    layers = [["gru", 25], ["lstm", 23], ["lstm", 10]]

    rnn = RNN(input_dim=5, layers=layers, hidden_activations="relu",
              output_activation="relu")

    assert rnn.output_layers[0].input_size == 23
    assert rnn.output_layers[0].hidden_size == 10

    layers = [["gru", 25], ["lstm", 23], [["lstm", 10], ["linear", 15]]]
    rnn = RNN(input_dim=5, layers=layers, hidden_activations="relu",
              output_activation="relu")

    assert rnn.output_layers[0].input_size == 23
    assert rnn.output_layers[0].hidden_size == 10

    assert rnn.output_layers[1].in_features == 23
    assert rnn.output_layers[1].out_features == 15

def test_output_dim_user_input():
    """Tests whether network rejects an invalid output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(input_dim=3, layers=[2, input_value], hidden_activations="relu",  output_activation="relu")
        with pytest.raises(AssertionError):
            RNN(input_dim=6, layers=input_value, hidden_activations="relu", output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(input_dim=4, layers=[["linear", 2]], hidden_activations=input_value,
                output_activation="relu")
            RNN(input_dim=4, layers=[["linear", 2]], hidden_activations="relu",
                output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(input_dim=4, layers=[["linear", 2]], hidden_activations="relu",
                output_activation="relu", initialiser=input_value)

            RNN(layers=[["linear", 2], ["linear", 2]], hidden_activations="relu",
            output_activation="relu", initialiser="xavier", input_dim=4)

def test_batch_norm_layers():
    """Tests whether batch_norm_layers method works correctly"""
    layers = [["gru", 20], ["lstm", 3], ["linear", 4], ["linear", 10]]
    rnn = RNN(layers=layers, hidden_activations="relu", input_dim=5,
              output_activation="relu", initialiser="xavier", batch_norm=True)
    assert len(rnn.batch_norm_layers) == 3
    assert rnn.batch_norm_layers[0].num_features == 20
    assert rnn.batch_norm_layers[1].num_features == 3
    assert rnn.batch_norm_layers[2].num_features == 4

    # layers = [["gru", 20]]
    # rnn = RNN(layers=layers, hidden_activations="relu", input_dim=5,
    #           output_activation="relu", initialiser="xavier", batch_norm=True)
    # assert len(rnn.batch_norm_layers) == 0


