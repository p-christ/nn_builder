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
                                [["adaptiveavgpoolX"]], [["linear", 40, -2]], [["lineafr", 40, 2]]]
    for input in inputs_that_should_fail:
        print(input)
        with pytest.raises(AssertionError):
            CNN(input_dim=1, hidden_layers_info=input, hidden_activations="relu", output_dim=2,
                output_activation="relu", output_layer_input_dim=2)

def test_user_hidden_layers_input_acceptances():
    """Tests whether network rejects invalid hidden_layers inputted from user"""

    inputs_that_should_work = [[["conv", 2, 2, 3331, 2]], [["CONV", 2, 2, 3331, 22]], [["ConV", 2, 2, 3331, 1]],
                               [["maxpool", 2, 2, 3331]], [["MAXPOOL", 2, 2, 3331]], [["MaXpOOL", 2, 2, 3331]],
                               [["avgpool", 2, 2, 3331]], [["AVGPOOL", 2, 2, 3331]], [["avGpOOL", 2, 2, 3331]],
                               [["adaptiveavgpool", 3, 22]], [["ADAPTIVEAVGpOOL", 1, 33]], [["ADAPTIVEaVGPOOL", 3, 6]],
                               [["adaptivemaxpool", 4, 66]], [["ADAPTIVEMAXpOOL", 2, 2]], [["ADAPTIVEmaXPOOL", 3, 3]],
                               [["adaptivemaxpool", 3, 3], ["ADAPTIVEMAXpOOL", 3, 1]],  [["linear", 40, 2]], [["lineaR", 4, 2]],
                               [["LINEAR", 4, 2]]]
    for ix, input in enumerate(inputs_that_should_work):
        print(input)
        CNN(input_dim=1, hidden_layers_info=input, hidden_activations="relu", output_dim=2,
            output_activation="relu", output_layer_input_dim=2)
        CNN(input_dim=1, hidden_layers_info=inputs_that_should_work[ix] + inputs_that_should_work[min(ix + 1, len(inputs_that_should_work) - 1)],
            hidden_activations="relu", output_dim=2, output_activation="relu", output_layer_input_dim=2)

def test_hidden_layers_created_correctly():
    """Tests that create_hidden_layers works correctly"""
    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["avgpool", 32, 42, 22], ["adaptivemaxpool", 3, 34],
              ["adaptiveavgpool", 23, 44], ["linear", 2, 22], ["linear", 222, 2222]]

    cnn = CNN(input_dim=3, hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
              output_activation="relu")

    assert cnn.hidden_layers[0].in_channels == 3
    assert cnn.hidden_layers[0].out_channels == 2
    assert cnn.hidden_layers[0].kernel_size == (4, 4)
    assert cnn.hidden_layers[0].stride == (3, 3)
    assert cnn.hidden_layers[0].padding == (2, 2)

    assert cnn.hidden_layers[1].kernel_size == 3
    assert cnn.hidden_layers[1].stride == 4
    assert cnn.hidden_layers[1].padding == 2

    assert cnn.hidden_layers[2].kernel_size == 32
    assert cnn.hidden_layers[2].stride == 42
    assert cnn.hidden_layers[2].padding == 22

    assert cnn.hidden_layers[3].output_size == (3, 34)
    assert cnn.hidden_layers[4].output_size == (23, 44)
    assert cnn.hidden_layers[5].in_features == 2
    assert cnn.hidden_layers[5].out_features == 22

    assert cnn.hidden_layers[6].in_features == 222
    assert cnn.hidden_layers[6].out_features == 2222


def test_output_layers_created_correctly():
    """Tests that create_output_layers works correctly"""
    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["avgpool", 32, 42, 22], ["adaptivemaxpool", 3, 34],
              ["adaptiveavgpool", 23, 44], ["linear", 2, 22], ["linear", 222, 2222]]

    cnn = CNN(input_dim=3, hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
              output_activation="relu")

    assert cnn.output_layers[0].in_features == 2222
    assert cnn.output_layers[0].out_features == 2

    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["avgpool", 32, 42, 22], ["adaptivemaxpool", 3, 34],
              ["adaptiveavgpool", 23, 44]]

    cnn = CNN(input_dim=3, hidden_layers_info=layers, hidden_activations="relu", output_dim=7,
              output_activation="relu")

    assert cnn.output_layers[0].in_features == 23 * 44
    assert cnn.output_layers[0].out_features == 7

    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["avgpool", 32, 42, 22], ["adaptivemaxpool", 3, 34]]


    cnn = CNN(input_dim=3, hidden_layers_info=layers, hidden_activations="relu", output_dim=6,
              output_activation="relu")

    assert cnn.output_layers[0].in_features == 3 * 34
    assert cnn.output_layers[0].out_features == 6

    cnn = CNN(input_dim=3, hidden_layers_info=layers, hidden_activations="relu", output_dim=[6, 22],
              output_activation=["softmax", None])

    assert cnn.output_layers[0].in_features == 3 * 34
    assert cnn.output_layers[0].out_features == 6
    assert cnn.output_layers[1].in_features == 3 * 34
    assert cnn.output_layers[1].out_features == 22

def test_output_dim_user_input():
    """Tests whether network rejects an invalid output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(input_dim=2, hidden_layers_info=[2], hidden_activations="relu", output_dim=input_value, output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(hidden_layers_info=[["conv", 2, 2, 3331, 2]], hidden_activations=input_value, output_dim=2,
                output_activation="relu")
            CNN(hidden_layers_info=[["conv", 2, 2, 3331, 2]], hidden_activations="relu", output_dim=2,
                output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(hidden_layers_info=[["conv", 2, 2, 3331, 2]], hidden_activations="relu", output_dim=2,
                output_activation="relu", initialiser=input_value, output_layer_input_dim=2)

        CNN(hidden_layers_info=[["conv", 2, 2, 3331, 2]], hidden_activations="relu", output_dim=2,
            output_activation="relu", initialiser="xavier", output_layer_input_dim=2)

def test_batch_norm_layers():
    """Tests whether batch_norm_layers method works correctly"""
    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["adaptivemaxpool", 3, 34]]
    cnn = CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
            output_activation="relu", initialiser="xavier", batch_norm=False, output_layer_input_dim=2)
    with pytest.raises(AttributeError):
        print(cnn.batch_norm_layers)

    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["adaptivemaxpool", 3, 34]]
    cnn = CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
              output_activation="relu", initialiser="xavier", batch_norm=True, output_layer_input_dim=2)
    assert len(cnn.batch_norm_layers) == 1
    assert cnn.batch_norm_layers[0].num_features == 2


    layers = [["conv", 2, 4, 3, 2], ["maxpool", 3, 4, 2], ["conv", 12, 4, 3, 2], ["adaptivemaxpool", 3, 34], ["linear", 22, 55]]
    cnn = CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
              output_activation="relu", initialiser="xavier", batch_norm=True, output_layer_input_dim=2)
    assert len(cnn.batch_norm_layers) == 3
    assert cnn.batch_norm_layers[0].num_features == 2
    assert cnn.batch_norm_layers[1].num_features == 12
    assert cnn.batch_norm_layers[2].num_features == 55

def test_linear_layers_only_come_at_end():
    """Tests that it throws an error if user tries to provide list of hidden layers that include linear layers where they
    don't only come at the end"""
    layers = [["conv", 2, 4, 3, 2], ["linear", 22, 55], ["maxpool", 3, 4, 2], ["adaptivemaxpool", 3, 34]]
    with pytest.raises(AssertionError):
        cnn = CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
                  output_activation="relu", initialiser="xavier", batch_norm=True, output_layer_input_dim=2)

    layers = [["conv", 2, 4, 3, 2], ["linear", 22, 55]]
    assert CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
                      output_activation="relu", initialiser="xavier", batch_norm=True, output_layer_input_dim=2)

    layers = [["conv", 2, 4, 3, 2], ["linear", 22, 55], ["linear", 22, 55], ["linear", 22, 55]]
    assert CNN(hidden_layers_info=layers, hidden_activations="relu", output_dim=2,
               output_activation="relu", initialiser="xavier", batch_norm=True, output_layer_input_dim=2)

def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = torch.randn((1, 1, 20, 20))
        CNN_instance = CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                           hidden_activations="relu",
                           output_dim=5, output_activation="relu", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)

        CNN_instance = CNN(hidden_layers_info=[["conv", 2, 20, 1, 0]],
                           hidden_activations="relu", output_layer_input_dim=2,
                           output_dim=5, output_activation="relu", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)

        CNN_instance = CNN(hidden_layers_info=[["conv", 5, 20, 1, 0], ["linear", 5, 22]],
                           hidden_activations="relu",
                           output_dim=5, output_activation="relu", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)

        CNN_instance = CNN(hidden_layers_info=[["conv", 5, 20, 1, 0], ["linear", 5, 22]],
                           hidden_activations="relu",
                           output_dim=5, output_activation="sigmoid", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) != 1.0

        CNN_instance = CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                           hidden_activations="relu",
                           output_dim=5, output_activation="softmax", initialiser="xavier")
        out = CNN_instance.forward(data)
        assert all(out.squeeze() >= 0)
        assert all(out.squeeze() <= 1)
        assert round(torch.sum(out.squeeze()).item(), 3) == 1.0


        CNN_instance = CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                           hidden_activations="relu",
                           output_dim=5, initialiser="xavier")
        out = CNN_instance.forward(data)
        assert not all(out.squeeze() >= 0)
        assert not round(torch.sum(out.squeeze()).item(), 3) == 1.0


def test_y_range():
    """Tests whether setting a y range works correctly"""
    for _ in range(100):
        val1 = random.random() - 3.0*random.random()
        val2 = random.random() + 2.0*random.random()
        lower_bound = min(val1, val2)
        upper_bound = max(val1, val2)
        CNN_instance = CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                           hidden_activations="relu", y_range=(lower_bound, upper_bound),
                           output_dim=5, initialiser="xavier")
        random_data = torch.randn((10, 1, 20, 20))
        out = CNN_instance.forward(random_data)
        assert torch.sum(out > lower_bound).item() == 10*5, "lower {} vs. {} ".format(lower_bound, out)
        assert torch.sum(out < upper_bound).item() == 10*5, "upper {} vs. {} ".format(upper_bound, out)

def test_deals_with_None_activation():
    """Tests whether is able to handle user inputting None as output activation"""
    assert CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                           hidden_activations="relu", output_activation=None,
                           output_dim=5, initialiser="xavier")

def test_check_input_data_into_forward_once():
    """Tests that check_input_data_into_forward_once method only runs once"""
    CNN_instance = CNN(hidden_layers_info=[["conv", 2, 2, 1, 2], ["adaptivemaxpool", 2, 2]],
                       hidden_activations="relu",
                       output_dim=5, output_activation="relu", initialiser="xavier")

    data_not_to_throw_error = torch.randn((1, 1, 20, 20))
    data_to_throw_error = torch.randn((1, 2, 20, 20))

    with pytest.raises(AssertionError):
        CNN_instance.forward(data_to_throw_error)
    with pytest.raises(RuntimeError):
        CNN_instance.forward(data_not_to_throw_error)
        CNN_instance.forward(data_to_throw_error)

