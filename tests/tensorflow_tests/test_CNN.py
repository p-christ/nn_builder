# Run from home directory with python -m pytest tests
import shutil
import pytest
import random
import numpy as np
import tensorflow as tf
import torch.nn as nn
from nn_builder.tensorflow.CNN import CNN
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, BatchNormalization, MaxPool2D, AveragePooling2D


N = 250
X = np.random.random((N, 5, 5, 1))
X[0:125, 3, 3, 0] += 20.0
y = X[:, 3, 3, 0] > 5.0

def test_user_hidden_layers_input_rejections():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_fail = [ ('maxpool', 3, 3, 3) , ['maxpool', 33, 22, 33], [['a']], [[222, 222, 222, 222]], [["conv", 2, 2, -1]], [["conv", 2, 2]], [["conv", 2, 2, 55, 999, 33]],
                                [["maxpool", 33, 33]], [["maxpool", -1, 33]], [["maxpool", 33]], [["maxpoolX", 1, 33]],
                                [["cosnv", 2, 2]], [["avgpool", 33, 33, 333, 99]], [["avgpool", -1, 33]], [["avgpool", 33]], [["avgpoolX", 1, 33]],
                                [["adaptivemaxpool", 33, 33, 333, 33]], [["adaptivemaxpool", 2]], [["adaptivemaxpool", 33]], [["adaptivemaxpoolX"]],
                                [["adaptiveavgpool", 33, 33, 333, 11]], [["adaptiveavgpool", 2]], [["adaptiveavgpool", 33]],
                                [["adaptiveavgpoolX"]], [["linear", 40, -2]], [["lineafr", 40, 2]]]
    for input in inputs_that_should_fail:
        print(input)
        with pytest.raises(AssertionError):
            CNN(layers_info=input, hidden_activations="relu",
                output_activation="relu")

def test_user_hidden_layers_input_acceptances():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_work = [[["conv", 2, 2, 3331, "VALID"]], [["CONV", 2, 2, 3331, "SAME"]], [["ConV", 2, 2, 3331, "valid"]],
                               [["maxpool", 2, 2, "same"]], [["MAXPOOL", 2, 2, "Valid"]], [["MaXpOOL", 2, 2, "SAme"]],
                               [["avgpool", 2, 2, "saME"]], [["AVGPOOL", 2, 2, "vaLID"]], [["avGpOOL", 2, 2, "same"]],
                               [["linear", 40]], [["lineaR", 2]], [["LINEAR", 2]]]

    for ix, input in enumerate(inputs_that_should_work):
        input.append(["linear", 5])
        CNN(layers_info=input, hidden_activations="relu",
            output_activation="relu")

def test_hidden_layers_created_correctly():
    """Tests that create_hidden_layers works correctly"""
    layers = [["conv", 2, 4, 3, "same"], ["maxpool", 3, 4, "vaLID"], ["avgpool", 32, 42, "vaLID"],
               ["linear", 22], ["linear", 2222], ["linear", 5]]

    cnn = CNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu")

    assert type(cnn.hidden_layers[0]) == Conv2D
    assert cnn.hidden_layers[0].filters == 2
    assert cnn.hidden_layers[0].kernel_size == (4, 4)
    assert cnn.hidden_layers[0].strides == (3, 3)
    assert cnn.hidden_layers[0].padding == "same"

    assert type(cnn.hidden_layers[1]) == MaxPool2D
    assert cnn.hidden_layers[1].pool_size == (3, 3)
    assert cnn.hidden_layers[1].strides == (4, 4)
    assert cnn.hidden_layers[1].padding == "valid"

    assert type(cnn.hidden_layers[2]) == AveragePooling2D
    assert cnn.hidden_layers[2].pool_size == (32, 32)
    assert cnn.hidden_layers[2].strides == (42, 42)
    assert cnn.hidden_layers[2].padding == "valid"

    assert type(cnn.hidden_layers[3]) == Dense
    assert cnn.hidden_layers[3].units == 22

    assert type(cnn.hidden_layers[4]) == Dense
    assert cnn.hidden_layers[4].units == 2222

    assert type(cnn.output_layers[0]) == Dense
    assert cnn.output_layers[0].units == 5


def test_output_layers_created_correctly():
    """Tests that create_output_layers works correctly"""
    layers = [["conv", 2, 4, 3, "valid"], ["maxpool", 3, 4, "same"], ["avgpool", 32, 42, "valid"],
              ["linear", 22], ["linear", 2222], ["linear", 2]]

    cnn = CNN(layers_info=layers, hidden_activations="relu", output_activation="relu")
    assert cnn.output_layers[0].units == 2

    layers = [["conv", 2, 4, 3,"valid"], ["maxpool", 3, 4, "same"], ["avgpool", 32, 42, "same"], ["linear", 7]]
    cnn = CNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu")
    assert cnn.output_layers[0].units == 7

    layers = [["conv", 5, 4, 3, "valid"], ["maxpool", 3, 4, "valid"], ["avgpool", 32, 42, "valid"], ["linear", 6]]
    cnn = CNN( layers_info=layers, hidden_activations="relu",
              output_activation="relu")
    assert cnn.output_layers[0].units == 6

    layers = [["conv", 5, 4, 3, "valid"], ["maxpool", 3, 4, "valid"], ["avgpool", 32, 42, "valid"],
              [["linear", 6], ["linear", 22]]]
    cnn = CNN(layers_info=layers, hidden_activations="relu",
              output_activation=["softmax", None])
    assert cnn.output_layers[0].units == 6
    assert cnn.output_layers[1].units == 22

def test_output_dim_user_input():
    """Tests whether network rejects an invalid output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(layers_info=[2, input_value], hidden_activations="relu",  output_activation="relu")
        with pytest.raises(AssertionError):
            CNN(layers_info=input_value, hidden_activations="relu", output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(layers_info=[["conv", 2, 2, 3331, "valid"], ["linear", 5] ], hidden_activations=input_value,
                output_activation="relu")
            CNN(layers_info=[["conv", 2, 2, 3331, "valid"], ["linear", 3]], hidden_activations="relu",
                output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            CNN(layers_info=[["conv", 2, 2, 3331, "valid"], ["linear", 3]], hidden_activations="relu",
                output_activation="relu", initialiser=input_value)

        CNN(layers_info=[["conv", 2, 2, 3331, "valid"], ["linear", 3]], hidden_activations="relu",
            output_activation="relu", initialiser="xavier")

def test_batch_norm_layers():
    """Tests whether batch_norm_layers method works correctly"""
    layers =[["conv", 2, 4, 3, "valid"], ["maxpool", 3, 4, "valid"], ["linear", 5]]
    cnn = CNN(layers_info=layers, hidden_activations="relu",
            output_activation="relu", initialiser="xavier", batch_norm=False)

    layers = [["conv", 2, 4, 3, "valid"], ["maxpool", 3, 4, "valid"], ["linear", 5]]
    cnn = CNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu", initialiser="xavier", batch_norm=True)
    assert len(cnn.batch_norm_layers) == 1
    assert isinstance(cnn.batch_norm_layers[0], tf.keras.layers.BatchNormalization)

    layers = [["conv", 2, 4, 3, "valid"], ["maxpool", 3, 4, "valid"], ["conv", 12, 4, 3, "valid"], ["linear", 22], ["linear", 55]]
    cnn = CNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu", initialiser="xavier", batch_norm=True)
    assert len(cnn.batch_norm_layers) == 3
    for layer in cnn.batch_norm_layers:
        assert isinstance(layer, tf.keras.layers.BatchNormalization)

def test_linear_layers_acceptance():
    """Tests that only accepts linear layers of correct shape"""
    layers_that_shouldnt_work = [[["linear", 2, 5]], [["linear", 2, 5, 5]], [["linear"]], [["linear", 2], ["linear", 5, 4]],
                                 ["linear", 0], ["linear", -5]]
    for layers in layers_that_shouldnt_work:
        with pytest.raises(AssertionError):
            cnn = CNN(layers_info=layers, hidden_activations="relu",
                      output_activation="relu", initialiser="xavier", batch_norm=True)
    layers_that_should_work = [[["linear", 44], ["linear", 2]], [["linear", 22]]]
    for layer in layers_that_should_work:
        assert CNN(layers_info=layer, hidden_activations="relu",
                      output_activation="relu", initialiser="xavier", batch_norm=True)

def test_linear_layers_only_come_at_end():
    """Tests that it throws an error if user tries to provide list of hidden layers that include linear layers where they
    don't only come at the end"""
    layers = [["conv", 2, 4, 3, "valid"], ["linear", 55], ["maxpool", 3, 4, "valid"]]
    with pytest.raises(AssertionError):
        cnn = CNN(layers_info=layers, hidden_activations="relu",
                  output_activation="relu", initialiser="xavier", batch_norm=True)

    layers = [["conv", 2, 4, 3, "valid"], ["linear", 55]]
    assert CNN(layers_info=layers, hidden_activations="relu",
                      output_activation="relu", initialiser="xavier", batch_norm=True)

    layers = [["conv", 2, 4, 3, "valid"], ["linear", 55], ["linear", 55], ["linear", 55]]
    assert CNN(layers_info=layers, hidden_activations="relu",
               output_activation="relu", initialiser="xavier", batch_norm=True)

def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    input_dim = (100, 100, 5)
    for _ in range(RANDOM_ITERATIONS):
        data = np.random.random((1, *input_dim))
        CNN_instance = CNN(layers_info=[["conv", 2, 2, 1, "valid"],  ["linear", 50]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = CNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        CNN_instance = CNN(layers_info=[["conv", 2, 20, 1, "same"], ["linear", 5]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = CNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        CNN_instance = CNN(layers_info=[["conv", 5, 20, 1, "same"], ["linear", 5]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = CNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        CNN_instance = CNN(layers_info=[["conv", 5, 2, 1, "valid"], ["linear",  22]],
                           hidden_activations="relu",
                           output_activation="sigmoid", initialiser="xavier")
        out = CNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        assert not np.round(tf.reduce_sum(out, axis=1), 3) == 1.0

        CNN_instance = CNN(layers_info=[["conv", 2, 2, 1, "same"], ["linear", 5]],
                           hidden_activations="relu",
                           output_activation="softmax", initialiser="xavier")
        out = CNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        assert np.round(tf.reduce_sum(out, axis=1), 3) == 1.0


        CNN_instance = CNN(layers_info=[["conv", 2, 2, 1, "valid"], ["linear", 5]],
                           hidden_activations="relu",
                           initialiser="xavier")
        out = CNN_instance(data)
        assert not all(tf.reshape(out, [-1]) >= 0)
        assert not np.round(tf.reduce_sum(out, axis=1), 3) == 1.0


def test_y_range():
    """Tests whether setting a y range works correctly"""
    for _ in range(100):
        val1 = random.random() - 3.0*random.random()
        val2 = random.random() + 2.0*random.random()
        lower_bound = min(val1, val2)
        upper_bound = max(val1, val2)
        CNN_instance = CNN(layers_info=[["conv", 2, 2, 1, "valid"], ["linear", 5]],
                           hidden_activations="relu", y_range=(lower_bound, upper_bound),
                           initialiser="xavier")
        random_data = np.random.random((10, 1, 20, 20))
        out = CNN_instance(random_data)
        assert all(tf.reshape(out, [-1]) > lower_bound)
        assert all(tf.reshape(out, [-1]) < upper_bound)

def test_deals_with_None_activation():
    """Tests whether is able to handle user inputting None as output activation"""
    assert CNN(layers_info=[["conv", 2, 2, 1, "valid"], ["linear", 5]],
                           hidden_activations="relu", output_activation=None,
                           initialiser="xavier")

def test_y_range_user_input():
    """Tests whether network rejects invalid y_range inputs"""
    invalid_y_range_inputs = [ (4, 1), (2, 4, 8), [2, 4], (np.array(2.0), 6.9)]
    for y_range_value in invalid_y_range_inputs:
        with pytest.raises(AssertionError):
            print(y_range_value)
            CNN_instance = CNN(layers_info=[["conv", 2, 2, 1, "valid"], ["linear", 5]],
                           hidden_activations="relu", y_range=y_range_value,
                           initialiser="xavier")

def test_model_trains():
    """Tests whether a small range of networks can solve a simple task"""
    for output_activation in ["sigmoid", "None"]:
        CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"],  ["linear", 1]],
                           hidden_activations="relu", output_activation=output_activation,
                           initialiser="xavier")
        print(CNN_instance.hidden_layers[0].kernel_size)
        assert solves_simple_problem(X, y, CNN_instance)


def test_model_trains_part_2():
    """Tests whether a small range of networks can solve a simple task"""
    z = X[:, 3:4, 3:4, 0:1] > 5.0
    z = np.concatenate([z == 1, z == 0], axis=1)
    z = z.reshape((-1, 2))
    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 2]],
                           hidden_activations="relu", output_activation="softmax", dropout=0.01,
                           initialiser="xavier")
    assert solves_simple_problem(X, z, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "same"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier", batch_norm=True)
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["maxpool", 1, 1, "same"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "same"], ["avgpool", 1, 1, "same"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 5, 3, 1, "same"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["conv", 5, 3, 1, "valid"], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)


def solves_simple_problem(X, y, nn_instance):
    """Checks if a given network is able to solve a simple problem"""
    nn_instance.compile(optimizer='adam',
                  loss='mse')
    nn_instance.fit(X, y, epochs=800)
    results = nn_instance.evaluate(X, y)
    print("FINAL RESULT ", results)
    return results < 0.1


def test_model_trains_linear_layer():
    """Tests whether a small range of networks can solve a simple task"""
    CNN_instance = CNN(layers_info=[["conv", 5, 3, 1, "valid"], ["linear", 5], ["linear", 5], ["linear", 1]],
                       hidden_activations="relu", output_activation="sigmoid",
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

    CNN_instance = CNN(layers_info=[["linear", 5], ["linear", 5], ["linear", 1]],
                       hidden_activations="relu", output_activation="sigmoid",
                       initialiser="xavier")
    assert solves_simple_problem(X, y, CNN_instance)

def test_max_pool_working():
    """Tests whether max pool layers work properly"""
    N = 250
    X = np.random.random((N, 8, 8, 1))
    X[0:125, 3, 3, 0] = 999.99
    CNN_instance = CNN(layers_info=[["maxpool", 2, 2, "valid"], ["maxpool", 2, 2, "valid"], ["maxpool", 2, 2, "valid"], ["linear", 1]],
                       hidden_activations="relu",
                       initialiser="xavier")
    assert CNN_instance(X).shape == (N, 1)

def test_dropout():
    """Tests whether dropout layer reads in probability correctly"""
    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                           hidden_activations="relu", output_activation="sigmoid", dropout=0.9999,
                           initialiser="xavier")
    assert CNN_instance.dropout_layer.rate == 0.9999
    assert not solves_simple_problem(X, y, CNN_instance)
    CNN_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                           hidden_activations="relu", output_activation=None, dropout=0.0000001,
                           initialiser="xavier")
    assert CNN_instance.dropout_layer.rate == 0.0000001
    assert solves_simple_problem(X, y, CNN_instance)

def test_MNIST_progress():
    """Tests that network made using CNN module can make progress on MNIST"""
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create model using nn_builder
    model = CNN(layers_info=[["conv", 32, 3, 1, "valid"], ["maxpool", 2, 2, "valid"], ["conv", 64, 3, 1, "valid"],
                             ["linear", 10]],
                hidden_activations="relu", output_activation="softmax", dropout=0.0,
                initialiser="xavier", batch_norm=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2, batch_size=64)


    model.evaluate(x_test, y_test)

    results = model.evaluate(x_test, y_test)
    assert results[1] > 0.9

    model = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 10]],
                           hidden_activations="relu", output_activation="softmax", dropout=0.9999,
                           initialiser="xavier")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2, batch_size=64)

    model.evaluate(x_test, y_test)

    results = model.evaluate(x_test, y_test)
    assert not results[1] > 0.9

    model = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 10]],
                           hidden_activations="relu", output_activation="softmax", dropout=0.0,
                           initialiser="xavier")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2, batch_size=64)

    model.evaluate(x_test, y_test)

    results = model.evaluate(x_test, y_test)
    assert results[1] > 0.9

def test_all_activations_work():
    """Tests that all activations get accepted"""
    nn_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                                   dropout=0.0000001,
                                   initialiser="xavier")
    for key in nn_instance.str_to_activations_converter.keys():
        assert CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                                   hidden_activations=key, output_activation=key, dropout=0.0000001,
                                   initialiser="xavier")

def test_all_initialisers_work():
    """Tests that all initialisers get accepted"""
    nn_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                                   dropout=0.0000001,
                                   initialiser="xavier")
    for key in nn_instance.str_to_initialiser_converter.keys():
        assert CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["linear", 1]],
                                    dropout=0.0000001,
                                   initialiser=key)

def test_print_model_summary():
    nn_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1]],
                      dropout=0.0000001, batch_norm=True,
                      initialiser="xavier")
    nn_instance.print_model_summary((64, 11, 11, 3))

def test_output_heads_error_catching():
    """Tests that having multiple output heads catches errors from user inputs"""
    output_dims_that_should_break = [["linear", 2, 2, "SAME", "conv", 3, 4, "SAME"], [[["conv", 3, 2, "same"], ["linear", 4]]],
                                     [[2, 8]], [-33, 33, 33, 33, 33]]
    for output_dim in output_dims_that_should_break:
        with pytest.raises(AssertionError):
            CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1], output_dim],
                hidden_activations="relu", output_activation="relu")
    output_activations_that_should_break = ["relu", ["relu"], ["relu", "softmax"]]
    for output_activation in output_activations_that_should_break:
        with pytest.raises(AssertionError):
            CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1],  [["linear", 4], ["linear", 10], ["linear", 4]]],
               hidden_activations="relu", output_activation=output_activation)


def test_output_head_layers():
    """Tests whether the output head layers get created properly"""
    for output_dim in [[["linear", 3],["linear", 9]], [["linear", 4], ["linear", 20]], [["linear", 1], ["linear", 1]]]:
        nn_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1], output_dim],
                          hidden_activations="relu", output_activation=["softmax", None])
        assert nn_instance.output_layers[0].units == output_dim[0][1]
        assert nn_instance.output_layers[1].units == output_dim[1][1]

def test_output_head_activations_work():
    """Tests that output head activations work properly"""

    output_dim = [["linear", 5], ["linear", 10], ["linear", 3]]
    nn_instance = CNN(layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1], output_dim],
                          hidden_activations="relu", output_activation=["softmax", None, "relu"])
    x = np.random.random((20, 10, 10, 4)) * -20.0
    out = nn_instance(x)

    assert out.shape == (20, 18)

    sums = tf.reduce_sum(out[:, :5], axis=1)
    sums_others = tf.reduce_sum(out[:, 5:], axis=1)
    sums_others_2 = tf.reduce_sum(out[:, 5:15], axis=1)
    sums_others_3 = tf.reduce_sum(out[:, 15:18], axis=1)


    for row in range(out.shape[0]):
        assert tf.math.equal(np.round(sums[row], 4), 1.0), sums[row]
        assert not tf.math.equal(np.round(sums_others[row], 4), 1.0), np.round(sums_others[row], 4)
        assert not tf.math.equal(np.round(sums_others_2[row], 4), 1.0), np.round(sums_others_2[row], 4)
        assert not tf.math.equal(np.round(sums_others_3[row], 4), 1.0), np.round(sums_others_3[row], 4)
        for col in range(3):
            assert out[row, 15 + col] >= 0.0, out[row, 15 + col]

def test_output_head_shapes_correct():
    """Tests that the output shape of network is correct when using multiple outpout heads"""
    N = 20
    X = np.random.random((N, 25, 25, 2))
    for _ in range(25):
        nn_instance = CNN(
            layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1], ["linear", 12]],
            hidden_activations="relu")
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == 12

    for output_dim in [[ ["linear", 10], ["linear", 4], ["linear", 6]], [["linear", 3], ["linear", 8], ["linear", 9]]]:
        nn_instance = CNN(
            layers_info=[["conv", 25, 5, 1, "valid"], ["conv", 25, 5, 1, "valid"], ["linear", 1], output_dim],
            hidden_activations="relu", output_activation=["softmax", None, "relu"])
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == 20

