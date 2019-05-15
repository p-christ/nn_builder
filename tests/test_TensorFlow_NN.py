# Run from home directory with python -m pytest tests
import pytest
import torch
import random
import numpy as np
import tensorflow as tf
from nn_builder.tensorflow.NN import NN
import tensorflow.keras.initializers as initializers
import tensorflow.keras.activations as activations

N = 250
X = np.random.random((N, 5))
X[:, [2, 4]] += 10.0
y = X[:, 0] > 0 * 1.0

def test_linear_hidden_units_user_input():
    """Tests whether network rejects an invalid linear_hidden_units input from user"""
    inputs_that_should_fail = ["a", ["a", "b"], [2, 4, "ss"], [-2], 2]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            NN(input_dim=2, layers_info=input_value, hidden_activations="relu", output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            NN(input_dim=2, layers_info=[2], hidden_activations=input_value,
               output_activation="relu")
            NN(input_dim=2, layers_info=[2], hidden_activations="relu",
               output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            NN(input_dim=2, layers_info=[2], hidden_activations="relu",
               output_activation="relu", initialiser=input_value)
        NN(input_dim=2, layers_info=[2], hidden_activations="relu",
           output_activation="relu", initialiser="xavier")

def test_output_shape_correct():
    """Tests whether network returns output of the right shape"""
    input_dims = [x for x in range(1, 3)]
    output_dims = [x for x in range(4, 6)]
    linear_hidden_units_options = [ [2, 3, 4], [2, 9, 1], [55, 55, 55, 234, 15]]
    for input_dim, output_dim, linear_hidden_units in zip(input_dims, output_dims, linear_hidden_units_options):
        linear_hidden_units.append(output_dim)
        nn_instance = NN(input_dim=input_dim, layers_info=linear_hidden_units, hidden_activations="relu",
                         output_activation="relu", initialiser="xavier")
        data = np.random.random((25, input_dim))
        output = nn_instance(data)
        assert output.shape == (25, output_dim)

def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = np.random.random((1, 100))
        nn_instance = NN(input_dim=100, layers_info=[5, 5, 5],
                         hidden_activations="relu",
                         output_activation="relu", initialiser="xavier")
        out = nn_instance(data)
        assert all(tf.squeeze(out) >= 0)

        nn_instance = NN(input_dim=100, layers_info=[5, 5, 5],
                         hidden_activations="relu",
                         output_activation="sigmoid", initialiser="xavier")
        out = nn_instance(data)
        assert all(tf.squeeze(out) >= 0)
        assert all(tf.squeeze(out) <= 1)

        nn_instance = NN(input_dim=100, layers_info=[5, 5, 5],
                         hidden_activations="relu",
                         output_activation="softmax", initialiser="xavier")
        out = nn_instance(data)
        assert all(tf.squeeze(out) >= 0)
        assert all(tf.squeeze(out) <= 1)
        assert np.round(tf.reduce_sum(tf.squeeze(out)), 3) == 1.0

        nn_instance = NN(input_dim=100, layers_info=[5, 5, 5],
                         hidden_activations="relu")

        out = nn_instance(data)
        assert not all(tf.squeeze(out) >= 0)
        assert not np.round(tf.reduce_sum(tf.squeeze(out)), 3) == 1.0

def test_linear_layers_info():
    """Tests whether create_hidden_layers_info method works correctly"""
    for input_dim, output_dim, hidden_units in zip( range(5, 8), range(9, 12), [[2, 9, 2], [3, 5, 6], [9, 12, 2]]):
        hidden_units.append(output_dim)
        nn_instance = NN(input_dim=input_dim, layers_info=hidden_units,
                         hidden_activations="relu",
                         output_activation="softmax", initialiser="xavier")
        assert len(nn_instance.hidden_layers) == len(hidden_units) - 1
        for layer_ix in range(len(hidden_units) - 1):
            layer = nn_instance.hidden_layers[layer_ix]
            print(nn_instance.hidden_layers[layer_ix])
            assert type(layer) == tf.keras.layers.Dense
            assert layer.units == hidden_units[layer_ix]
            assert layer.kernel_initializer == initializers.glorot_uniform
            assert layer.activation == activations.relu

        output_layer = nn_instance.output_layers[0]
        assert type(output_layer) == tf.keras.layers.Dense
        assert output_layer.units == hidden_units[-1]
        assert output_layer.kernel_initializer == initializers.glorot_uniform
        assert output_layer.activation == activations.softmax

def test_embedding_layers():
    """Tests whether create_embedding_layers_info method works correctly"""
    for embedding_in_dim_1, embedding_out_dim_1, embedding_in_dim_2, embedding_out_dim_2 in zip(range(5, 8), range(3, 6), range(1, 4), range(24, 27)):
        nn_instance = NN(input_dim=5, layers_info=[5], columns_of_data_to_be_embedded=[0, 1],
                         embedding_dimensions =[[embedding_in_dim_1, embedding_out_dim_1], [embedding_in_dim_2, embedding_out_dim_2]])
        for layer in nn_instance.embedding_layers:
            assert isinstance(layer, tf.keras.layers.Embedding)
        assert len(nn_instance.embedding_layers) == 2
        assert nn_instance.embedding_layers[0].input_dim == embedding_in_dim_1
        assert nn_instance.embedding_layers[0].output_dim == embedding_out_dim_1
        assert nn_instance.embedding_layers[1].input_dim == embedding_in_dim_2
        assert nn_instance.embedding_layers[1].output_dim == embedding_out_dim_2

def test_non_integer_embeddings_rejected():
    """Tests whether an error is raised if user tries to provide non-integer data to be embedded"""
    with pytest.raises(AssertionError):
        nn_instance = NN(input_dim=5, layers_info=[5],
                         columns_of_data_to_be_embedded=[2, 4],
                         embedding_dimensions=[[50, 3],
                                               [55, 4]])
        out = nn_instance(X)

def test_incorporate_embeddings():
    """Tests the method incorporate_embeddings"""
    X_new = X
    X_new[:, [2, 4]] = tf.round(X_new[:, [2, 4]])
    nn_instance = NN(input_dim=5, layers_info=[10],
                     columns_of_data_to_be_embedded=[2, 4],
                     embedding_dimensions=[[50, 3],
                                                       [55, 4]])
    out = nn_instance.incorporate_embeddings(X)
    assert out.shape == (N, X.shape[1]+3+4-2)

def test_embedding_network_can_solve_simple_problem():
    """Tests whether network can solve simple problem using embeddings"""
    X = np.random.random((N, 5)) * 5.0 + 20.0
    y = (X[:, 0] >= 20) * (X[:, 1] <= 20) * 1.0
    nn_instance = NN(input_dim=2, layers_info=[5, 1],
                     columns_of_data_to_be_embedded=[0, 1],
                     embedding_dimensions=[[50, 3],
                                           [55, 3]])
    assert solves_simple_problem(X, y, nn_instance)


def test_batch_norm_layers_info():
    """Tests whether batch_norm_layers_info method works correctly"""
    for input_dim, output_dim, hidden_units in zip( range(5, 8), range(9, 12), [[2, 9, 2], [3, 5, 6], [9, 12, 2]]):
        hidden_units.append(output_dim)
        nn_instance = NN(input_dim=input_dim, layers_info=hidden_units,
                         hidden_activations="relu", batch_norm=True,
                         output_activation="relu", initialiser="xavier", print_model_reduce_summary=False)
        for layer in nn_instance.batch_norm_layers_info:
            assert isinstance(layer, nn.BatchNorm1d)
        assert len(nn_instance.batch_norm_layers_info) == len(hidden_units) - 1
        assert nn_instance.batch_norm_layers_info[0].num_features == hidden_units[0]
        assert nn_instance.batch_norm_layers_info[1].num_features == hidden_units[1]
        assert nn_instance.batch_norm_layers_info[2].num_features == hidden_units[2]

def test_model_trains():
    """Tests whether a small range of networks can solve a simple task"""
    for output_activation in ["sigmoid", "None"]:
        nn_instance = NN(input_dim=X.shape[1], layers_info=[10, 10, 10, 1],
                         output_activation=output_activation, dropout=0.01, batch_norm=True)
        assert solves_simple_problem(X, y, nn_instance)
    z = X[:, 0:1] > 0
    z =  torch.cat([z ==1, z==0], dim=1).float()
    nn_instance = NN(input_dim=X.shape[1], layers_info=[10, 10, 10, 2],
                     output_activation="softmax", dropout=0.01, batch_norm=True)
    assert solves_simple_problem(X, z, nn_instance)


def solves_simple_problem(X, y, nn_instance):
    """Checks if a given network is able to solve a simple problem"""
    nn_instance.compile(optimizer='adam',
                  loss='mse')

    nn_instance.fit(X, y, epochs=800)

    results = nn_instance.evaluate(X, y)
    return results < 0.1



def test_dropout():
    """Tests whether dropout layer reads in probability correctly"""
    nn_instance = NN(input_dim=X.shape[1], layers_info=[10, 10, 1], dropout=0.9999)
    assert nn_instance.dropout_layer.p == 0.9999
    assert not solves_simple_problem(X, y, nn_instance)
    nn_instance = NN(input_dim=X.shape[1], layers_info=[10, 10, 1], dropout=0.00001)
    assert solves_simple_problem(X, y, nn_instance)

def test_y_range_user_input():
    """Tests whether network rejects invalid y_range inputs"""
    invalid_y_range_inputs = [ (4, 1), (2, 4, 8), [2, 4], (np.array(2.0), 6.9)]
    for y_range_value in invalid_y_range_inputs:
        with pytest.raises(AssertionError):
            print(y_range_value)
            nn_instance = NN(input_dim=5, layers_info=[10, 10, 3],
                             y_range=y_range_value)

def test_y_range():
    """Tests whether setting a y range works correctly"""
    for _ in range(100):
        val1 = random.random() - 3.0*random.random()
        val2 = random.random() + 2.0*random.random()
        lower_bound = min(val1, val2)
        upper_bound = max(val1, val2)
        nn_instance = NN(input_dim=5, layers_info=[10, 10, 3],  y_range=(lower_bound, upper_bound))
        random_data = torch.randn(15, 5)
        out = nn_instance.forward(random_data)
        assert torch.reduce_sum(out > lower_bound).item() == 3*15, "lower {} vs. {} ".format(lower_bound, out)
        assert torch.reduce_sum(out < upper_bound).item() == 3*15, "upper {} vs. {} ".format(upper_bound, out)

def test_deals_with_None_activation():
    """Tests whether is able to handle user inputting None as output activation"""
    assert NN(input_dim=5, layers_info=[10, 10, 3], output_activation=None)

def test_check_input_data_into_forward_once():
    """Tests that check_input_data_into_forward_once method only runs once"""
    data_to_throw_error = torch.randn(N, 2)
    X = torch.randn(N, 2) * 5.0 + 20.0
    y = (X[:, 0] >= 20) * (X[:, 1] <= 20)
    X = X.long()
    nn_instance = NN(input_dim=2, layers_info=[5, 1],
                     columns_of_data_to_be_embedded=[0, 1],
                     embedding_dimensions=[[50, 3],
                                           [55, 3]])
    with pytest.raises(AssertionError):
        nn_instance.forward(data_to_throw_error)
    with pytest.raises(RuntimeError):
        nn_instance.forward(X)
        nn_instance.forward(data_to_throw_error)


# Add test that each layer has right initialiser ... and activation...
