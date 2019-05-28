# Run from home directory with python -m pytest tests
import pytest
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization, GRU, LSTM
from nn_builder.tensorflow.RNN import RNN

N = 250
X = np.random.random((N, 3, 5))
X = X.astype('float32')

X[0:125, :, 3] += 10.0
y = X[:, 2, 3] > 5.0

def test_user_hidden_layers_input_rejections():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_fail = [[["linearr", 33]], [["linear", 12, 33]], [["gru", 2, 33]], [["lstm", 2, 33]], [["lstmr", 33]],
                               [["gruu", 33]], [["gru", 33], ["xxx", 33]], [["linear", 33], ["gru", 12], ["gru", 33]] ]
    for input in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(layers_info=input, hidden_activations="relu",
                output_activation="relu")

def test_user_hidden_layers_input_acceptances():
    """Tests whether network rejects invalid hidden_layers inputted from user"""
    inputs_that_should_work = [[["linear", 33]], [["linear", 12]], [["gru", 2]], [["lstm", 2]], [["lstm", 1]],
                               [["gru", 330]], [["gru", 33], ["linear", 2]] ]
    for input in inputs_that_should_work:
        assert  RNN(layers_info=input, hidden_activations="relu",
                output_activation="relu")


def test_hidden_layers_created_correctly():
    """Tests that create_hidden_layers works correctly"""
    layers = [["gru", 25], ["lstm", 23], ["linear", 5], ["linear", 10]]

    rnn = RNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu")

    assert type(rnn.hidden_layers[0]) == GRU
    assert rnn.hidden_layers[0].units == 25

    assert type(rnn.hidden_layers[1]) == LSTM
    assert rnn.hidden_layers[1].units == 23

    assert type(rnn.hidden_layers[2]) == Dense
    assert rnn.hidden_layers[2].units == 5

    assert type(rnn.output_layers[0]) == Dense
    assert rnn.output_layers[0].units == 10


def test_output_layers_created_correctly():
    """Tests that create_output_layers works correctly"""
    layers = [["gru", 25], ["lstm", 23], ["linear", 5], ["linear", 10]]

    rnn = RNN(layers_info=layers, hidden_activations="relu", output_activation="relu")
    assert rnn.output_layers[0].units == 10

    layers = [["gru", 25], ["lstm", 23], ["lstm", 10]]
    rnn = RNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu")
    assert rnn.output_layers[0].units == 10

    layers = [["gru", 25], ["lstm", 23], [["lstm", 10], ["linear", 15]]]
    rnn = RNN(layers_info=layers, hidden_activations="relu",
              output_activation=["relu", "softmax"])
    assert rnn.output_layers[0].units == 10
    assert rnn.output_layers[1].units == 15

def test_output_dim_user_input():
    """Tests whether network rejects an invalid output_dim input from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(layers_info=[2, input_value], hidden_activations="relu",  output_activation="relu")
        with pytest.raises(AssertionError):
            RNN(layers_info=input_value, hidden_activations="relu", output_activation="relu")

def test_activations_user_input():
    """Tests whether network rejects an invalid hidden_activations or output_activation from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(layers_info=[["linear", 2]], hidden_activations=input_value,
                output_activation="relu")
            RNN(layers_info=[["linear", 2]], hidden_activations="relu",
                output_activation=input_value)

def test_initialiser_user_input():
    """Tests whether network rejects an invalid initialiser from user"""
    inputs_that_should_fail = [-1, "aa", ["dd"], [2], 0, 2.5, {2}, "Xavier_"]
    for input_value in inputs_that_should_fail:
        with pytest.raises(AssertionError):
            RNN(layers_info=[["linear", 2]], hidden_activations="relu",
                output_activation="relu", initialiser=input_value)

            RNN(layers_info=[["linear", 2], ["linear", 2]], hidden_activations="relu",
            output_activation="relu", initialiser="xavier")

def test_batch_norm_layers():
    """Tests whether batch_norm_layers method works correctly"""
    layers = [["gru", 20], ["lstm", 3], ["linear", 4], ["linear", 10]]
    rnn = RNN(layers_info=layers, hidden_activations="relu",
              output_activation="relu", initialiser="xavier", batch_norm=True)
    assert len(rnn.batch_norm_layers) == 3
    for layer in rnn.batch_norm_layers:
        assert isinstance(layer, BatchNormalization)

def test_linear_layers_only_come_at_end():
    """Tests that it throws an error if user tries to provide list of hidden layers that include linear layers where they
    don't only come at the end"""
    layers = [["gru", 20],  ["linear", 4], ["lstm", 3], ["linear", 10]]
    with pytest.raises(AssertionError):
        rnn = RNN(layers_info=layers, hidden_activations="relu",
                  output_activation="relu", initialiser="xavier", batch_norm=True)

    layers = [["gru", 20], ["lstm", 3],  ["linear", 4], ["linear", 10]]
    assert RNN(layers_info=layers, hidden_activations="relu",
                      output_activation="relu", initialiser="xavier", batch_norm=True)

def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = np.random.random((25, 10, 30))
        data = data.astype('float32')
        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier", batch_norm=True)
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="sigmoid", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)

        summed_result = tf.reduce_sum(out, axis=1)
        summed_result = tf.reshape(summed_result, [-1, 1])
        assert summed_result != 1.0

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="softmax", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["lstm", 25]],
                           hidden_activations="relu",
                           output_activation="softmax", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[["linear", 20], ["linear", 50]],
                           hidden_activations="relu")

        out = RNN_instance(data)
        assert not all(tf.reshape(out, [-1]) >= 0)
        assert not all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert not (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[ ["lstm", 25], ["linear", 10]],
                           hidden_activations="relu")

        out = RNN_instance(data)
        assert not all(tf.reshape(out, [-1]) >= 0)
        assert not all(tf.reshape(out, [-1]) <= 0)
        summed_result = tf.reduce_sum(out, axis=1)
        assert not (np.round(summed_result, 3) == 1.0).all()


def test_output_activation():
    """Tests whether network outputs data that has gone through correct activation function"""
    RANDOM_ITERATIONS = 20
    for _ in range(RANDOM_ITERATIONS):
        data = np.random.random((25, 10, 30))
        data = data.astype('float32')
        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier", batch_norm=True)
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="relu", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="sigmoid", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)

        summed_result = tf.reduce_sum(out, axis=1)
        summed_result = tf.reshape(summed_result, [-1, 1])
        assert summed_result != 1.0

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["linear", 10], ["linear", 3]],
                           hidden_activations="relu",
                           output_activation="softmax", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[["lstm", 20], ["gru", 5], ["lstm", 25]],
                           hidden_activations="relu",
                           output_activation="softmax", initialiser="xavier")
        out = RNN_instance(data)
        assert all(tf.reshape(out, [-1]) >= 0)
        assert all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[["linear", 20], ["linear", 50]],
                           hidden_activations="relu")

        out = RNN_instance(data)
        assert not all(tf.reshape(out, [-1]) >= 0)
        assert not all(tf.reshape(out, [-1]) <= 1)
        summed_result = tf.reduce_sum(out, axis=1)
        assert not (np.round(summed_result, 3) == 1.0).all()

        RNN_instance = RNN(layers_info=[ ["lstm", 25], ["linear", 10]],
                           hidden_activations="relu")

        out = RNN_instance(data)
        assert not all(tf.reshape(out, [-1]) >= 0)
        assert not all(tf.reshape(out, [-1]) <= 0)
        summed_result = tf.reduce_sum(out, axis=1)
        assert not (np.round(summed_result, 3) == 1.0).all()

def test_y_range():
    """Tests whether setting a y range works correctly"""
    for _ in range(20):
        val1 = random.random() - 3.0*random.random()
        val2 = random.random() + 2.0*random.random()
        lower_bound = min(val1, val2)
        upper_bound = max(val1, val2)
        rnn = RNN(layers_info=[["lstm", 20], ["gru", 5], ["lstm", 25]],
                  hidden_activations="relu", y_range=(lower_bound, upper_bound), initialiser="xavier")
        random_data = np.random.random((10, 11, 22))
        random_data = random_data.astype('float32')
        out = rnn(random_data)
        assert all(tf.reshape(out, [-1]) > lower_bound)
        assert all(tf.reshape(out, [-1]) < upper_bound)

def test_deals_with_None_activation():
    """Tests whether is able to handle user inputting None as output activation"""
    assert RNN(layers_info=[["lstm", 20], ["gru", 5], ["lstm", 25]],
                           hidden_activations="relu", output_activation=None,
                           initialiser="xavier")

def test_y_range_user_input():
    """Tests whether network rejects invalid y_range inputs"""
    invalid_y_range_inputs = [ (4, 1), (2, 4, 8), [2, 4], (np.array(2.0), 6.9)]
    for y_range_value in invalid_y_range_inputs:
        with pytest.raises(AssertionError):
            print(y_range_value)
            rnn = RNN(layers_info=[["lstm", 20], ["gru", 5], ["lstm", 25]],
                           hidden_activations="relu", y_range=y_range_value,
                           initialiser="xavier")


def solves_simple_problem(X, y, nn_instance):
    """Checks if a given network is able to solve a simple problem"""
    print("X shape ", X.shape)
    print("y shape ", y.shape)
    nn_instance.compile(optimizer='adam',
                  loss='mse')
    nn_instance.fit(X, y, epochs=25)
    results = nn_instance.evaluate(X, y)
    print("FINAL RESULT ", results)
    return results < 0.1

def test_model_trains():
    """Tests whether a small range of networks can solve a simple task"""
    for output_activation in ["sigmoid", "None"]:
        rnn = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 1]],
                           hidden_activations="relu", output_activation=output_activation,
                           initialiser="xavier")

        assert solves_simple_problem(X, y, rnn)

def test_model_trains_part_2():
    """Tests whether a small range of networks can solve a simple task"""
    z = X[:, 2:3, 3:4] > 5.0
    z = np.concatenate([z == 1, z == 0], axis=1)
    z = z.reshape((-1, 2))

    rnn = RNN(layers_info=[["gru", 20], ["lstm", 2]],
                           hidden_activations="relu", output_activation="softmax", dropout=0.01,
                           initialiser="xavier")
    assert solves_simple_problem(X, z, rnn)

    rnn = RNN(layers_info=[["lstm", 20], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, rnn)

    rnn = RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier")
    assert solves_simple_problem(X, y, rnn)

def test_model_trains_with_batch_norm():
    """Tests whether a model with batch norm on can solve a simple task"""
    rnn = RNN(layers_info=[["lstm", 20], ["linear", 20], ["linear", 1]],
                       hidden_activations="relu", output_activation=None,
                       initialiser="xavier", batch_norm=True)
    assert solves_simple_problem(X, y, rnn)

def test_dropout():
    """Tests whether dropout layer reads in probability correctly"""
    rnn = RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           hidden_activations="relu", output_activation="sigmoid", dropout=0.9999,
                           initialiser="xavier")
    assert rnn.dropout_layer.rate == 0.9999
    assert not solves_simple_problem(X, y, rnn)
    rnn = RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           hidden_activations="relu", output_activation=None, dropout=0.0000001,
                           initialiser="xavier")
    assert rnn.dropout_layer.rate == 0.0000001
    assert solves_simple_problem(X, y, rnn)


def test_all_activations_work():
    """Tests that all activations get accepted"""
    nn_instance = RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           hidden_activations="relu", output_activation=None, dropout=0.0000001,
                           initialiser="xavier")
    for key in nn_instance.str_to_activations_converter.keys():
        assert RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           hidden_activations=key, output_activation=key, dropout=0.0000001,
                           initialiser="xavier")

def test_all_initialisers_work():
    """Tests that all initialisers get accepted"""
    nn_instance = RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           hidden_activations="relu", output_activation=None, dropout=0.0000001,
                           initialiser="xavier")
    for key in nn_instance.str_to_initialiser_converter.keys():
        assert RNN(layers_info=[["lstm", 20], ["gru", 10], ["linear", 20], ["linear", 1]],
                           dropout=0.0000001,
                           initialiser=key)

def test_output_shapes():
    """Tests whether network outputs of correct shape"""
    rnn = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 3]],
              hidden_activations="relu", initialiser="xavier")
    output = rnn(X)
    assert output.shape == (N, 3)

    rnn = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
              hidden_activations="relu", initialiser="xavier", return_final_seq_only=False)
    output = rnn(X)
    assert output.shape == (N, 3, 7)

    rnn = RNN(layers_info=[["gru", 20], ["lstm", 8], ["lstm", 3]],
              hidden_activations="relu", initialiser="xavier")
    output = rnn(X)
    assert output.shape == (N, 3)

    rnn = RNN(layers_info=[["gru", 20], ["lstm", 8], ["lstm", 7]],
              hidden_activations="relu", initialiser="xavier", return_final_seq_only=False)
    output = rnn(X)
    assert output.shape == (N, 3, 7)

def test_return_final_seq_user_input_valid():
    """Checks whether network only accepts a valid boolean value for return_final_seq_only"""
    for valid_case in [True, False]:
        assert RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                  hidden_activations="relu", initialiser="xavier", return_final_seq_only=valid_case)

    for invalid_case in [[True], 22, [1, 3], (True, False), (5, False)]:
        with pytest.raises(AssertionError):
            print(invalid_case)
            RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                hidden_activations="relu", initialiser="xavier", return_final_seq_only=invalid_case)

def test_embedding_layers():
    """Tests whether create_embedding_layers_info method works correctly"""
    for embedding_in_dim_1, embedding_out_dim_1, embedding_in_dim_2, embedding_out_dim_2 in zip(range(5, 8), range(3, 6), range(1, 4), range(24, 27)):
        nn_instance = RNN( layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]], columns_of_data_to_be_embedded=[0, 1],
                         embedding_dimensions =[[embedding_in_dim_1, embedding_out_dim_1], [embedding_in_dim_2, embedding_out_dim_2]])
        for layer in nn_instance.embedding_layers:
            assert isinstance(layer, tf.keras.layers.Embedding)
        assert len(nn_instance.embedding_layers) == 2
        assert nn_instance.embedding_layers[0].input_dim == embedding_in_dim_1
        assert nn_instance.embedding_layers[0].output_dim == embedding_out_dim_1
        assert nn_instance.embedding_layers[1].input_dim == embedding_in_dim_2
        assert nn_instance.embedding_layers[1].output_dim == embedding_out_dim_2

def test_incorporate_embeddings():
    """Tests the method incorporate_embeddings"""
    X_new = X
    X_new[:, [0, 2], :] = tf.round(X_new[:, [0, 2], :])
    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                     columns_of_data_to_be_embedded=[0, 2],
                     embedding_dimensions=[[50, 3], [55, 4]])
    out = nn_instance.incorporate_embeddings(X)
    assert out.shape == (N, 3, 10)

    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                     columns_of_data_to_be_embedded=[0, 1, 2],
                     embedding_dimensions=[[50, 3], [55, 4], [55, 4]])
    out = nn_instance.incorporate_embeddings(X)
    assert out.shape == (N, 3, 13)

    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                     columns_of_data_to_be_embedded=[2],
                     embedding_dimensions=[[150, 30]])
    out = nn_instance.incorporate_embeddings(X)
    assert out.shape == (N, 3, 34)

def test_embedding_network_can_solve_simple_problem():
    """Tests whether network can solve simple problem using embeddings"""
    X = (np.random.random((N, 4, 5)) - 0.5) * 5.0 + 20.0
    y = (X[:, :, 0] >= 25) * (X[:, :, 1] <= 25) * 1.0
    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 1]],
                     columns_of_data_to_be_embedded=[0, 1],
                     embedding_dimensions=[[50, 3],
                                           [55, 3]])
    assert solves_simple_problem(X, y, nn_instance)


def test_print_model_summary():
    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], ["linear", 7]],
                     columns_of_data_to_be_embedded=[2],
                     embedding_dimensions=[[150, 30]])
    nn_instance.print_model_summary((64, 11, 11))

def test_output_heads_error_catching():
    """Tests that having multiple output heads catches errors from user inputs"""
    output_dims_that_should_break = [["linear", 2, 2, "SAME", "conv", 3, 4, "SAME"], [[["lstm", 3], ["gru", 4]]],
                                     [[2, 8]], [-33, 33, 33, 33, 33]]
    for output_dim in output_dims_that_should_break:
        with pytest.raises(AssertionError):
            RNN(layers_info=[["gru", 20], ["lstm", 8], output_dim],
                hidden_activations="relu", output_activation="relu")
    output_activations_that_should_break = ["relu", ["relu"], ["relu", "softmax"]]
    for output_activation in output_activations_that_should_break:
        with pytest.raises(AssertionError):
            RNN(layers_info=[["gru", 20], ["lstm", 8], [["linear", 5], ["linear", 2], ["linear", 5]]],
               hidden_activations="relu", output_activation=output_activation)


def test_output_head_layers():
    """Tests whether the output head layers get created properly"""
    for output_dim in [[["linear", 3],["linear", 9]], [["linear", 4], ["linear", 20]], [["linear", 1], ["linear", 1]]]:
        nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], output_dim],
                          hidden_activations="relu", output_activation=["softmax", None])
        assert nn_instance.output_layers[0].units == output_dim[0][1]
        assert nn_instance.output_layers[1].units == output_dim[1][1]

def test_output_head_activations_work():
    """Tests that output head activations work properly"""

    output_dim = [["linear", 5], ["linear", 10], ["linear", 3]]
    nn_instance = RNN(layers_info=[["gru", 20], ["lstm", 8], output_dim],
                          hidden_activations="relu", output_activation=["softmax", None, "relu"])
    x = np.random.random((20, 10, 4)) * -20.0
    x = x.astype('float32')
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
    X = np.random.random((N, 10, 4)) * -20.0
    X = X.astype('float32')
    for _ in range(25):
        nn_instance = RNN(
            layers_info=[["gru", 20], ["lstm", 8], ["linear", 1], ["linear", 12]],
            hidden_activations="relu")
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == 12

    for output_dim in [[ ["linear", 10], ["linear", 4], ["linear", 6]], [["linear", 3], ["linear", 8], ["linear", 9]]]:
        nn_instance = RNN(
            layers_info=[["gru", 20], ["lstm", 8], ["linear", 1], ["linear", 12], output_dim],
            hidden_activations="relu", output_activation=["softmax", None, "relu"])
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == 20
