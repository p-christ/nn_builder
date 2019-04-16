import random

import torch
import pytest
from nn_builder.pytorch.NN import NN

def test_output_heads_error_catching():
    """Tests that having multiple output heads catches errors from user inputs"""
    output_dims_that_should_break = [[2, 8], [33, 33, 33, 33, 33]]
    for output_dim in output_dims_that_should_break:
        with pytest.raises(AssertionError):
            NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=output_dim,
                           output_activation="relu")

    output_activations_that_should_break = ["relu", ["relu"], ["relu", "softmax"]]
    for output_activation in output_activations_that_should_break:
        with pytest.raises(AssertionError):
            NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=[4, 6, 1],
               output_activation=output_activation)

def test_output_head_layers():
    """Tests whether the output head layers get created properly"""
    for output_dim in [[3, 9], [4, 20], [1, 1]]:
        nn_instance = NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=output_dim,
           output_activation=["softmax", None])
        assert nn_instance.output_layers[0].in_features == 9
        assert nn_instance.output_layers[1].in_features == 9
        assert nn_instance.output_layers[0].out_features == output_dim[0]
        assert nn_instance.output_layers[1].out_features == output_dim[1]

def test_output_head_activations_work():
    """Tests that output head activations work properly"""

    nn_instance = NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=[5, 10, 3],
                     output_activation=["softmax", None, "relu"])

    x = torch.randn((20, 2)) * -20.0
    out = nn_instance.forward(x)
    assert torch.allclose(torch.sum(out[:, :5], dim=1), torch.Tensor([1.0]))
    assert not torch.allclose(torch.sum(out[:, 5:], dim=1), torch.Tensor([1.0]))
    for row in range(out.shape[0]):
        assert all(out[row, -3:] >= 0)

def test_output_head_shapes_correct():
    """Tests that the output shape of network is correct when using multiple outpout heads"""
    N = 20
    X = torch.randn((N, 2))
    for _ in range(25):
        output_dim = random.randint(1, 100)
        nn_instance = NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=output_dim)
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == output_dim

    for output_dim in [[3, 9, 5, 3], [5, 5, 5, 5], [2, 1, 1, 16]]:
        nn_instance = NN(input_dim=2, linear_hidden_units=[4, 7, 9], hidden_activations="relu", output_dim=output_dim,
                         output_activation=["softmax", None, None, "relu"])
        out = nn_instance(X)
        assert out.shape[0] == N
        assert out.shape[1] == 20




