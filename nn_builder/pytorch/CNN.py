import torch
import torch.nn as nn
from nn_builder.pytorch.Base_Network import Base_Network

class CNN(nn.Module, Base_Network):
    """Creates a PyTorch convolutional neural network
    Args:
        - input_dim: Integer to indicate the number of input channels into the network. e.g. if image is RGB then this should be 3
        - hidden_layers_info: List of layer specifications to specify the hidden layers of the network. Each element of the list must be
                         one of these 6 forms:
                         - ["conv", channels, kernel_size, stride, padding]
                         - ["maxpool", kernel_size, stride, padding]
                         - ["avgpool", kernel_size, stride, padding]
                         - ["adaptivemaxpool", output height, output width]
                         - ["adaptiveavgpool", output height, output width]
                         - ["linear", in, out]
        - output_layer_input_dim: Integer to indicate dimension of input into output layer. Only needed if the final hidden layer you
                                  provide is not adaptivemaxpool, adaptiveavgpool or linear
        - output_dim: Integer to indicate the dimension of the output of the network if you want 1 output head. Provide a list of integers
                      if you want multiple output heads
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                              (not including the output layer). Default is ReLU.
        - dropout: Float to indicate what dropout probability you want applied after each hidden layer
        - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                       initialisers are supported. PyTorch's default initialisation is the default.
        - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
        - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                   output values to in regression tasks. Default is no range restriction
        - print_model_summary: Boolean to indicate whether you want a model summary printed after model is created. Default is False.
    """

    def __init__(self, hidden_layers_info, output_dim, output_layer_input_dim=None, input_dim=1, output_activation=None, hidden_activations="relu",
                 dropout: float = 0.0, initialiser: str = "default", batch_norm: bool = False, y_range: tuple = (),
                 random_seed=0, print_model_summary: bool =False):
        nn.Module.__init__(self)
        self.valid_cnn_hidden_layer_types = {'conv', 'maxpool', 'avgpool', 'adaptivemaxpool', 'adaptiveavgpool', 'linear'}
        self.valid_layer_types_with_no_parameters = [nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d]
        self.output_layer_input_dim = output_layer_input_dim
        Base_Network.__init__(self, input_dim, hidden_layers_info, output_dim, output_activation,
                              hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed,
                              print_model_summary)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_input_dim_valid()
        self.check_output_dim_valid()
        self.check_cnn_hidden_layers_valid()
        self.check_activations_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def create_hidden_layers(self):
        """Creates the linear layers in the network"""
        cnn_hidden_layers = nn.ModuleList([])
        in_channels = self.input_dim
        for layer in self.hidden_layers_info:
            layer_name = layer[0].lower()
            assert layer_name in self.valid_cnn_hidden_layer_types, "Layer name {} not valid, use one of {}".format(layer_name, self.valid_cnn_hidden_layer_types)
            if layer_name == "conv":
                cnn_hidden_layers.extend([nn.Conv2d(in_channels=in_channels, out_channels=layer[1], kernel_size=layer[2],
                                             stride=layer[3], padding=layer[4])])
                in_channels = layer[1]
            elif layer_name == "maxpool":
                cnn_hidden_layers.extend([nn.MaxPool2d(kernel_size=layer[1],
                                             stride=layer[2], padding=layer[3])])
            elif layer_name == "avgpool":
                cnn_hidden_layers.extend([nn.AvgPool2d(kernel_size=layer[1],
                                             stride=layer[2], padding=layer[3])])
            elif layer_name == "adaptivemaxpool":
                cnn_hidden_layers.extend([nn.AdaptiveMaxPool2d(output_size=(layer[1], layer[2]))])
            elif layer_name == "adaptiveavgpool":
                cnn_hidden_layers.extend([nn.AdaptiveAvgPool2d(output_size=(layer[1], layer[2]))])
            elif layer_name == "linear":
                cnn_hidden_layers.extend([nn.Linear(in_features=layer[1], out_features=layer[2])])
            else:
                raise ValueError("Wrong layer name")
        return cnn_hidden_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        if self.output_layer_input_dim is not None:
            input_dim = self.output_layer_input_dim
        elif self.hidden_layers_info[-1][0].lower() in ["adaptivemaxpool", "adaptiveavgpool"]:
            input_dim = self.hidden_layers[-1].output_size[0] * self.hidden_layers[-1].output_size[1]
            for layer_info_ix in range(len(self.hidden_layers_info)):
                layer_info = self.hidden_layers_info[-(1+layer_info_ix)]
                if layer_info[0].lower() == "conv":
                    input_dim = input_dim * layer_info[1]
                    break
        elif self.hidden_layers_info[-1][0].lower() == "linear":
            input_dim = self.hidden_layers[-1].out_features
        else:
            raise ValueError("Don't know dimensions for output layer. Must use adaptivemaxpool, adaptiveavgpool, or linear as final hidden layer")
        if not isinstance(self.output_dim, list): self.output_dim = [self.output_dim]
        for output_dim in self.output_dim:
            output_layers.extend([nn.Linear(input_dim, output_dim)])
        return output_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        initialisable_layers = [layer for layer in self.hidden_layers if not type(layer) in self.valid_layer_types_with_no_parameters]
        self.initialise_parameters(initialisable_layers)
        self.initialise_parameters(self.output_layers)

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([])
        for layer in self.hidden_layers_info:
            layer_type = layer[0].lower()
            if layer_type == "conv":
                batch_norm_layers.extend([nn.BatchNorm2d(num_features=layer[1])])
            elif layer_type == "linear":
                batch_norm_layers.extend([nn.BatchNorm2d(num_features=layer[2])])
        return batch_norm_layers

    def forward(self, x):
        """Forward pass for the network"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x)
        flattened=False

        for layer_ix, layer in enumerate(self.hidden_layers):
            if type(layer) in self.valid_layer_types_with_no_parameters:
                x = layer(x)
            else:
                if type(layer) == nn.Linear and not flattened:
                    x = self.flatten_tensor(x)
                    flattened = True
                x = self.get_activation(self.hidden_activations, layer_ix)(layer(x))
                if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
                x = self.dropout_layer(x)

        if not flattened: x = self.flatten_tensor(x)
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            temp_output = output_layer(x)
            if activation is not None: temp_output = activation(temp_output)
            if out is None: out = temp_output
            else: out = torch.cat((out, temp_output), dim=1)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(out)
        return out

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        assert len(x.shape) == 4, "x should have the shape (batch_size, channel, height, width)"
        assert x.shape[1] == self.input_dim
        self.checked_forward_input_data_once = True #So that it doesn't check again

