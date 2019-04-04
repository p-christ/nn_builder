# Neural Network Builder (nn_builder)

`pip install nn_builder`

nn_builder creates neural networks in 1 line. 

It makes it easier for you to create neural networks and quickly experiment
with different architectures. 

On the left is the code you would need to write to create a PyTorch neural network equivalent to the one line of code on the right using nn_builder. 

![Screenshot](miscellaneous/material_for_readme/simple_use_case.png)

For more complicated networks the difference becomes even larger:

![Screenshot](miscellaneous/material_for_readme/more_complicated_use_case.png)


## Usage

### 1. PyTorch NN Module

To import the PyTorch NN module run: 

`from nn_builder.pytorch.NN import NN`

NN then takes the following fields:
`
1. ***input_dim***: integer to indicate the dimension of the input into the network
1. ***linear_hidden_units***: list of integers to indicate the width and number of linear hidden layers you want in your network
1. ***output_dim***: integer to indicate the dimension of the output of the network
1. ***output_activation***: string to indicate the activation function you want the output to go through. Default is no activation 
and then for example "softmax" puts it through a softmax activation, and "sigmoid" through a sigmoid activation.
1. ***hidden_activations***: string or list of string to indicate the activations you want used on the output of hidden layers (not including the output layer),
default is ReLU and for example "tanh" would have tanh applied on all hidden layer activations
1. ***dropout***: float to indicate what dropout probability you want applied after each hidden layer. Default = 0.
1. ***initialiser***: string to indicate which initialiser you want used to initialise all the parameters. All PyTorch initialisers are supported. 
Default = the PyTorch default initialisation
1. ***batch_norm***: boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default = False
1. ***embedding_dimensions***: list to indicate the activation function you want the output to go through. Default is no activation 
and then for example "softmax" puts it through a softmax activation, and "sigmoid" through a sigmoid activation.
1. ***y_range***: tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
 output values to in regression tasks. Leave the default for no restriction. 

1. ***print_model_summary***: boolean to indicate whether you want a model summary printed after model is created
`

Notebook to play around with: https://colab.research.google.com/drive/1abxTEaUrJqbTuk8e8tOa3y9DYQQVrF_N

| Field | Description | Default |
| :---: | :---: | :---: |
| input_dim | Integer to indicate the dimension of the input into the network | N/A |
| linear_hidden_units | List of integers to indicate the width and number of linear hidden layers you want in your network | N/A |
| output_dim | Integer to indicate the dimension of the output of the network | N/A |
| output_activation | String to indicate the activation function you want the output to go through | No activation |
| hidden_activations | String or list of string to indicate the activations you want used on the output of hidden layers (not including the output layer),
default is ReLU and for example "tanh" would have tanh applied on all hidden layer activations | ReLU after every hidden layer |
| dropout | Float to indicate what dropout probability you want applied after each hidden layer | 0 |
| initialiser | String to indicate which initialiser you want used to initialise all the parameters. All PyTorch initialisers are supported. | PyTorch Default |
| batch_norm | Boolean to indicate whether you want batch norm applied to the output of every hidden layer | False |
| embedding_dimensions | write me... | No embeddings |
| y_range | Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the output values to in regression tasks | No range |
| print_model_summary | Boolean to indicate whether you want a model summary printed after model is created | False |