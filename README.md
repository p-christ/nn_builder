# Neural Network Builder (nn_builder)

![Image](https://travis-ci.org/p-christ/nn_builder.svg?branch=master)

`pip install nn_builder`

**nn_builder creates neural networks in 1 line**. It makes it easier for you to create neural networks and quickly experiment
with different architectures. 

On the left is the large amount of code you would need to write to create a PyTorch neural network equivalent to the only one line of code on the right using nn_builder: 

![Screenshot](miscellaneous/material_for_readme/simple_use_case.png)

For more complicated networks the difference is even greater:

![Screenshot](miscellaneous/material_for_readme/more_complicated_use_case.png)


## Usage

### 1. PyTorch NN Module

The PyTorch NN module lets you build feedforward (rather than CNNs or RNNs) PyTorch neural networks in one line. 

First run `from nn_builder.pytorch.NN import NN` and then NN takes the below arguments:

| Field | Description | Default |
| :---: | :----------: | :---: |
| *input_dim*| Integer to indicate the dimension of the input into the network | N/A |
| *linear_hidden_units* | List of integers to indicate the width and number of linear hidden layers you want in your network | N/A |
| *output_dim* | Integer to indicate the dimension of the output of the network | N/A |
| *output_activation* | String to indicate the activation function you want the output to go through | No activation |
| *hidden_activations* | String or list of string to indicate the activations you want used on the output of hidden layers (not including the output layer), default is ReLU and for example "tanh" would have tanh applied on all hidden layer activations | ReLU after every hidden layer |
| *dropout* | Float to indicate what dropout probability you want applied after each hidden layer | 0 |
| *initialiser* | String to indicate which initialiser you want used to initialise all the parameters. All PyTorch initialisers are supported. | PyTorch Default |
| *batch_norm* | Boolean to indicate whether you want batch norm applied to the output of every hidden layer | False |
| *embedding_dimensions* | If you have categorical variables you want embedded before flowing through the network then you specify the embedding dimensions here with a list like so: [ [embedding_input_dim_1, embedding_output_dim_1], [embedding_input_dim_2, embedding_output_dim_2] ...] | No embeddings |
| *y_range* | Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the output values to in regression tasks | No range |
| *print_model_summary* | Boolean to indicate whether you want a model summary printed after model is created | False |

See this [colab notebook](https://colab.research.google.com/drive/1abxTEaUrJqbTuk8e8tOa3y9DYQQVrF_N) for demonstrations 
of how to use this module.  

### 2. Other Modules

Coming soon.

## Contributing

Anyone is very welcome to contribute via a pull request. Please see the [Issues](https://github.com/p-christ/nn_builder/issues) 
page for ideas on the best areas to contribute to and try to:
1. Add tests to the tests folder that cover any code you write
1. Write comments for every function
1. Create a colab notebook demonstrating how any extra functionality you created works 

 

