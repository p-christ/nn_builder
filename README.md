


![Image](https://travis-ci.org/p-christ/nn_builder.svg?branch=master) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) ![nn_builder](miscellaneous/material_for_readme/nn_builder.png)





**nn_builder builds neural networks in 1 line**. It saves you time by removing the need for boilerplate code when making neural networks. 
It is also well tested so you know you aren't making mistakes when creating a network this way. 

### Install

`pip install nn_builder`

### Example

On the right is the large amount of code you would need to write to create a PyTorch neural network equivalent to the network
 created in only one line of code on the left using nn_builder: 

![Screenshot](miscellaneous/material_for_readme/nn_builder_use_case.png)

### Usage

Types of network supported so far: PyTorch NN, CNN & RNN. Soon we will add Tensorflow networks.


| **Network Type**         | **PyTorch**                       | **TensorFlow**                    |
| --------------------------- | --------------------------------- | --------------------------------- |
|             NN              | :heavy_check_mark:                | :x:                               |
| CNN                         | :heavy_check_mark:                | :x:                               |
| RNN                         | :heavy_check_mark:                | :x:                               |

 

#### 1. PyTorch NN Module

The PyTorch NN module lets you build feedforward (rather than CNNs or RNNs) PyTorch neural networks in one line. 

First run `from nn_builder.pytorch.NN import NN` and then NN takes the below arguments:

| Field | Description | Default |
| :---: | :----------: | :---: |
| *input_dim*| Integer to indicate the dimension of the input into the network | N/A |
| *layers* | List of integers to indicate the width and number of linear layers you want in your network | N/A |
| *output_activation* | String to indicate the activation function you want the output to go through. Provide a list of strings if you want multiple output heads | No activation |                              
| *hidden_activations* | String or list of string to indicate the activations you want used on the output of hidden layers (not including the output layer), default is ReLU and for example "tanh" would have tanh applied on all hidden layer activations | ReLU after every hidden layer |
| *dropout* | Float to indicate what dropout probability you want applied after each hidden layer | 0 |
| *initialiser* | String to indicate which initialiser you want used to initialise all the parameters. All PyTorch initialisers are supported. | PyTorch Default |
| *batch_norm* | Boolean to indicate whether you want batch norm applied to the output of every hidden layer | False |
| *columns_of_data_to_be_embedded* | List to indicate the column numbers of the data that you want to be put through an embedding layer before being fed through the hidden layers of the network | No embeddings |
| *embedding_dimensions* | If you have categorical variables you want embedded before flowing through the network then you specify the embedding dimensions here with a list of the form: [ [embedding_input_dim_1, embedding_output_dim_1], [embedding_input_dim_2, embedding_output_dim_2] ...] | No embeddings |
| *y_range* | Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the output values to in regression tasks | No range |
| *print_model_summary* | Boolean to indicate whether you want a model summary printed after model is created | False |

See this [colab notebook](https://colab.research.google.com/drive/1abxTEaUrJqbTuk8e8tOa3y9DYQQVrF_N) for demonstrations 
of how to use this module.  

#### 2. PyTorch CNN Module

The PyTorch CNN module lets you build convolutional PyTorch neural networks in one line. 

First run `from nn_builder.pytorch.CNN import CNN` and then CNN takes the below arguments:


## Contributing

Anyone is very welcome to contribute via a pull request. Please see the [Issues](https://github.com/p-christ/nn_builder/issues) 
page for ideas on the best areas to contribute to and try to:
1. Add tests to the tests folder that cover any code you write
1. Write comments for every function
1. Create a colab notebook demonstrating how any extra functionality you created works 

 

