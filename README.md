![Image](https://travis-ci.org/p-christ/nn_builder.svg?branch=master) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) 


![nn_builder](miscellaneous/material_for_readme/nn_builder_new.png)


**nn_builder builds neural networks in 1 line** to save you time :clock1:. You specify the type of network you want and it builds it.

### Install

`pip install nn_builder`


### Support

| Network Type       | **NN**  | **CNN** | **RNN** |
| ------- | ------- | ------- | ------- |
| PyTorch     | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:    |
| TensorFlow 2.0  |        :heavy_check_mark:  |  :heavy_check_mark: | :heavy_check_mark: |                             |



### Examples

On the left is how you can create the neural network on the right in only 1 line of code using nn_builder:

![Screenshot](miscellaneous/material_for_readme/nn_builder_use_case.png)

Tensorflow example... 

See this [colab notebook](https://colab.research.google.com/drive/1abxTEaUrJqbTuk8e8tOa3y9DYQQVrF_N) for demonstrations 
of how to use this module.  


### Usage

Every network requires the same arguments: 

| Field | Description | Default |
| :---: | :----------: | :---: |
| *input_dim*| Dimension of the input into the network excluding the batch size. Not needed for Tensorflow.  | N/A |
| *layers_info* | List of integers to indicate the width and number of linear layers you want in your network | N/A |
| *output_activation* | String to indicate the activation function you want the output to go through. Provide a list of strings if you want multiple output heads | No activation |                              
| *hidden_activations* | String or list of string to indicate the activations you want used on the output of hidden layers (not including the output layer), default is ReLU and for example "tanh" would have tanh applied on all hidden layer activations | ReLU after every hidden layer |
| *dropout* | Float to indicate what dropout probability you want applied after each hidden layer | 0 |
| *initialiser* | String to indicate which initialiser you want used to initialise all the parameters | PyTorch Default |
| *batch_norm* | Boolean to indicate whether you want batch norm applied to the output of every hidden layer | False |
| *columns of_data_to_be_embedded* | List to indicate the column numbers of the data that you want to be put through an embedding layer before being fed through the hidden layers of the network | No embeddings |
| *embedding_dimensions* | If you have categorical variables you want embedded before flowing through the network then you specify the embedding dimensions here with a list of the form: [ [embedding_input_dim_1, embedding_output_dim_1], [embedding_input_dim_2, embedding_output_dim_2] ...] | No embeddings |
| *y_range* | Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the output values to in regression tasks | No range |
| *random_seed* | Integer to indicate the random seed you want to use | 0 |
| *print_model_summary* | Boolean to indicate whether you want a model summary printed after model is created | False |


After creating a model, the function print_model_summary() can be used to summarise the layers.  

## More Examples

## Contributing

Anyone is very welcome to contribute via a pull request. Please see the [Issues](https://github.com/p-christ/nn_builder/issues) 
page for ideas on the best areas to contribute to and try to:
1. Add tests to the tests folder that cover any code you write
1. Write comments for every function
1. Create a colab notebook demonstrating how any extra functionality you created works 

 



