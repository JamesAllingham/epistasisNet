"""This module supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import utilities
import model

class ConvolutionalModel(model.Model):
    """A class which builds a TensorFlow graph for a deep neural network with convolutional and fully connected layers.

    The network structure is as follows:

    input --> reshape --> conv1 --> conv2 --> reshape --> hidden1 --> hidden2 --> softmax
                                                                              --> softmax
    [?, x, 3] --> [?, x, 3, 1] --> [?, x, 3, 8] --> [?, x, 3, 16] --> [?, 48x, 1] --> [?, 12x, 1] --> [?, 6x, 1] --> [?, 2, 1]
                                                                                                                 --> [?, 2, x]
    """

    def __init__(self, x, y1_, y2_, learning_rate):
        """Creates a ConvolutionalModel.

        Inherits from Model.

        Parameters:
            x: the placeholder for the input tensor.
            y1_: the placeholder for the output 1 tensor.
            y2_: the placeholder for the output 2 tensor.

        Returns:
            A ConvolutionalModel object.
        """
        model.Model.__init__(self)

        # get sizes for the input and outputs
        num_cols_in = x.get_shape().as_list()[1]
        num_states_in = x.get_shape().as_list()[2]
        num_states_out1 = y1_.get_shape().as_list()[1]
        num_cols_out2 = y2_.get_shape().as_list()[1]
        num_states_out2 = y2_.get_shape().as_list()[2]

        # first layer reshapes the input to make it 4d as required by the convolution layers
        x_4d = utilities.reshape(x, [-1, num_cols_in, num_states_in, 1])

        # the first convolution layer preserves the shape and increases the number of channels to 8
        conv1 = utilities.conv_layer(x_4d, [3, 3, 1, 8], padding='SAME', name_suffix='1')
        # the first convolution layer preserves the shape and increases the number of channels to 16
        conv2 = utilities.conv_layer(conv1, [3, 3, 8, 16], padding='SAME', name_suffix='2')

        # the next layer flattens the data so that it can be passed through a fully connected layer
        flatten = utilities.reshape(conv2, [-1, num_cols_in*num_states_in*16], name_suffix='2')

        # the first hidden layer reduces the data size to the number of outputs
        hidden1 = utilities.fc_layer(flatten, num_cols_in*num_states_in*16, num_cols_in*num_states_in*4, layer_name='hidden_1')
        # the second hidden layer halves the data size
        hidden2 = utilities.fc_layer(hidden1, num_cols_in*num_states_in*4, num_cols_in*num_states_in*2, layer_name='hidden_2')

        # the dropout layer reduces overfitting
        dropped, self._keep_prob = utilities.dropout(hidden2)

        # the network splits here:
        # the first softmax layer reduces the output to a percentage chance for each of the output states
        output1 = utilities.fc_layer(dropped, 2*num_cols_in*num_states_in, num_states_out1, 'softmax_1', act=tf.nn.softmax)

        # the second softmax layer reduces the output to a percentage chance for each SNPs output states
        with tf.name_scope('softmax_2'):
            fc_layer = utilities.fc_layer(dropped, 2*num_cols_in*num_states_in, num_states_out2*num_cols_out2, layer_name='identity', act=tf.identity)
            output2 = tf.nn.softmax(utilities.reshape(fc_layer, [-1, num_cols_out2, num_states_out2], name_suffix='3'))

        # each of the loss layers compares the probability distributions between the correspinding outputs to get an error metric for the network's outputs
        self._loss1 = utilities.calculate_cross_entropy(output1, y1_, name_suffix='1')
        self._loss2 = utilities.calculate_cross_entropy(output2, y2_, name_suffix='2')
        # these losses are compined into one for the training
        with tf.name_scope('combined_loss'):
            combined_loss = tf.add(self._loss1, self._loss2)

        # the loss is used with the back propagtion algorithm to use gradient descent based ADAM optimization to teach the network
        self._train_step = utilities.train(learning_rate, combined_loss, training_method=utilities.Optimizer.Adam, name_suffix='1')

        # the accuracies for each output are calculated by comparing them to the correct outputs
        self._accuracy1 = utilities.calculate_epi_accuracy(output1, y1_, name_suffix='1')
        self._accuracy2 = utilities.calculate_snp_accuracy(output2, y2_, name_suffix='2')

        # merge all the summaries
        self._merged = tf.merge_all_summaries()
        