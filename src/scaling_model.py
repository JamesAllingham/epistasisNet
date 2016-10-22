"""This module supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import utilities
import model

class ScalingModel(model.Model):
    """A class which builds a TensorFlow graph for a deep neural network with pooling and convolutional layers that works for large input sizes.

    The network structure is as follows:

    """

    def __init__(self, x, y1_, y2_, learning_rate):
        """Creates a ScalingModel.

        Inherits from Model.

        Parameters:
            x: the placeholder for the input tensor.
            y1_: the placeholder for the output 1 tensor.
            y2_: the placeholder for the output 2 tensor.

        Returns:
            A ScalingModel object.
        """
        model.Model.__init__(self)

        # get sizes for the input and outputs
        num_cols_in = x.get_shape().as_list()[1]
        num_states_out1 = y1_.get_shape().as_list()[1]
        num_cols_out2 = y2_.get_shape().as_list()[1]
        num_states_out2 = y2_.get_shape().as_list()[2]

        # first layer reshapes the input to make it 4d as required by the convolution layers
        x_4d = utilities.reshape(x, [-1, num_cols_in, 3, 1], name_suffix='1')

        # the first convolution layer preserves the shape and increases the number of channels to 8
        conv1 = utilities.conv_layer(x_4d, [3, 3, 1, 8], padding='SAME', name_suffix='1')

        # the first pooling layer simply halves the data size along the SNP dimmension
        pool1 = utilities.pool_layer(conv1, shape=[1, 2, 1, 1], strides=[1, 2, 1, 1], name_suffix='1')

        # the second convolution layer preserves the shape and increases the number of channels to 16
        conv2 = utilities.conv_layer(pool1, [3, 3, 8, 16], padding='SAME', name_suffix='2')

        # the second pooling layer halves the data size along the SNP dimmension
        pool2 = utilities.pool_layer(conv2, shape=[1, 2, 1, 1], strides=[1, 2, 1, 1], name_suffix='2')

        # the third convolution layer reduces reduces the number of states dimmension to size 1 and increases the number of channels to 32
        conv3 = utilities.conv_layer(pool2, [1, 3, 16, 32], padding='VALID', name_suffix='3')

        # the third pooling layer halves the data size along the SNP dimmension
        pool3 = utilities.pool_layer(conv3, shape=[1, 2, 1, 1], strides=[1, 2, 1, 1], name_suffix='3')

        # the next layer flattens the data so that it can be passed through a fully connected layer
        final_shape = pool3.get_shape()
        flatten_size = int(final_shape[1]*final_shape[2]*final_shape[3])
        flatten = utilities.reshape(pool3, [-1, flatten_size], name_suffix='2')

        # the first fully connected layer halves the data size
        hidden1_1 = utilities.fc_layer(flatten, flatten_size, 1000, layer_name='hidden_1_1', act=tf.identity)
        hidden1_2 = utilities.fc_layer(hidden1_1, 1000, int(flatten_size/2), layer_name='hidden_1_2')

        # the dropout layer reduces over fitting
        dropped1, self._keep_prob = utilities.dropout(hidden1_2, name_suffix='1')

        # the second fully connected layer halves the data size again
        hidden2_1 = utilities.fc_layer(dropped1, int(flatten_size/2), 1000, layer_name='hidden_2_1', act=tf.identity)
        hidden2_2 = utilities.fc_layer(hidden2_1, 1000, int(flatten_size/4), layer_name='hidden_2_2')

        dropped2, _ = utilities.dropout(hidden2_2, name_suffix='2', keep_prob=self._keep_prob)

        # the network splits here:
        # the first softmax layer reduces the output to a percentage chance for each of the output states
        output1 = utilities.fc_layer(dropped2, int(flatten_size/4), num_states_out1, layer_name='softmax_1', act=tf.nn.softmax)

        # the second softmax layer reduces the output to a percentage chance for each SNPs output states
        with tf.name_scope('softmax_2'):
            fc_layer_1 = utilities.fc_layer(dropped2, int(flatten_size/4), 1000, layer_name='identity_1', act=tf.identity)
            fc_layer_2 = utilities.fc_layer(fc_layer_1, 1000, num_states_out2*num_cols_out2, layer_name='identity_2', act=tf.identity)
            output2 = tf.nn.softmax(utilities.reshape(fc_layer_2, [-1, num_cols_out2, num_states_out2], name_suffix='3'))

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

        # find the top predicted snps
        self._epi_snps, self._count = utilities.predict_snps(output2)

        # merge all the summaries
        self._merged = tf.merge_all_summaries()
