"""This module supplies a fully connected model with nonlinearities to test for epistasis on a GAMETES dataset
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import utilities
import model

class NonLinearModel(model.Model):
    """A class which builds a TensorFlow graph for a deep neural network with only non-linear fully connected layers.

    The network structure is as follows:

    input --> reshape --> hidden --> softmax
                                 --> softmax
    [?, x, 3] --> [?, 3x] --> [?, 6x] --> [?, 2, 1]
                                      --> [?, 2, x]
    """

    def __init__(self, x, y1_, y2_, learning_rate):
        """Creates a NonLinearModel.

        Inherits from Model.

        Parameters:
            x: the placeholder for the input tensor.
            y1_: the placeholder for the output 1 tensor.
            y2_: the placeholder for the output 2 tensor.

        Returns:
            A NonLinearModel object.
        """
        model.Model.__init__(self)

        # get sizes for the input and outputs
        num_cols_in = x.get_shape().as_list()[1]
        num_states_in = x.get_shape().as_list()[2]
        num_states_out1 = y1_.get_shape().as_list()[1]
        num_cols_out2 = y2_.get_shape().as_list()[1]
        num_states_out2 = y2_.get_shape().as_list()[2]

        # the first layer flattens the data so that it can be passed through a fully connected layer
        x_flat = utilities.reshape(x, [-1, num_cols_in*num_states_in])

        hidden1 = utilities.fc_layer(x_flat, num_cols_in*num_states_in, num_cols_in*num_states_in*4, layer_name='hidden_1')
        hidden2 = utilities.fc_layer(hidden1, num_cols_in*num_states_in*4, num_cols_in*num_states_in*2, layer_name='hidden_2')

        # the dropout layer reduces over fitting
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

        # find the top predicted snps
        self._epi_snps, self._count = utilities.predict_snps(output2)

        # merge all the summaries
        self._merged = tf.merge_all_summaries()
        