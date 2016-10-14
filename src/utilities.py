"""This module provides a number of wrapper functions for tensorflow.

These wrappers make it easier to create a computation graph with lebeled names and varaible summaries.

The following functions are avaialbe:
   tn_weight_variable: creates a matrix with a given shape sampling initial values from a truncated normal distribution.
   zeros_weight_varaible: creates a matrix with a given shape using zeros as the inial values.
   bias_varaible: creates a bais vector with initial valies of 0.1.
   fc_layer: creates a fully conected neural network layer that performas a Wx + b computation.
   rehape: creates a layer that reshapes its input to the desired shape.
   conv_layer: creates a convolutional layer with the given filter shape, padding, and strides.
   pool_layer: creates a max a pooling layer with the given shape, padding, and strides.
   dropout: creates a dropout layer with the given dropout rate.
   calculate_cross_entropy: calclulates the cross entropy between two given distributions.
   train: applies the selected optimization method to train the neural network parameters.
   calculate_accuracy: returns the accuracy of the given output compared with the specified desired output.
   varaible_summaries: atatches a number of summary operations to a given variable.
"""

import tensorflow as tf
import numpy as np
from enum import Enum


# # Variable utilities
def tn_weight_variable(shape, standard_deviation):
    """Create a weight matrix with the given shape.
    The weights are initialised with random values taken from a tuncated normal distribution.

    Arguments:
        shape: an array describing the shape of the weight matrix.
        standard_deviation: the standard deviation of the truncted normal distribution.

    Returns:
        a tf.Variable containing the weight matrix.
    """
    initial = tf.truncated_normal(shape=shape, stddev=standard_deviation, seed=42)
    return tf.Variable(initial)

def zeros_weight_variable(shape):
    """Create a weight matrix with the given shape.
    The weights are initialised with zeros.


    Arguments:
        shape: an array describing the shape of the weight matrix.

    Returns:
        a tf.Variable containing the weight matrix.
    """
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization.
    This needs to be slighly positive so that the ReLU activation functions aren't in an 'off' state.

    Arguments:
        shape: an array describing the shape of the bias vector.

    Returns:
        a tf.Variable containing the bias vector.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# # nn utilities

def fc_layer(x, input_dim, output_dim, layer_name, standard_deviation=0.1, act=tf.nn.relu):
    """Reusable code for making a hidden neural net layer.
    It does a matrix multiply, bias add, and then adds a nonlinearity.
    It also sets up name scoping so that the resultant graph is easy to read, and adds a number of summary ops.

    Arguments:
        x: the tensor which must travel through the layer.
        input_dim: the input tensor's dimension.
        output_dim: the output tensor's dimension.
        layer_name: the layer name for the graph visualization.
        act: the activation function to be applied to the output tensor before it is returned. The default is ReLU.

    Returns:
        the result of passing the input tensor through the Mx + b and activation layers.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = tn_weight_variable([input_dim, output_dim], standard_deviation)
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(x, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        print("%s shape: %s" % (layer_name, activations.get_shape()))
        return activations

def reshape(x, shape, name_suffix='1'):
    """Reshapes an input tensor to the given shape.

    Arguments:
        x: the input tensor to be reshaped.
        shape: an array describing the output tensor shape.

    Returns:
        a reshaped tensor.
    """
    layer_name = 'reshape_'+name_suffix
    with tf.name_scope(layer_name):
        reshaped = tf.reshape(x, shape)
    print("%s shape: %s" % (layer_name, reshaped.get_shape()))
    return reshaped

def conv_layer(x, shape, strides=[1, 1, 1, 1], standard_deviation=0.1, padding='SAME', name_suffix='1', act=tf.nn.relu):
    """Reusable code for making a convolutional neural net layer.
    It applies a convoltion to the input
    It also sets up name scoping so that the resultant graph is easy to read, and adds a number of summary ops.

    Arguments:
        x: the tensor which must travel through the layer. The tensor must have shape: [batch, in_height, in_width, in_channels]
        shape: an array describing the shape of the kernel: [filter_height, filter_width, in_channels, out_channels]
        strides: an array describing how often the filter is applied: [1, stride_horizontal, stride_verticle, 1].
        padding: the padding scheme applied by the convolutinal filter see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution for more details.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.
        act: the activation function to be applied to the output tensor before it is returned. The default is ReLU.

    Returns:
        the result of passing the input tensor through the convolutional layer, with a number of channels equal to out_channels.
    """
    layer_name = 'conv_' + name_suffix
    with tf.variable_scope(layer_name) as scope:
        with tf.name_scope('kernel'):
            kernel = tn_weight_variable(shape, standard_deviation)
            variable_summaries(kernel, layer_name + '/kernel')
        with tf.name_scope('convolution'):
            preactivate = tf.nn.conv2d(x, kernel, strides, padding=padding)
            tf.histogram_summary(layer_name + '/preactivate', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        print("%s shape: %s" % (layer_name, activations.get_shape()))
        return activations

def pool_layer(x, shape=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name_suffix='1'):
    """Reusable code for making a convolutional neural net layer.
    It applies a convoltion to the input
    It also sets up name scoping so that the resultant graph is easy to read, and adds a number of summary ops.

    Arguments:
        x: the tensor which must travel through the layer. The tensor must have shape: [batch, in_height, in_width, in_channels]
        shape: an array describing the shape of the kernel: [1, width, height, 1]
        strides: an array describing how often the pooling is applied: [1, stride_horizontal, stride_verticle, 1].
        padding: the padding scheme applied by the pooling layer see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution for more details.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        the result of passing the input tensor through the pooling layer.
    """
    layer_name = 'pool_' + name_suffix
    with tf.variable_scope(layer_name) as scope:
        with tf.name_scope('max_pooling'):
            pooled = tf.nn.max_pool(x, shape, strides, padding)
        print("%s shape: %s" % (layer_name, pooled.get_shape()))
        return pooled

def dropout(x, name_suffix='1'):
    """Apply dropout to a neural network layer.
    This is done to prevent over fitting.

    Arguments:
        x: the tensor which must have dropout applied to it.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a dropped out version of the input tensor.
    """
    # Need to instantiate keep_prob here to correctly make the graph visualization
    layer_name = 'dropout_'+name_suffix
    with tf.name_scope(layer_name):
        keep_prob = tf.placeholder(tf.float32)
        dropped = tf.nn.dropout(x, keep_prob, seed=42)
    print("%s shape: %s" % (layer_name, dropped.get_shape()))
    return dropped, keep_prob

# loss function utilities

def calculate_cross_entropy(y, y_, name_suffix='1'):
    """Calculate the cross entropy as a loss function.

    Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        the cross entropy of the expexted and given outputs.
    """
    # computes cross entropy between trained y and label y_
    with tf.name_scope('cross_entropy_'+name_suffix):
        diff = y_ * tf.log(y + 1e-10)
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross_entropy_'+name_suffix, cross_entropy)
        return cross_entropy

# training utilities

class Optimizer(Enum):
    """ An Enumeration class for the different types of optimizers.

        The following optimizers are avaialbe:
            1. GradientDescent
            2. Adam
            3. Adadelta
            4. Adagrad
            5. RMSProp
            6. Ftrl (only cpu compatible)
    """
    GradientDescent = 1
    Adam = 2
    Adadelta = 3
    Adagrad = 4
    RMSProp = 5
    Ftrl = 6

def train(learning_rate, loss_function, training_method=Optimizer.GradientDescent, name_suffix='1'):
    """Call the optimizer to train the neural network.
    The options for the Optimizer are GradientDescent, Adam, Adadelta, Adagrad, RMSProp, and Frlr.

    Arguments:
        learning_rate: a scalar describing how fast the network should learn.
        loss_function: the function for calcualting the loss which must be minimized.
        training_method: the method used to minimize the loss. The default is GradientDescent.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a tf session that can be run to train the network.
    """
    with tf.name_scope('train_'+name_suffix):
        if training_method == Optimizer.GradientDescent:
            train_step = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent_"+name_suffix).minimize(loss_function)
        elif training_method == Optimizer.Adam:
            train_step = tf.train.AdamOptimizer(learning_rate, name="Adam_"+name_suffix).minimize(loss_function)
        elif training_method == Optimizer.Adadelta:
            train_step = tf.train.AdadeltaOptimizer(learning_rate, name="Adadelta_"+name_suffix).minimize(loss_function)
        elif training_method == Optimizer.Adagrad:
            train_step = tf.train.AdagradOptimizer(learning_rate, name="Adagrad_"+name_suffix).minimize(loss_function)
        elif training_method == Optimizer.RMSProp:
            train_step = tf.train.RMSPropOptimizer(learning_rate, name="RMSProp_"+name_suffix).minimize(loss_function)
        elif training_method == Optimizer.Ftrl:
            train_step = tf.train.FtrlOptimizer(learning_rate, name="Ftrl_"+name_suffix).minimize(loss_function)
    return train_step

# accuracy utilities

def calculate_epi_accuracy(y, y_, snps_to_check=0, name_suffix='1'):
    """Compares the epi output of the neural network with the expected epi output and returns the accuracy.

    Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a scalar describing the accuracy of the given output when compared with the expected output.
    """
    with tf.name_scope('epi_accuracy_'+name_suffix):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('epi_accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('epi_accuracy_'+name_suffix, accuracy)
        return accuracy

def calculate_snp_accuracy(y, y_, name_suffix='1'):
    """Compares the snp output of the neural network with the expected snp output and returns the accuracy.

    Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a scalar describing the accuracy of the given snp output when compared with the expected snp output.
    """

    # split y, labels
    with tf.name_scope('snp_accuracy_'+name_suffix):
        y_left = get_causing_epi_probs(y)
        labels_left = get_causing_epi_probs(y_)
        with tf.name_scope('predictions'):
            # find indices of predictions >.5
            predicted_snps = tf.where(tf.greater_equal(y_left, 0.5))
            # tf.gather_nd on labels_left to get a 0s/1s tensor -> stored at [0] on shape (2)
            prediction_results = tf.gather_nd(labels_left, predicted_snps)
            shape_of_output = tf.shape(prediction_results)
        # find number of 1s -> stored at [0] on shape (2)
        with tf.name_scope('correct_predictions'):
            correct_predictions = tf.shape(tf.where(tf.cast(prediction_results, tf.bool)))
        # find number of 1s in labels_left -> stored at [0] on shape (2)
        with tf.name_scope('expected_output'):
            indices_of_ones_labels = tf.where(tf.cast(labels_left, tf.bool))
            number_of_ones_labels = tf.shape(indices_of_ones_labels)
        # Find predictions_length = incorrect predictions + correct predictions + missed predictions -> stored at [0] on shape (2)
        with tf.name_scope('all_predictions'):
            all_predictions = number_of_ones_labels - correct_predictions + shape_of_output
        # accuracy = correct_predictions / all_predictions -> stored at [0] on shape (2)
        with tf.name_scope('snp_accuracy'):
            accuracy = (tf.cast(correct_predictions, tf.float32) / tf.cast(all_predictions, tf.float32))[0]
        tf.scalar_summary('snp_accuracy_'+name_suffix, accuracy)
        return accuracy

def predict_snps(y, snps_to_predict, test=False):
    """Predicts which snps are causing epistasis based on one epoch and how many snps to detect

    Arguments:
        y: the given output tensor
        snps_to_predict: an integer defining the number of snps to return

    Returns:
        predicted_snps: a tensor with the indices of the predicted snps
    """
    with tf.name_scope('snp_prediction'):
        y_left = get_causing_epi_probs(y)
        y_left_t = tf.transpose(y_left, [0, 2, 1])
        if not test:
            _, predicted_snps = tf.nn.top_k(y_left_t, snps_to_predict, name='find_top_snps')
            reshaped = tf.reshape(predicted_snps, [-1])
            top_pred_snps, _, count = tf.unique_with_counts(reshaped)
            count = count + [0, 1, 0]
            _, top_counts = tf.nn.top_k(count, 1, name='strip_low_values')
            predictions = tf.gather(top_pred_snps, top_counts)
            return top_pred_snps, count, predictions
        else:
            top_snps = tf.where(tf.greater_equal(y_left, 0.5))
            print('top_snps: %s' % top_snps)
            _, top_snp_indices, _ = tf.split(1, 3, top_snps, name='split')
            top_snp_indices = tf.reshape(top_snp_indices, [-1])
            top_pred_snps, _, count = tf.unique_with_counts(top_snp_indices)
            return top_pred_snps

def get_snp_headers(snp_labels, headers):
    """Finds the header names for the snp labels

    Arguments:
        snp_labels: a numpy array with the snp labels to find
        headers: a numpy array with the headers for all the snp columns

    Returns:
        snp_headers: a numpy array with the snp label names
    """
    snp_names = np.array([])
    for snp in np.nditer(snp_labels):
        snp_names = np.append(snp_names, headers[snp])
    return snp_names

def get_causing_epi_probs(tensor_in):
    """Gets the 'causing epi' probabilities on a (?, ?, 2) tensor containing predictions of whether snps are causing epi

    Arguments:
        y: the tensor to split

    Returns:
        y_left: a tensor with the 'causing epi' probabilities
    """
    with tf.name_scope('split'):
        left, _ = tf.split(2, 2, tensor_in, name='split')
        return left

def variable_summaries(var, name):
    """Attach min, max, mean, and standard deviation summaries to a variable.

    Arguments:
        var: the tf.Variable to be summarised.
        name: the name to display on the graph visualization.

    Returns:
        nothing.
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
