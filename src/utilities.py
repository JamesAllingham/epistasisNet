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
    This needs to be slighly positive so that the ReLU activation functions aren't in an 'off' state

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
        return activations

def reshape(x, shape, name_suffix='1'):
    """Reshapes an input tensor to the given shape.

    Arguments:
        x: the input tensor to be reshaped.
        shape: an array describing the output tensor shape.

    Returns:
        a reshaped tensor.
    """
    with tf.name_scope('reshape_'+name_suffix):
        reshaped = tf.reshape(x, shape)
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
        with tf.name_scope('biases'):
            biases = bias_variable([shape[3]])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('convolution'):
            prebias = tf.nn.conv2d(x, kernel, strides, padding=padding)
            tf.histogram_summary(layer_name + '/pre_bias', prebias)
        with tf.name_scope('convolution_and_bias'):
            preactivate = tf.nn.bias_add(prebias, biases)
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
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
        return pooled

def dropout(x):
    """Apply dropout to a neural network layer.
    This is done to prevent over fitting.

    Arguments:
        x: the tensor which must have dropout applied to it.

    Returns:
        a dropped out version of the input tensor.
    """
    # Need to instantiate keep_prob here to correctly make the graph visualization
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(x, keep_prob, seed=42)
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

# #accuracy utilities

def calculate_accuracy(y, y_, snps_to_check=0, name_suffix='1'):
    """Compares the output of the neural network with the expected output and returns the accuracy.

    Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a scalar describing the accuracy of the given output when compared with the expected output.
    """
    # check for whether model has correctly predicted the snps causing epi
    with tf.name_scope('accuracy_'+name_suffix):
        if snps_to_check != 0:
            # split y
            y_left, _ = tf.split(2, 2, y, name='split')
            # transpose as top_k runs on rows not columns
            y_left_t = tf.transpose(y_left, perm=[0, 2, 1])
            # get k many max values
            values, _ = tf.nn.top_k(y_left_t, snps_to_check, name='snp_probabilities')
            # get smallest value as a 0D tensor
            min_value = tf.reduce_min(values, name='find_min_snp')
            # create tensor of smallest snp value same size as y_left
            ones_tensor = tf.ones([y.get_shape()[1], 1], tf.float32)
            min_value_tensor = tf.mul(ones_tensor, min_value)
            # compare all snps with that of min_value_tensor to find which have been predicted
            predicted_snps = tf.greater_equal(y_left, min_value_tensor, name='predicted_snps')
            # create mirrored tensor and concat together (needs to be in bool for xor), then cast to 1s and 0s
            y = tf.cast(tf.concat(2, [predicted_snps, tf.logical_xor(predicted_snps, tf.cast(ones_tensor, tf.bool))], name='concat'), tf.float32)

        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            # print("y: %s" % y.get_shape())
            # print("y_: %s" % y_.get_shape())
            # print("correct_prediction: %s" % correct_prediction.get_shape())
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy_'+name_suffix, accuracy)
        return accuracy

def calculate_accuracy_test(y, y_, snps_to_check=0, name_suffix='1'):
    """Compares the output of the neural network with the expected output and returns the accuracy.

    Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        snps_to_check: how many snps cause epi
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

    Returns:
        a scalar describing the accuracy of the given output when compared with the expected output.
    """

    # check for whether model has correctly predicted the snps causing epi
    with tf.name_scope('accuracy_'+name_suffix):
        if snps_to_check != 0:

            # split y
            y_left, _ = tf.split(2, 2, y, name='split')
            y_left_t = tf.transpose(y_left, perm=[0, 2, 1])

            ########################################## 
            # Attempt to find accuracy through parsing the case, epi-causing snp indexes

            # 50 percent certainty for confidence check
            point_five = tf.constant([0.5])
            # get yes/no labels
            labels_left, _ = tf.split(2, 2, y_, name='split')
            # labels_left_bool = tf.cast(labels_left, tf.bool)
            # transpose (?,1,10) -> (?,10,1) for top_k to work
            labels_left_t = tf.transpose(labels_left, perm=[0, 2, 1])
            # get labels that cause epi
            labels_values, labels_indices = tf.nn.top_k(labels_left_t, snps_to_check, name='snp_probabilities')
            # create check to only consider indices of epi snps when case not control
            ones_tensor_labels = tf.ones([labels_values.get_shape()[1],labels_values.get_shape()[2]], tf.float32)
            point_five_tensor_labels = tf.mul(ones_tensor_labels, point_five)
            greater_than_point_five_labels = tf.greater_equal(labels_values, point_five_tensor_labels, name='predicted_snps')
            greatest_indices_labels = tf.where(greater_than_point_five_labels)
            print("greatest_indices_labels: %s" % greatest_indices_labels.get_shape())
            # convert 'where' indexing to 'top_k' indexing
            greatest_indices_labels_left = tf.slice(greatest_indices_labels, [0,0], [-1,1])
            greatest_indices_labels_string = tf.cast(greatest_indices_labels, tf.string)
            print("greatest_indices_labels_left: %s" % greatest_indices_labels_left.get_shape())
            # gather indexes of the epi snp indices in the cases
            epi_snp_indices = tf.gather(labels_indices, greatest_indices_labels_left)
            # reshape from a 5D tensor to a manageable 2D tensor
            reshaped_epi_snp_indices = tf.reshape(epi_snp_indices, [-1])
            # epi_snp_indexes = tf.where(labels_left_bool_t, name='snp_indexes')

            # get k many max values
            values, indices = tf.nn.top_k(y_left_t, snps_to_check, name='snp_probabilities')
            # cut the predictions that aren't above .5 probability
            ones_tensor = tf.ones([values.get_shape()[1],values.get_shape()[2]], tf.float32)
            point_five_tensor = tf.mul(ones_tensor, point_five)
            greatest_indices = tf.where(tf.greater_equal(values, point_five_tensor, name='predicted_snps'))
            # convert 'where' indexing to 'top_k' indexing
            greatest_indices_left = tf.slice(greatest_indices, [0,0], [-1,1])
            # gather indexes of the predicted epi-causing snps
            predicted_snps = tf.gather(indices, greatest_indices_left)
            # reshape from a 5D tensor to a manageable 2D tensor
            reshaped_predicted_snps = tf.reshape(predicted_snps, [-1])

            diff, _ = tf.listdiff(reshaped_predicted_snps, reshaped_epi_snp_indices)

            # END
            ##########################################################################

            # THIS WORKS
            
            # get labelled snps
            # find indices of predictions >.5, and compare indices
            ones_tensor_y = tf.ones([y_left.get_shape()[1],y_left.get_shape()[2]], tf.float32)
            point_five = tf.mul(ones_tensor_y, point_five)
            greater_than_point_five_predictions_indices = tf.where(tf.greater_equal(y_left, point_five))
            # tf.gather_nd on labels_left to get a 0s/1s tensor
            gather_prediction_results = tf.gather_nd(labels_left, greater_than_point_five_predictions_indices)
            shape_of_results = tf.shape(gather_prediction_results)
            shape_of_results_float = tf.saturate_cast(shape_of_results.get_shape()[0], tf.float32)
            print("shape_of_results_float: %s" % type(shape_of_results_float))
            gather_prediction_results_bool = tf.cast(gather_prediction_results, tf.bool)
            # find number of 1s in gather_prediction_results with get_shape
            indices_of_ones = tf.where(gather_prediction_results_bool)
            number_of_ones = tf.shape(indices_of_ones)
            test_type = tf.saturate_cast(number_of_ones.get_shape()[0], tf.float32)
            # find number of 1s in labels_left
            indices_of_ones_labels = tf.where(tf.cast(labels_left,tf.bool))
            number_of_ones_labels = tf.shape(indices_of_ones_labels)
            # test_type_label = tf.saturate_cast(number_of_ones_labels.get_shape()[0], tf.float32)
            print("number_of_ones_labels: %s" % number_of_ones_labels.get_shape())
            # print("test_type_label: %s" % test_type_label.get_shape())
            # Do the maths
            epi_snps_missed = number_of_ones_labels - number_of_ones + shape_of_results
            test_accuracy = (tf.cast(number_of_ones, tf.float32) / tf.cast(epi_snps_missed, tf.float32))[0]

            ##########################################################################
            # find how many 1s are in labels_left (tf.get_shape()[0] on tf.where output )
            # find that many top_k in y_left
            # tf.gather_nd from top_k indices on 

            ##########################################################################

            greatest_values = tf.cast(tf.greater_equal(y_left, point_five), tf.float32)

            ############################################################
            #  FIRST ALGORITHM

            # get smallest value for each person to give a 1D tensor
            min_top_values = tf.slice(values, [0,0,snps_to_check-1], [-1,-1,snps_to_check-1])
            # reshaped_min_top_values = tf.reshape(min_top_values, [-1,1])
            # print("reshaped_min_top_values: %s" % reshaped_min_top_values.get_shape())
            # min_value = tf.reduce_min(values, name='find_min_snp')
            # min_value_test_check = tf.reduce_min(values, reduction_indices=1, name='find_min_snp')
            # create tensor of smallest snp value same size as y_left
            ones_tensor = tf.ones([y.get_shape()[1], 1], tf.float32)
            # min_value_tensor = tf.mul(ones_tensor, min_value)
            # compare all snps with that of min_value_tensor to find which have been predicted
            predicted_snps = tf.greater_equal(y_left, min_top_values, name='predicted_snps')
            # create mirrored tensor and concat together (needs to be in bool for xor), then cast to 1s and 0s
            y_remade = tf.cast(tf.concat(2, [predicted_snps, tf.logical_xor(predicted_snps, tf.cast(ones_tensor, tf.bool))], name='concat'), tf.float32)
            # print('y: %s' % y.get_shape())
            with tf.name_scope('correct_prediction'):
                # correct_prediction = tf.equal(y_remade, y_)
                correct_prediction = tf.equal(tf.argmax(y_remade, 2), tf.argmax(y_, 2))
                # correct_prediction_left, _ = tf.split(2, 2, correct_prediction, name='split')
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # END
            ############################################################

            tf.scalar_summary('accuracy_'+name_suffix, accuracy)
            return accuracy, number_of_ones, epi_snps_missed, test_accuracy

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
