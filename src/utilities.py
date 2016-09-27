import tensorflow as tf
import numpy as np
import sys
import getopt
from enum import Enum

# # Variable utilities
def tn_weight_variable(shape, standard_deviation):
  """ Create a weight matrix with the given shape. 
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
  """ Create a weight matrix with the given shape. 
      The weights are initialised with zeros.


      Arguments:
        shape: an array describing the shape of the weight matrix.

      Returns:
        a tf.Variable containing the weight matrix.
  """
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  """ Create a bias variable with appropriate initialization.
      This needs to be slighly positive so that the ReLU activation functions aren't in an 'off' state

      Arguments:
        shape: an array describing the shape of the bias vector.

      Returns:
        a tf.Variable containing the bias vector.
  """
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# # nn utilities

def identity(x, name):  
  """ identity function that can be used as the activation function of the fc_layer function to create a linear layer.

      Arguments:
        x: the tensor which must be 'activated'. 
        name: the scope name for the graph visualization.

      Returns:
        the 'activated' tensor.
  """
  with tf.name_scope(name):
    return x

def fc_layer(x, input_dim, output_dim, layer_name, standard_deviation=0.1, act=tf.nn.relu):
  """ Reusable code for making a hidden neural net layer.
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

def reshape(x, shape):
  """ Reshapes an input tensor to the given shape.

      Arguments:
        x: the input tensor to be reshaped.
        shape: an array describing the output tensor shape.

      Returns:
        a reshaped tensor.
  """
  with tf.name_scope('reshape'):
    reshaped = tf.reshape(x, shape) 
  return reshaped

def conv_layer(x, kernel_shape, standard_deviation=0.1, filter_strides=[1,1,1,1], filter_padding='SAME', name_suffix='1', act=tf.nn.relu):
  """ Reusable code for making a convolutional neural net layer.
      It applies a convoltion to the input
      It also sets up name scoping so that the resultant graph is easy to read, and adds a number of summary ops.

      Arguments:
        x: the tensor which must travel through the layer. The tensor must have shape: [batch, in_height, in_width, in_channels]
        kernel_shape: an array describing the shape of the kernel: [filter_height, filter_width, in_channels, out_channels]
        filter_strides: an array describing how often the filter is applied: [1, stride_horizontal, stride_verticle, 1].
        filter_padding: the padding scheme applied by the convolutinal filter see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#convolution for more details.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.
        act: the activation function to be applied to the output tensor before it is returned. The default is ReLU.
      
      Returns:
        the result of passing the input tensor through the convolutional layer, with a number of channels equal to out_channels.
  """
  layer_name = 'conv_' + name_suffix
  with tf.variable_scope(layer_name) as scope:
    with tf.name_scope('kernel'):
      kernel = tn_weight_variable(kernel_shape, standard_deviation)
      variable_summaries(kernel, layer_name + '/kernel') 
    with tf.name_scope('biases'):
      biases = bias_variable([kernel_shape[3]])
      variable_summaries(biases, layer_name + '/biases')   
    with tf.name_scope('convolution_and_bias'):   
      preactivate = tf.nn.conv2d(x, kernel, filter_strides, padding=filter_padding)
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def dropout(x):
  """ Apply dropout to a neural network layer.
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
  """ Calculate the cross entropy as a loss function.

      Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

      Returns:
        the cross entropy of the expexted and given outputs.
  """
  # computes cross entropy between trained y and label y_
  with tf.name_scope('cross_entropy_'+name_suffix):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross_entropy_'+name_suffix, cross_entropy)
  return cross_entropy

# training utilities

class Optimizer(Enum):
  GradientDescent = 1
  Adam = 2
  Adadelta = 3
  Adagrad = 4
  RMSProp = 5
  Ftrl = 6

def train(learning_rate, loss_function, training_method=Optimizer.GradientDescent, name_suffix='1'):
  """ Call the optimizer to train the neural network. 
      The options for the Optimizer are Gradient Descent and Adam.

      Arguments:
        learning_rate: a scalar describing how fast the network should learn.
        loss_function: the function for calcualting the loss which must be minimized.
        training_method: the method used to minimize the loss. The default is gradient descent.
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

def calculate_accuracy (y, y_, snps_to_check=0, name_suffix='1'):
  """ Compares the output of the neural network with the expected output and returns the accuracy.

      Arguments:
        y: the given output tensor.
        y_: the expected output tensor.
        name_suffix: the suffix of the name for the graph visualization. The default value is '1'.

      Returns:
        a scalar describing the accuracy of the given output when compared with the expected output.
  """
  with tf.name_scope('accuracy_'+name_suffix):
    if snps_to_check != 0:
      print('y: %s, y_: %s' % (y.get_shape(), y_.get_shape()))
      # split y
      y_left, y_right = tf.split(2, 2, y, name = 'split')
      print('y_left: %s, y_right: %s' % (y_left.get_shape(), y_right.get_shape()))

      # get k many max values
      values, indices = tf.nn.top_k(y, snps_to_check, name='snp_probabilities')
      print('values: %s' % values.get_shape())

      # get smallest value as a 0D tensor
      min_value = tf.reduce_min(values, name='find_min')
      print('min_value: %s' % min_value.get_shape())

      # create tensor of smallest snp value same size as y_left
      ones_tensor = tf.ones([y.get_shape()[1], 1], tf.float32)
      true_bool_tensor = tf.cast(ones_tensor, tf.bool)
      print('ones_tensor: %s' % ones_tensor.get_shape())

      min_value_tensor = tf.mul(ones_tensor, min_value)
      print('min_value_tensor: %s' % min_value_tensor.get_shape())

      # compare all snps with that of min_value_tensor to find which are >= to it
      predicted_snps = tf.greater_equal(y_left, min_value_tensor, name='predicted_snps')
      print('predicted_snps: %s' % predicted_snps.get_shape())

      # create mirrored tensor and concat together (needs to be in bool for xor)
      prediction_tensor_bool = tf.concat(2, [predicted_snps, tf.logical_xor(predicted_snps, true_bool_tensor)], name='concat')

      # cast to 1s and 0s
      y = tf.cast(prediction_tensor_bool, tf.float32)
      print('prediction_tensor: %s' % y.get_shape())
      
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy_'+name_suffix, accuracy)
  return accuracy

def variable_summaries(var, name):
  """ Attach min, max, mean, and standard deviation summaries to a variable.

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