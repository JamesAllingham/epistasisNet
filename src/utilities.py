import tensorflow as tf
import numpy as np
import sys
import getopt
from enum import Enum

# # Variable utilities
def tn_weight_variable(shape, stddev_value):
  initial = tf.truncated_normal(shape=shape, stddev=stddev_value, seed=42)
  return tf.Variable(initial)

def zeros_weight_variable(shape):
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  # Create a bias variable with appropriate initialization - slightly positive to avoid deadness.
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# # nn utilities

def identity(x, name):  
  return x

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = tn_weight_variable([input_dim, output_dim], 0.1)
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

# def nn_layer(layer_name, ):
#   # Ideally want different activation types

#   with tf.name_scope(layer_name):

def reshape(x, num_cols_out, num_states_out):
  # Reshape a tensor into desired type
  with tf.name_scope('reshape'):
    if num_states_out==1:
      flattened = tf.reshape(x, [-1, num_cols_out])
    else:
      flattened = tf.reshape(x, [-1, num_cols_out, num_states_out])
  return flattened

def dropout(x):
  # Need to instantiate keep_prob here to correctly make the graph visualization
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(x, keep_prob, seed=42)
  return dropped, keep_prob

# loss function utilities

def calculate_cross_entropy(y, y_, name='1'):
  # computes cross entropy between trained y and label y_
  with tf.name_scope('cross_entropy_'+name):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross_entropy_'+name, cross_entropy)
  return cross_entropy

# training utilities

class Optimizer(Enum):
  GradientDescent = 1
  Adam = 2

def train(training_type, learning_rate, loss_value):
  # Call the optimizer to train the net
  with tf.name_scope('train'):
    if training_type == Optimizer.GradientDescent:
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value)
    elif training_type == Optimizer.Adam:
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_value)
  return train_step


# #accuracy utilities

def calculate_accuracy (y, y_, name='1'):
  with tf.name_scope('accuracy_'+name):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy_'+name, accuracy)
  return accuracy

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)