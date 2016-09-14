from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data_holder
import sys
import getopt
import utilities

def train(file_name_and_path, test_train_ratio, log_file_path, max_steps, learning_rate, dropout_rate):

  # Import data
  dh = data_holder.DataHolder(file_name_and_path, test_train_ratio, 1)

  # get the data dimmensions
  num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
  num_rows_out, num_states_out = dh.get_training_data().get_output_shape()

  tf.set_random_seed(42)
  sess = tf.InteractiveSession()

  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_states_out], name='y-input')

  # We can't initialize these variables to 0 - the network will get stuck.
  # def weight_variable(shape):
  #   """Create a weight variable with appropriate initialization."""
  #   initial = tf.truncated_normal(shape, stddev=0.1)
  #   return tf.Variable(initial)

  # def bias_variable(shape):
  #   """Create a bias variable with appropriate initialization."""
  #   initial = tf.constant(0.1, shape=shape)
  #   return tf.Variable(initial)

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
        weights = utilities.tn_weight_variable([input_dim, output_dim], 0.1)
        variable_summaries(weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        biases = utilities.bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
      activations = act(preactivate, 'activation')
      tf.histogram_summary(layer_name + '/activations', activations)
      return activations
  
  x_flat = utilities.reshape(x, num_cols_in * num_states_in, 1)

  hidden1 = nn_layer(x_flat, num_cols_in*num_states_in, 500, 'layer1')

  dropped, keep_prob = utilities.dropout(hidden1)

  y = nn_layer(dropped, 500, num_states_out, 'layer2', act=tf.nn.softmax)
  # W_1 = tf.Variable(tf.truncated_normal([num_cols_in*num_states_in,num_states_out], 0.1))
  # b_1 = tf.Variable(tf.truncated_normal([num_states_out],0.1))
  # y = tf.nn.softmax(tf.matmul(x_flat, W_1) + b_1)

  loss = utilities.calculate_cross_entropy(y, y_)

  train_step = utilities.train(utilities.Optimizer.GradientDescent, learning_rate, loss)
  # with tf.name_scope('train'):
  #   train_step = tf.train.AdamOptimizer(learning_rate).minimize(
  #       loss)

  accuracy = utilities.calculate_accuracy(y, y_)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(log_file_path + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(log_file_path + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = dh.get_training_data().next_batch(100)
      k = dropout_rate
    else:
      xs, ys = dh.get_testing_data().next_batch(1000)
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(args):
  # Try get user input   
  input_file, test_train_ratio, log_file_path, max_steps, learning_rate, dropout_rate = utilities.get_command_line_input(args)

  if tf.gfile.Exists(log_file_path):
    tf.gfile.DeleteRecursively(log_file_path)
  tf.gfile.MakeDirs(log_file_path)
  train(input_file, test_train_ratio, log_file_path, max_steps, learning_rate, dropout_rate)


if __name__ == '__main__':
  tf.app.run()