from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data_holder
import sys
import getopt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/logs', 'Summaries directory')


def train(file_name_and_path, test_train_ratio):
  # Import data
  dh = data_holder.DataHolder(file_name_and_path, test_train_ratio, 1)

  # get the data dimmensions
  num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
  num_rows_out, num_states_out = dh.get_training_data().get_output_shape()

  sess = tf.InteractiveSession()

  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_states_out], name='y-input')

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
        weights = weight_variable([input_dim, output_dim])
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
  
  with tf.name_scope('reshape'):
    x_flat = tf.reshape(x, [-1, num_cols_in*num_states_in])

  hidden1 = nn_layer(x_flat, num_cols_in*num_states_in, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  #y = nn_layer(dropped, 500, num_states_out, 'layer2', act=tf.nn.softmax)
  W_1 = tf.Variable(tf.truncated_normal([num_cols_in*num_states_in,num_states_out], 0.1))
  b_1 = tf.Variable(tf.truncated_normal([num_states_out],0.1))
  y = tf.nn.softmax(tf.matmul(x_flat, W_1) + b_1)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = dh.get_training_data().next_batch(100)
      k = FLAGS.dropout
    else:
      xs, ys = dh.get_testing_data().next_batch(1000)
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
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
  try:                                
    opts, args = getopt.getopt(args[1:], "f:r:h", ["file=", "help"]) 
  except getopt.GetoptError:           
    print("The allowed arguments are '-h' for help, '-r' to specify the test-train ratio, and '-f' to specify the input file.")                         
    sys.exit(2) 

  test_train_ratio = 0.2
  file_arg_given = False
  for opt, arg in opts:                
    if opt in ("-h", "--help"):      
      print("The allowed arguments are '-h' for help, '-r' to specify the test-train ratio, and '-f' to specify the input file.")                   
      sys.exit(2)                     
    elif opt in ("-f", "--file"): 
      input_file = arg
      file_arg_given = True  
    elif opt in ("-r", "--ratio"):                
			test_train_ratio = float(arg)     
    if not file_arg_given:
      print("Please specify the input file using the '-f' flag.")
      sys.exit(2)

  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train(input_file, test_train_ratio)


if __name__ == '__main__':
  tf.app.run()