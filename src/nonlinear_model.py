
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import sys
import getopt

import data_holder
import utilities

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('file_in', '', 'data in file location')
flags.DEFINE_float('tt_ratio', 0.8, 'test:train ratio')
flags.DEFINE_integer('max_steps', 10000, 'maximum steps')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_string('log_dir', '/tmp/logs/runx', 'Directory for storing data')
flags.DEFINE_integer('max_steps', 10000, 'maximum steps ')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout')

def train(file_name_and_path, test_train_ratio, log_file_path, max_steps, batch_size, learning_rate, dropout_rate):

  print(file_name_and_path)
  # Import data
  dh = data_holder.DataHolder(file_name_and_path, test_train_ratio, 1)

  # get the data dimmensions
  num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
  num_rows_out1, num_states_out1 = dh.get_training_data().get_output1_shape()
  num_rows_out2, num_cols_out2, num_states_out2 = dh.get_training_data().get_output2_shape()

  tf.set_random_seed(42)
  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in], name='x-input')
    y1_ = tf.placeholder(tf.float32, [None, num_states_out1], name='y-input1')
    y2_ = tf.placeholder(tf.float32, [None, num_cols_out2, num_states_out2], name='y-input2')
  
  x_flat = utilities.reshape(x, [-1, num_cols_in*num_states_in])

  hidden1 = utilities.fc_layer(x_flat, num_cols_in*num_states_in, 1000, layer_name='hidden_1')
  hidden2 = utilities.fc_layer(hidden1, 1000, 500, layer_name='hidden_2')

  dropped, keep_prob = utilities.dropout(hidden2)

  y1 = utilities.fc_layer(dropped, 500, num_states_out1, layer_name='softmax_1', act=tf.nn.softmax)
  y2 = utilities.reshape(utilities.fc_layer(dropped, 500, num_states_out2*num_cols_out2, layer_name='softmax_2', act=tf.nn.softmax), [-1, num_cols_out2, num_states_out2])

  loss1 = utilities.calculate_cross_entropy(y1, y1_, name_suffix='1')
  loss2 = utilities.calculate_cross_entropy(y2, y2_, name_suffix='2')

  train_step1 = utilities.train(learning_rate, loss1, training_method=utilities.Optimizer.Adam, name_suffix='1')
  train_step2 = utilities.train(learning_rate, loss2, training_method=utilities.Optimizer.Adam, name_suffix='2')

  accuracy1 = utilities.calculate_accuracy(y1, y1_, name_suffix='1')
  accuracy2 = utilities.calculate_accuracy(y2, y2_, name_suffix='2')

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(log_file_path + '/train',
                                        sess.graph)
  test_writer = tf.train.SummaryWriter(log_file_path + '/test')
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
  def feed_dict(train, batch_size):
    """ Make a TensorFlow feed_dict: maps data onto Tensor placeholders. 
    """
    if train:
      xs, y1s, y2s = dh.get_training_data().next_batch(batch_size)
      k = dropout_rate
    else:
      xs, y1s, y2s = dh.get_testing_data().next_batch(1000)
      k = 1.0
    return {x: xs, y1_: y1s, y2_: y2s, keep_prob: k}

  for i in range(max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc1, acc2, cost1, cost2 = sess.run([merged, accuracy1, accuracy2, loss1, loss2], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s for output 1: %s' % (i, acc1))
      print('Accuracy at step %s for output 2: %s' % (i, acc2))
      print('Cost at step %s for output 2: %s' % (i, acc2))
      print('Cost at step %s for output 2: %s' % (i, acc2))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, batch_size),
                              options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, batch_size))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(args):
  # Try get user input   
  if not FLAGS.file_in:
    print("Please specify the input file using the '--file_in=' flag.")
    sys.exit(2)
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train(FLAGS.file_in, FLAGS.tt_ratio, FLAGS.log_dir, FLAGS.max_steps, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout)

if __name__ == '__main__':
  tf.app.run()