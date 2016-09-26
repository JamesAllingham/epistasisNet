
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import sys
import getopt
from math import sqrt

import data_holder
import utilities

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('file_in', '', 'data in file location')
flags.DEFINE_float('tt_ratio', 0.8, 'test:train ratio')
flags.DEFINE_integer('max_steps', 10000, 'maximum steps')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_string('log_dir', '/tmp/logs/runx', 'Directory for storing data')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout')
flags.DEFINE_string('model_dir', '/tmp/tf_models/', 'Directory for storing the saved models')

def train(file_name_and_path, test_train_ratio, log_file_path, max_steps, batch_size, learning_rate, dropout_rate, model_dir):

  print("Loading data from: %s" % file_name_and_path)
  # Import data
  dh = data_holder.DataHolder(file_name_and_path, test_train_ratio, 1)

  # get the data dimmensions
  num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
  num_rows_out1, num_states_out1 = dh.get_training_data().get_output1_shape()
  num_rows_out2, num_cols_out2, num_states_out2 = dh.get_training_data().get_output2_shape()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in], name='x-input')
    y1_ = tf.placeholder(tf.float32, [None, num_states_out1], name='y-input1')
    y2_ = tf.placeholder(tf.float32, [None, num_cols_out2, num_states_out2], name='y-input2')

  print("x Shape: %s" % x.get_shape())
  print("y1_ Shape: %s" % y1_.get_shape())
  print("y2_ Shape: %s" % y2_.get_shape())

  x_4d = utilities.reshape(x, [-1, 100, 3, 1])  

  print("x_4d_ Shape: %s" %  x_4d.get_shape())

  conv1 = utilities.conv_layer(x_4d, [3,3,1,8], filter_padding='SAME', name_suffix='1')
  conv2 = utilities.conv_layer(conv1, [3,3,8,16], filter_padding='SAME', name_suffix='2')

  print("conv1 Shape: %s" % conv1.get_shape())
  print("conv2 Shape: %s" % conv2.get_shape())

  flatten = utilities.reshape(conv2, [-1, 3*100*16])

  print("flatten Shape: %s" % flatten.get_shape())
  
  hidden1 = utilities.fc_layer(flatten, 4800, 1000, layer_name='hidden1')
  hidden2 = utilities.fc_layer(hidden1, 1000, 500, layer_name='hidden2')

  print("hidden1 Shape: %s" % hidden1.get_shape())
  print("hidden2 Shape: %s" % hidden2.get_shape())

  dropped, keep_prob = utilities.dropout(hidden2)

  print("dropped Shape: %s" % dropped.get_shape())

  y1 = utilities.fc_layer(dropped, 500, num_states_out1, layer_name='softmax_1', act=tf.nn.softmax)

  print("y1 Shape: %s" % y1.get_shape())

  y2 = utilities.reshape(utilities.fc_layer(dropped, 500, num_states_out2*num_cols_out2, layer_name='softmax_2', act=tf.nn.softmax), [-1, num_cols_out2, num_states_out2])

  print("y2 Shape: %s" % y2.get_shape())

  loss1 = utilities.calculate_cross_entropy(y1, y1_, name_suffix='1')
  loss2 = utilities.calculate_cross_entropy(y2, y2_, name_suffix='2')

  train_step1 = utilities.train(learning_rate, loss1, training_method=utilities.Optimizer.Adam, name_suffix='1')
  train_step2 = utilities.train(learning_rate, loss2, training_method=utilities.Optimizer.Adam, name_suffix='2')

  accuracy1 = utilities.calculate_accuracy(y1, y1_, name_suffix='1')
  accuracy2 = utilities.calculate_accuracy(y2, y2_, name_suffix='2')

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()

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

  with tf.Session() as sess:    
    # Create a saver.
    saver = tf.train.Saver()

    train_writer = tf.train.SummaryWriter(log_file_path + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(log_file_path + '/test')

    sess.run(tf.initialize_all_variables())
    save_path = ''

    best_cost = float('inf')
    for i in range(max_steps):
      
      if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc1, acc2, cost1, cost2 = sess.run([merged, accuracy1, accuracy2, loss1, loss2], feed_dict=feed_dict(False, batch_size))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s for output 1: %f' % (i, acc1))
        print('Accuracy at step %s for output 2: %f' % (i, acc2))
        print('Cost at step %s for output 1: %f' % (i, cost1))
        print('Cost at step %s for output 2: %f' % (i, cost2))

        # save the model every time a new best accuracy is reached
        if sqrt(cost1**2 + cost2**2) <= best_cost:
          best_cost = sqrt(cost1**2 + cost2**2)
          save_path = saver.save(sess, model_dir + 'convolutional_model')
          print("saving model at iteration %i" % i)

      else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, batch_size), options=run_options, run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          print('Adding run metadata for', i)
        else:  # Record a summary
          summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, batch_size))
          train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()

    saver.restore(sess, save_path)

    best_acc1, best_acc2 = sess.run([accuracy1, accuracy2], feed_dict=feed_dict(False, batch_size))
    print("The best accuracies were %s and %s" % (best_acc1, best_acc2))


def main(args):
  tf.set_random_seed(42) 

  # Try get user input   
  if not FLAGS.file_in:
    print("Please specify the input file using the '--file_in=' flag.")
    sys.exit(2)
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)
  train(FLAGS.file_in, FLAGS.tt_ratio, FLAGS.log_dir, FLAGS.max_steps, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.dropout, FLAGS.model_dir)

if __name__ == '__main__':
  tf.app.run()