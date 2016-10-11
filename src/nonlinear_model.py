"""This module is a nonlinear model to test for epistasis on a GAMETES dataset
"""

from __future__ import absolute_import, division, print_function

import sys
from math import sqrt

import tensorflow as tf

import data_holder
import utilities

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('file_in', '', 'data in file location')
flags.DEFINE_float('tt_ratio', 0.8, 'test:train ratio')
flags.DEFINE_integer('max_steps', 10000, 'maximum steps')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_integer('test_batch_size', 1000, 'testing batch size')
flags.DEFINE_string('log_dir', '/tmp/logs/runx', 'Directory for storing data')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout')
flags.DEFINE_string('model_dir', '/tmp/tf_models/', 'Directory for storing the saved models')
flags.DEFINE_bool('write_binary', True, 'Write the processed numpy array to a binary file.')
flags.DEFINE_bool('read_binary', True, 'Read a binary file rather than a text file.')

def train(dh, log_file_path, max_steps, train_batch_size, test_batch_size, learning_rate, dropout_rate, model_dir):

    # get the data dimmensions
    _, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
    _, num_states_out1 = dh.get_training_data().get_output1_shape()
    _, num_cols_out2, num_states_out2 = dh.get_training_data().get_output2_shape()

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
    y2 = tf.nn.softmax(utilities.reshape(utilities.fc_layer(dropped, 500, num_states_out2*num_cols_out2, layer_name='identity', act=tf.identity), [-1, num_cols_out2, num_states_out2]), name='softmax_2')

    loss1 = utilities.calculate_cross_entropy(y1, y1_, name_suffix='1')
    loss2 = utilities.calculate_cross_entropy(y2, y2_, name_suffix='2')

    train_step1 = utilities.train(learning_rate, loss1, training_method=utilities.Optimizer.Adam, name_suffix='1')
    train_step2 = utilities.train(learning_rate, loss2, training_method=utilities.Optimizer.Adam, name_suffix='2')

    accuracy1 = utilities.calculate_epi_accuracy(y1, y1_, name_suffix='1')
    # accuracy2 = utilities.calculate_epi_accuracy(y2, y2_, name_suffix='2')
    accuracy2 = utilities.calculate_snp_accuracy(y2, y2_, name_suffix='2')

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    def feed_dict(is_training, train_batch_size, test_batch_size):
        """ Make a TensorFlow feed_dict: maps data onto Tensor placeholders. 
        """
        if is_training:
            xs, y1s, y2s = dh.get_training_data().next_batch(train_batch_size)
            k = dropout_rate
        else:
            xs, y1s, y2s = dh.get_testing_data().next_batch(test_batch_size)
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
                summary, acc1, acc2, cost1, cost2 = sess.run([merged, accuracy1, accuracy2, loss1, loss2], feed_dict=feed_dict(False, train_batch_size, test_batch_size))
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s for output 1: %f' % (i, acc1))
                print('Accuracy at step %s for output 2: %f' % (i, acc2))
                print('Cost at step %s for output 1: %f' % (i, cost1))
                print('Cost at step %s for output 2: %f' % (i, cost2))

                # save the model every time a new best accuracy is reached
                if sqrt(cost1**2 + cost2**2) <= best_cost:
                    best_cost = sqrt(cost1**2 + cost2**2)
                    save_path = saver.save(sess, model_dir + 'nonlinear_model')
                    print("saving model at iteration %i" % i)

            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, train_batch_size, test_batch_size), options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _, _ = sess.run([merged, train_step1, train_step2], feed_dict=feed_dict(True, train_batch_size, test_batch_size))
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

        saver.restore(sess, save_path)

        best_acc1, best_acc2 = sess.run([accuracy1, accuracy2], feed_dict=feed_dict(False, train_batch_size, test_batch_size))
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

    # Import data.
    print("Loading data from: %s" % FLAGS.file_in)
    dh = data_holder.DataHolder()
    if not FLAGS.read_binary:
        try:
            dh.read_from_txt(FLAGS.file_in, FLAGS.tt_ratio, 1)
        except IOError as excep:
            print("Unable to read from text file: %s" % FLAGS.file_in)
            print(excep)
            sys.exit(2)
        if FLAGS.write_binary:
            try:
                dh.write_to_binary(FLAGS.file_in.replace('.txt', '.npz'))
            except IOError as excep:
                print("Unable to write to binary file")
                print(excep)
                sys.exit(2)
    else:
        try:
            dh.read_from_npz(FLAGS.file_in)
        except IOError as excep:
            print("Unable to read from binary file: %s" % FLAGS.file_in)
            print(excep)
            sys.exit(2)

    train(dh, FLAGS.log_dir, FLAGS.max_steps, FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.learning_rate, FLAGS.dropout, FLAGS.model_dir)

if __name__ == '__main__':
    tf.app.run()
