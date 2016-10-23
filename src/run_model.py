"""This module trains a TensorFlow model.
"""
from __future__ import absolute_import, division, print_function

import sys

import tensorflow as tf
from tensorflow.python.client import timeline

import data_holder as dh
import utilities

# import the various models which can be run
import recurrent_model
import scaling_model
import pool_conv_model
import convolutional_model
import nonlinear_model
import linear_model

APP_FLAGS = tf.app.flags
FLAGS = APP_FLAGS.FLAGS
APP_FLAGS.DEFINE_string('file_in', '', 'data in file location')
APP_FLAGS.DEFINE_float('tt_ratio', 0.8, 'test:train ratio')
APP_FLAGS.DEFINE_integer('max_steps', 1000, 'maximum steps')
APP_FLAGS.DEFINE_integer('train_batch_size', 100, 'training batch size')
APP_FLAGS.DEFINE_integer('test_batch_size', 1000, 'testing batch size')
APP_FLAGS.DEFINE_string('log_dir', '/tmp/logs/runx', 'Directory for storing data')
APP_FLAGS.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
APP_FLAGS.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout')
APP_FLAGS.DEFINE_string('model_dir', '/tmp/tf_models/', 'Directory for storing the saved models')
APP_FLAGS.DEFINE_bool('write_binary', True, 'Write the processed numpy array to a binary file.')
APP_FLAGS.DEFINE_bool('read_binary', True, 'Read a binary file rather than a text file.')
APP_FLAGS.DEFINE_bool('save_model', True, 'Save the best model asa the training progresses.')

def train_model(data_holder):
    """A function that builds and trains the model.

    Arguments:
            data_holder: a DataHolder object containing the data.

        Returns:
            Nothing.
    """

    # get the data dimmensions
    _, num_cols_in, num_states_in = data_holder.get_training_data().get_input_shape()
    _, num_states_out1 = data_holder.get_training_data().get_output1_shape()
    _, num_cols_out2, num_states_out2 = data_holder.get_training_data().get_output2_shape()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in], name='x-input')
        y1_ = tf.placeholder(tf.float32, [None, num_states_out1], name='y-input1')
        y2_ = tf.placeholder(tf.float32, [None, num_cols_out2, num_states_out2], name='y-input2')

    print("x Shape: %s" % x.get_shape())
    print("y1_ Shape: %s" % y1_.get_shape())
    print("y2_ Shape: %s" % y2_.get_shape())

    model = pool_conv_model.PoolConvModel(x, y1_, y2_, FLAGS.learning_rate)

    keep_prob = model.get_keep_prob()
    loss1, loss2 = model.get_losses()
    accuracy1, accuracy2 = model.get_accuracies()
    epi_snps, count = model.get_snp_predictions()
    merged = model.get_merged()
    train_step = model.get_train_step()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(training, batch_size):
        """ Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
        """
        if training:
            xs, y1s, y2s = data_holder.get_training_data().next_batch(batch_size)
            k = FLAGS.dropout
        else:
            xs, y1s, y2s = data_holder.get_testing_data().next_batch(batch_size)
            k = 1.0
        return {x: xs, y1_: y1s, y2_: y2s, keep_prob: k}

    # config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session() as sess:
        # Create a saver this will be used to save the current best model.
        # If the model starts to over fit then it can be restored to the previous best version.
        saver = tf.train.Saver()

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')

        sess.run(tf.initialize_all_variables())
        save_path = ''

        best_cost = float('inf')
        for i in range(FLAGS.max_steps):

            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc1, acc2, cost1, cost2 = sess.run([merged, accuracy1, accuracy2, loss1, loss2], feed_dict=feed_dict(False, FLAGS.test_batch_size))
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s for output 1: %f' % (i, acc1))
                print('Accuracy at step %s for output 2: %f' % (i, acc2))
                print('Cost at step %s for output 1: %f' % (i, cost1))
                print('Cost at step %s for output 2: %f' % (i, cost2))

                # save the model every time a new best accuracy is reached
                if cost1 + cost2 <= best_cost and FLAGS.save_model:
                    best_cost = cost1 + cost2
                    save_path = saver.save(sess, FLAGS.model_dir + 'model')
                    print("saving model at iteration %i" % i)

            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, FLAGS.train_batch_size), options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)

                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, FLAGS.train_batch_size))
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

        if FLAGS.save_model:
            saver.restore(sess, save_path)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        best_acc1, best_acc2, epi_snp_locations, epi_snp_counts = sess.run([accuracy1, accuracy2, epi_snps, count], feed_dict=feed_dict(False, FLAGS.test_batch_size), options=run_options, run_metadata=run_metadata)
        epi_snp_names = utilities.get_snp_headers(epi_snp_locations, data_holder.get_header_data())
        print("The best accuracies were %s and %s" % (best_acc1, best_acc2))
        print("The SNPs predicted to cause epistasis are %s" % epi_snp_names)
        print("Their respective occurrance counts are %s" % epi_snp_counts)
        
        tl = timeline.Timeline(run_metadata.step_stats)
        # print(tl.generate_chrome_trace_format(show_memory=True))
        trace_file = tf.gfile.Open(name='../data/timeline', mode='w')
        trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

def main(args):
    """The main function which invokes the model_training function after reading the input data file.

    Arguments:
        args: command line arguments for the script.

    Returns:
        Nothing.
    """
    # Set the random seed so that results will be reproducable.
    tf.set_random_seed(42)

    # Try get user input.
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
    data_holder = dh.DataHolder()
    if not FLAGS.read_binary:
        try:
            data_holder.read_from_txt(FLAGS.file_in, FLAGS.tt_ratio, 1)
        except IOError as excep:
            print("Unable to read from text file: %s" % FLAGS.file_in)
            print(excep)
            sys.exit(2)
        if FLAGS.write_binary:
            try:
                data_holder.write_to_binary(FLAGS.file_in.replace('.txt', '.npz'))
            except IOError as excep:
                print("Unable to write to binary file")
                print(excep)
                sys.exit(2)
    else:
        try:
            data_holder.read_from_npz(FLAGS.file_in)
        except IOError as excep:
            print("Unable to read from binary file: %s" % FLAGS.file_in)
            print(excep)
            sys.exit(2)

    # Use the data to train a neural network.
    train_model(data_holder)

if __name__ == '__main__':
    tf.app.run()
