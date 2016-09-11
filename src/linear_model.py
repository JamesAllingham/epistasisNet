import tensorflow as tf
import numpy as np
import data_holder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#load input data
dh = data_holder.DataHolder('../data/medium_EDM-1/medium_EDM-1_1.txt', 0.2, 0.75)

# get the data dimmensions
num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
num_rows_out, num_states_out = dh.get_training_data().get_output_shape()

#set session type
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in])
x_flat = tf.reshape(x, [-1, num_cols_in*num_states_in]) # flatten x
W_1 = tf.Variable(tf.truncated_normal([num_cols_in*num_states_in,num_states_out], 0.1))
b_1 = tf.Variable(tf.truncated_normal([num_states_out],0.1))
y = tf.nn.softmax(tf.matmul(x_flat, W_1) + b_1)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_states_out])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)  

# Define how to calculate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
tf.initialize_all_variables().run()

for i in range(11):
  batch_xs, batch_ys = dh.get_training_data().next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})
  test_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
  valid_xs, valid_ys = dh.get_validation_data().next_batch(100)
  valid_accuracy = accuracy.eval({x: valid_xs, y_: valid_ys})
  if (i % 1) == 0:
    print('Batch %i complete, test accuracy = %f, validation accuracy = %f' % (i, test_accuracy, valid_accuracy))
  if valid_accuracy + 0.1 < test_accuracy:
    print('Stoping training with test accuracy = %f and validation accuracy = %f' % (test_accuracy, valid_accuracy))
    break

print('Training Complete')

# Test trained model
test_xs, test_ys = dh.get_testing_data().next_batch(1200)
print(accuracy.eval({x: test_xs, y_: test_ys}))