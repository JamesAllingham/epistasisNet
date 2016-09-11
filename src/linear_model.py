import tensorflow as tf
import numpy as np
import data_holder

#load input data
dh = data_holder.DataHolder('../data/small_EDM-1/small_EDM-1_1.txt')

# get the data dimmensions
num_rows_in, num_cols_in, num_states_in = dh.get_training_data().get_input_shape()
num_rows_out, num_states_out = dh.get_training_data().get_output_shape()

#set session type
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, num_cols_in, num_states_in])
x_flat = tf.reshape(x, [-1, num_cols_in*num_states_in]) # flatten x
W = tf.Variable(tf.zeros([num_cols_in*num_states_in,num_states_out]))
b = tf.Variable(tf.zeros([num_states_out]))
y = tf.nn.softmax(tf.matmul(x_flat, W) + b)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_states_out])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)  

# Train
tf.initialize_all_variables().run()

for i in range(2001):
  batch_xs, batch_ys = dh.get_training_data().next_batch(50)
  train_step.run({x: batch_xs, y_: batch_ys})
  if i % 100 == 0:
    print('Batch %i complete' % (i + 1,))

print('Training Complete')

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_xs, test_ys = dh.get_testing_data().next_batch(1000)
print(accuracy.eval({x: test_xs, y_: test_ys}))

# need to test different optimisers and different training rates
# need to impliment validation to prevent over fitting
# need to test different batch sizes a number of batches