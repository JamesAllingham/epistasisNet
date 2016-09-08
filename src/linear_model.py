import tensorflow as tf
import numpy as np

#load input data
training_data = np.load('../data/train_x.npy')
training_labels = np.load('../data/train_y.npy')
test_data = np.load('../data/test_x.npy')
test_labels = np.load('../data/test_y.npy')

#training_data = tf.to_float(training_data)
#training_labels = tf.to_float(training_labels)

#set session type
sess = tf.InteractiveSession()

#set up variable environment
data_to_train = tf.constant(training_data, name='data_to_train')
labels_to_train = tf.constant(training_labels, name='labels_to_train')


x = tf.placeholder(tf.float32, [None, 10, 3]) #100 snps per person
x_flat = tf.reshape(x, [-1, 10*3]) # flatten x
W = tf.Variable(tf.zeros([30,2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x_flat, W) + b)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()

'''
for i in range(1000):
  batch_xs = data_to_train.next_batch(100)
  batch_ys = labels_to_train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})
  print('Batch 1 complete')
'''

train_step.run({x: training_data, y_: training_labels})
print('Training Complete')

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: test_data, y_: test_labels}))