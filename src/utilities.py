import tensorflow as tf
import numpy as np
import sys
import getopt

# File Management Utilities
def get_command_line_input(args):
  try:                                
    opts, args = getopt.getopt(args[1:], "f:r:hl:", ["file=", 'ratio=', "help", 'logfile=']) 
  except getopt.GetoptError:           
    print("The allowed arguments are '-h' for help, '-r' to specify the test-train ratio, and '-f' to specify the input file.")                         
    sys.exit(2) 

  test_train_ratio = 0.2
  log_file_path = '/tmp/logs/'
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
    elif opt in ("-l", "--logfile"):
      log_file_path += arg
  if not file_arg_given:
    print("Please specify the input file using the '-f' flag.")
    sys.exit(2)

# # Variable utilities
def tn_weight_variable(shape, stddev_value):
  initial = tf.truncated_normal(shape=shape, stddev=stddev_value)
  return tf.Variable(initial)

def zeros_weight_variable(shape):
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial)

def bias_variable(shape):
  # Create a bias variable with appropriate initialization - slightly positive to avoid deadness.
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# # nn utilities

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
    dropped = tf.nn.dropout(x, keep_prob)
  return dropped, keep_prob

# loss function utilities

def calculate_cross_entropy(y, y_):
  # computes cross entropy between trained y and label y_
  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross_entropy', cross_entropy)
  return cross_entropy

# training utilities

class Optimizer:
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

def calculate_accuracy (y, y_):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
  return accuracy