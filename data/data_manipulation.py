import numpy as np
from random import sample, shuffle
import math

# Define some constants - perhaps these should be reworked so that they are input when running the script
IN_FILE = 'small_EDM-1/small_EDM-1_1.txt'
TEST_TRAIN_RATIO = 0.8

# Read the data file, and get the numer of rows and collumns
data = np.genfromtxt(IN_FILE, dtype='intc', skip_header=1)
num_samples, num_loci = data.shape

# Split into the inputs and outputs
x = data[:,0:(num_loci-1)]
y = data[:,num_loci-1]

# We want the data to be in a 1-hot format indicating whether the SNP is double major, major-minor, or double minor
# To do this we iterate through the data and for each element, we create a new 1-hot array
x_1_hot = np.zeros((num_samples, num_loci, 3))
for (i,row) in enumerate(x):
	for (j,cell) in enumerate(row):
		x_1_hot[i][j][0] = int(cell == 0)
		x_1_hot[i][j][1] = int(cell == 1)
		x_1_hot[i][j][2] = int(cell == 2)

#Labels need to also be 1-hot with index 0 is control and index 1 is case
y_1_hot = np.zeros((num_samples, 2))
for (i,cell) in enumerate(y):
		y_1_hot[i][0] = int(cell == 0)
		y_1_hot[i][1] = int(cell == 1)

# We now want to split the data into training and testing sets
# We randomly choose a number of indices in the data that will be used for training
training_indices = sample(range(num_samples),int(math.ceil(TEST_TRAIN_RATIO*num_samples)))
shuffle(training_indices)
train_x = x_1_hot[training_indices]
train_y = y_1_hot[training_indices]

# All of the other indices are to become the testing set
testing_indices = [elem for elem in range(num_samples) if elem not in training_indices]
shuffle(testing_indices)
test_x = x_1_hot[testing_indices]
test_y = y_1_hot[testing_indices]

# Because we are sampling randomly, for large data sets, the ratio of case and controls in the data should remain 50% in both sets
print "The number of training samples is %i with %i cases (%d percent)"%(len(train_y), sum(train_y[:,1]), np.mean(train_y[:,1])*100)
print "The number of testing samples is %i with %i cases (%d percent)"%(len(test_y), sum(test_y[:,1]), np.mean(test_y[:,1])*100)

# Save the numpy arrays given above in a binary format (this is reqired because of the 3-D arrays for the X data which cannot be saved in a human readable format)
# Now the data can be read by the Tensorflow training script using the numpy.load() function: http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html
with open('train_x.npy','wb') as f:
	np.save(f, train_x)
with open('train_y.npy','wb') as f:	
	np.save(f, train_y)
with open('test_x.npy','wb') as f:
	np.save(f, test_x)
with open('test_y.npy','wb') as f:
	np.save(f, test_y)

'''
#Different data for linear model

# We now want to split the data into training and testing sets
# We randomly choose a number of indices in the data that will be used for training
train_snp_data = data[training_indices]
train_1_hot_labels = y_1_hot[training_indices]

# All of the other indices are to become the testing set
testing_indices = [elem for elem in range(num_samples) if elem not in training_indices]
test_snp_data = data[testing_indices]
test_1_hot_labels = y_1_hot[testing_indices]


# Because we are sampling randomly, for large data sets, the ratio of case and controls in the data should remain 50% in both sets
print "The number of training samples is %i with %i cases (%d percent)"%(len(train_y), sum(train_y), np.mean(train_y)*100)
print "The number of testing samples is %i with %i cases (%d percent)"%(len(test_y), sum(test_y), np.mean(test_y)*100)



#saving data for initial linear model
with open('train_snp_data.npy','wb') as f:
	np.save(f, train_snp_data)
with open('train_1_hot_labels.npy','wb') as f:
	np.save(f,train_1_hot_labels)
with open('test_snp_data.npy','wb') as f:
	np.save(f, test_snp_data)
with open('test_1_hot_labels.npy','wb') as f:
	np.save(f,test_1_hot_labels)
'''