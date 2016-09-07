import numpy as np
from random import sample
import math

# Define some constants - perhaps these should be reworked so that they are input when running the script
NUM_LOCI = 100 
NUM_SAMPLES = 2000
IN_FILE = 'simple_EDM-1/simple_EDM-1_1.txt'
TEST_TRAIN_RATIO = 0.8

# Read the data file and get the labels which are the last collumn
labels = np.genfromtxt(IN_FILE, usecols=(NUM_LOCI), dtype='intc', skip_header=1, max_rows=NUM_SAMPLES)
# Read the data files and get the SNP information from the otherr collumns
data = np.genfromtxt(IN_FILE, usecols=(range(NUM_LOCI)), dtype='intc', skip_header=1, max_rows=NUM_SAMPLES)

# We want the data to be in a 1-hot format indicating whether the SNP is double major, major-minor, or double minor
# To do this we iterate through the data and for each element, we create a new 1-hot array
data_1_hot = np.zeros((NUM_SAMPLES, NUM_LOCI, 3))
for (i,row) in enumerate(data):
	for (j,cell) in enumerate(row):
		data_1_hot[i][j][0] = int(cell == 0)
		data_1_hot[i][j][1] = int(cell == 1)
		data_1_hot[i][j][2] = int(cell == 2)

# We now want to split the data into training and testing sets
# We randomly choose a number of indices in the data that will be used for training
training_indices = sample(range(NUM_SAMPLES),int(math.ceil(TEST_TRAIN_RATIO*NUM_SAMPLES)))
train_x = data_1_hot[training_indices]
train_y = labels[training_indices]

# All of the other indices are to become the testing set
testing_indices = [elem for elem in range(NUM_SAMPLES) if elem not in training_indices]
test_x = data_1_hot[testing_indices]
test_y = labels[testing_indices]

# Because we are sampling randomly, for large data sets, the ratio of case and controls in the data should remain 50% in both sets
print "The number of training samples is %i with %i cases (%d percent)"%(len(train_y), sum(train_y), np.mean(train_y)*100)
print "The number of testing samples is %i with %i cases (%d percent)"%(len(test_y), sum(test_y), np.mean(test_y)*100)

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