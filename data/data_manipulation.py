import numpy as np
from random import sample
import math

NUM_LOCI = 100
NUM_SAMPLES = 2000
IN_FILE = 'simple_EDM-1/simple_EDM-1_1.txt'
TEST_TRAIN_RATIO = 0.8

labels = np.genfromtxt(IN_FILE, usecols=(NUM_LOCI), dtype='intc', skip_header=1, max_rows=NUM_SAMPLES)
data = np.genfromtxt(IN_FILE, usecols=(range(NUM_LOCI)), dtype='intc', skip_header=1, max_rows=NUM_SAMPLES)

data_1_hot = np.zeros((NUM_SAMPLES, NUM_LOCI, 3))
for (i,row) in enumerate(data):
	for (j,cell) in enumerate(row):
		data_1_hot[i][j][0] = int(cell == 0)
		data_1_hot[i][j][1] = int(cell == 1)
		data_1_hot[i][j][2] = int(cell == 2)

training_indices = sample(range(NUM_SAMPLES),int(math.ceil(TEST_TRAIN_RATIO*NUM_SAMPLES)))
train_x = data_1_hot[training_indices]
train_y = labels[training_indices]

testing_indices = [elem for elem in range(NUM_SAMPLES) if elem not in training_indices]
test_x = data_1_hot[testing_indices]
test_y = labels[testing_indices]

print "The number of training samples is %i with %i cases (%d percent)"%(len(train_y), sum(train_y), np.mean(train_y)*100)
print "The number of testing samples is %i with %i cases (%d percent)"%(len(test_y), sum(test_y), np.mean(test_y)*100)

with open('train_x.npy','wb') as f:
	np.save(f, train_x)
with open('train_y.npy','wb') as f:	
	np.save(f, train_y)
with open('test_x.npy','wb') as f:
	np.save(f, test_x)
with open('test_y.npy','wb') as f:
	np.save(f, test_y)