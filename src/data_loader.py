import numpy as np
from random import sample, shuffle
import math

class DataLoader:

    def __init__(self, file_name_and_path, test_train_ratio, train_valid_ratio):
        self.__path = file_name_and_path
        self.__test_train_ratio = test_train_ratio
        self.__train_valid_ratio = train_valid_ratio

        # Read the data file, and get the numer of rows and collumns
        data = np.genfromtxt(file_name_and_path, dtype='intc', skip_header=1)
        self.__num_samples, num_rows = data.shape
        self.__num_loci = num_rows - 1

        # Split into the inputs and outputs
        self.__x = data[:,0:(self.__num_loci)]
        self.__y = data[:,self.__num_loci]

    def convert_data_to_1_hot(self):        
        # We want the data to be in a 1-hot format indicating whether the SNP is double major, major-minor, or double minor
        # To do this we iterate through the data and for each element, we create a new 1-hot array
        self.__x_1_hot = np.zeros((self.__num_samples, self.__num_loci, 3))
        for (i,row) in enumerate(self.__x):
            for (j,cell) in enumerate(row):
                self.__x_1_hot[i][j][0] = int(cell == 0)
                self.__x_1_hot[i][j][1] = int(cell == 1)
                self.__x_1_hot[i][j][2] = int(cell == 2)

        #Labels need to also be 1-hot with index 0 is control and index 1 is case
        self.__y_1_hot = np.zeros((self.__num_samples, 2))
        for (i,cell) in enumerate(self.__y):
                self.__y_1_hot[i][0] = int(cell == 0)
                self.__y_1_hot[i][1] = int(cell == 1)

    def split_data(self):
        # We now want to split the data into training, validation and testing sets
        # We randomly choose a number of indices in the data that will be used for training/ validation
        not_testing_indices = sample(range(self.__num_samples), int(math.ceil(self.__test_train_ratio*self.__num_samples)))
        # Now split those indices into training and validation
        training_indices = sample(not_testing_indices, int(math.ceil(self.__train_valid_ratio*len(not_testing_indices))))
        shuffle(training_indices) # does this actually do anything?
        self.__testing_x = self.__x_1_hot[training_indices]
        self.__testing_y = self.__y_1_hot[training_indices]

        validation_indices = [elem for elem in not_testing_indices if elem not in training_indices]
        shuffle(validation_indices)
        self.__validation_x = self.__x_1_hot[validation_indices]
        self.__validation_y = self.__y_1_hot[validation_indices]

        # All of the other indices are to become the testing set
        testing_indices = [elem for elem in range(self.__num_samples) if elem not in not_testing_indices]
        shuffle(testing_indices)
        self.__training_x = self.__x_1_hot[testing_indices]
        self.__training_y = self.__y_1_hot[testing_indices]

        # Because we are sampling randomly, for large data sets, the ratio of case and controls in the data should remain 50% in both sets
        print "The number of training samples is %i with %i cases (%d percent)"%(len(self.__training_y), sum(self.__training_y[:,1]), np.mean(self.__training_y[:,1])*100)
        print "The number of testing samples is %i with %i cases (%d percent)"%(len(self.__testing_y), sum(self.__testing_y[:,1]), np.mean(self.__testing_y[:,1])*100)
        print "The number of validation samples is %i with %i cases (%d percent)"%(len(self.__validation_y), sum(self.__validation_y[:,1]), np.mean(self.__validation_y[:,1])*100)

    def get_testing_data(self):
        return (self.__testing_x, self.__testing_y)

    def get_training_data(self):
        return (self.__training_x, self.__training_y)
    
    def get_validation_data(self):
        return (self.__validation_x, self.__validation_y)

    def get_1_hot_data(self):
        return (self.__x_1_hot, self.__y_1_hot)

    def get_data(self):
        return (self.__x, self.__y)