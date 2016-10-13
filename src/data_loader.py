"""This module provides a single class: DataLoader, which manages reading of raw data and formatting appropriately.
"""

import math
from random import sample, seed, shuffle

import numpy as np


class DataLoader(object):
    """A class which loads data from .txt files.

    It also formats data into 1-hot and splits data into training, testing, and validation sets.
    """

    def __init__(self, file_name_and_path, test_train_ratio, valid_train_ratio):
        """Creates a DataLoader

        It reads from the given file and splits the data into x, y1 and y2.

        Data members for the file path, test-train ratio, and validation-train ratio are initilised with the given values.

        All other data member varibles are initialised to None.

        Arguments:
            file_name_and_path: A string describing the file name (and relative path) of the .txt file to read.
            test_train_ratio: A float describing how much of the data to use for training and how much to use for testing.
            valid_train_ratio: A float describing how much of the training data to use for actual training and how much to use for validation.

        Returns:
            A DataLoader object.
        """
        self.__path = file_name_and_path
        self.__test_train_ratio = test_train_ratio
        self.__valid_train_ratio = valid_train_ratio

        # Read the data file, and get the numer of rows and collumns
        data = np.genfromtxt(file_name_and_path, dtype='intc', skip_header=1)
        self.__num_samples, num_rows = data.shape
        self.__num_loci = num_rows - 1

        # Split into the inputs and outputs
        self.__x = data[:, 0:(self.__num_loci)]
        self.__y_1 = data[:, self.__num_loci]

        # Generate the secondary output
        with open(file_name_and_path, 'r') as open_file:
            header = open_file.readline().strip()
            headers = header.split("\t")
            self.__y_2 = np.zeros(self.__x.shape)
            for (i, row) in enumerate(self.__y_2):
                for (j, _) in enumerate(row):
                    if self.__y_1[i] == 1 and headers[j][0] == 'M':
                        self.__y_2[i][j] = 1

        self.__x_1_hot = None
        self.__y_1_hot_1 = None
        self.__y_1_hot_2 = None

        self.__training_x = None
        self.__training_y_1 = None
        self.__training_y_2 = None

        self.__testing_x = None
        self.__testing_y_1 = None
        self.__testing_y_2 = None

        self.__validation_x = None
        self.__validation_y_1 = None
        self.__validation_y_2 = None


    def convert_data_to_1_hot(self):
        """Converts the x, y1, and y2 data read from the .txt file to a 1-hot encoding.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        # We want the data to be in a 1-hot format indicating whether the SNP is
        # double major, major-minor, or double minor
        # To do this we iterate through the data and for each element, we create a new 1-hot array
        self.__x_1_hot = np.zeros((self.__num_samples, self.__num_loci, 3))
        for (i, row) in enumerate(self.__x):
            for (j, cell) in enumerate(row):
                self.__x_1_hot[i][j][0] = int(cell == 0)
                self.__x_1_hot[i][j][1] = int(cell == 1)
                self.__x_1_hot[i][j][2] = int(cell == 2)

        # Labels need to also be 1-hot with index 0 is control and index 1 is case
        self.__y_1_hot_1 = np.zeros((self.__num_samples, 2))
        for (i, cell) in enumerate(self.__y_1):
            self.__y_1_hot_1[i][0] = int(cell == 0)
            self.__y_1_hot_1[i][1] = int(cell == 1)

        # Make the secondary output also 1 hot
        self.__y_1_hot_2 = np.zeros([self.__y_2.shape[0], self.__y_2.shape[1], 2])
        for (i, row) in enumerate(self.__y_2):
            for (j, cell) in enumerate(row):
                self.__y_1_hot_2[i][j][0] = int(cell == 1)
                self.__y_1_hot_2[i][j][1] = int(cell == 0)

    def split_data(self):
        """Splits the data set into three smaller data sets for training, testing and validation.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        seed(42)
        # We now want to split the data into training, validation and testing sets
        # We randomly choose a number of training/validation indices
        not_testing_indices = sample(range(self.__num_samples),
                                     int(math.ceil(self.__test_train_ratio*self.__num_samples)))
        # Now split those indices into training and validation
        training_indices = sample(not_testing_indices,
                                  int(math.ceil(self.__valid_train_ratio*len(not_testing_indices))))
        shuffle(training_indices) # does this actually do anything?
        self.__training_x = self.__x_1_hot[training_indices]
        self.__training_y_1 = self.__y_1_hot_1[training_indices]
        self.__training_y_2 = self.__y_1_hot_2[training_indices]

        validation_indices = [elem for elem in not_testing_indices if elem not in training_indices]
        shuffle(validation_indices)
        self.__validation_x = self.__x_1_hot[validation_indices]
        self.__validation_y_1 = self.__y_1_hot_1[validation_indices]
        self.__validation_y_2 = self.__y_1_hot_2[validation_indices]

        # All of the other indices are to become the testing set
        testing_indices = [elem for elem in range(self.__num_samples) if elem not in not_testing_indices]
        shuffle(testing_indices)
        self.__testing_x = self.__x_1_hot[testing_indices]
        self.__testing_y_1 = self.__y_1_hot_1[testing_indices]
        self.__testing_y_2 = self.__y_1_hot_2[testing_indices]

        # Because we are sampling randomly, for large data sets,
        # the ratio of case and controls in the data should remain 50% in both sets
        print("The number of training samples is %i with %i cases (%d percent)"
              %(len(self.__training_y_1), sum(self.__training_y_1[:, 1]), np.mean(self.__training_y_1[:, 1])*100))
        if testing_indices:
            print("The number of testing samples is %i with %i cases (%d percent)"
                  %(len(self.__testing_y_1), sum(self.__testing_y_1[:, 1]), np.mean(self.__testing_y_1[:, 1])*100))
        if validation_indices:
            print("The number of validation samples is %i with %i cases (%d percent)"
                  %(len(self.__validation_y_1), sum(self.__validation_y_1[:, 1]), np.mean(self.__validation_y_1[:, 1])*100))

    def get_testing_data(self):
        """Returns a protion of the 1-hot data to be used for testing. This portion is based on the test-train ratio that the DataLoader was initialised with.

        Arguments:
            Nothing.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        return (self.__testing_x, self.__testing_y_1, self.__testing_y_2)

    def get_training_data(self):
        """Returns a protion of the 1-hot data to be used for training. This portion is based on the test-train ratio that the DataLoader was initialised with.

        Arguments:
            Nothing.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        return (self.__training_x, self.__training_y_1, self.__training_y_2)

    def get_validation_data(self):
        """Returns a protion of the 1-hot data to be used for validation. This portion is based on the test-train ratio that the DataLoader was initialised with.

        Arguments:
            Nothing.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        return (self.__validation_x, self.__validation_y_1, self.__validation_y_2)

    def get_1_hot_data(self):
        """Returns all of the 1-hot encoded data.

        Arguments:
            Nothing.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        return (self.__x_1_hot, self.__y_1_hot_1, self.__y_1_hot_2)

    def get_data(self):
        """Returns all of the input data.
        Arguments:
            Nothing.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        return (self.__x, self.__y_1, self.__y_2)
