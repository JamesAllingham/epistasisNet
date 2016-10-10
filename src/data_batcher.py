"""This module provides a single class: DataLoader, which manages reading of raw data and formatting is appropriately.
"""

import sys

import numpy as np


class DataBatcher(object):
    """A class which batches data.

    After being initialised with a large amount of data this class provides functionality for accessing small batches of the data in order.
    """

    def __init__(self, x, y1, y2):
        """Creates a DataBatcher.

        Arguments:
            x: a numpy array containing all of the input data.
            y1: a numpy array containing all of the output 1 data.
            y2: a numpy array containing all of the output 2 data.

        Returns:
            A DataBatcher object.
        """
        self.__x = x
        self.__y1 = y1
        self.__y2 = y2
        self.__batch_cursor = 0
        self.__data_size = self.__x.shape[0]
        self.__num_epochs = 0

        if self.__data_size != self.__y1.shape[0]:
            print("The input and output sets must have the same number of entries")
            sys.exit(2)

        if self.__y2.shape[0] != self.__y1.shape[0]:
            print("The output sets must have the same number of entries")
            sys.exit(2)

    def next_batch(self, batch_size):
        """Returns the next batch of the data.

        Arguments:
            batch_size: an int describing the number of samples to include in the batch.

        Returns:
            A triple containing (x, y1, y2). Each element is a numpy array.
        """
        # If the caller wants all of the data simply return the whole data set as a triple
        if batch_size is None:
            self.__num_epochs += 1
            return (self.__x, self.__y1, self.__y2)

        if batch_size > self.__data_size:
            print("Please specify a batch size less than the number of entries in the data set")
            sys.exit(2)

        if batch_size + self.__batch_cursor < self.__data_size:
            # If the batch size is less than the number of entries left in the data:
            # Take the next batch size number of elements and move the cursor forwards.
            x_batch = self.__x[self.__batch_cursor:batch_size + self.__batch_cursor]
            y1_batch = self.__y1[self.__batch_cursor:batch_size + self.__batch_cursor]
            y2_batch = self.__y2[self.__batch_cursor:batch_size + self.__batch_cursor]
            self.__batch_cursor = self.__batch_cursor + batch_size
        else:
            # If there is not enough data left then take the remaining data from the end and start again at the begining.
            x_batch = self.__x[self.__batch_cursor:]
            y1_batch = self.__y1[self.__batch_cursor:]
            y2_batch = self.__y2[self.__batch_cursor:]
            number_still_required = batch_size - (self.__data_size - self.__batch_cursor)
            x_batch = np.concatenate((x_batch, self.__x[0:number_still_required]))
            y1_batch = np.concatenate((y1_batch, self.__y1[0:number_still_required]))
            y2_batch = np.concatenate((y2_batch, self.__y2[0:number_still_required]))
            self.__batch_cursor = number_still_required
            self.__num_epochs += 1

        return (x_batch, y1_batch, y2_batch)

    def get_input_shape(self):
        """ Returns the tensor shape of the input data.

        Arguments:
            Nothing.

        Returns:
            An n-tuple containing the integer dimension sizes of the input data.
        """
        return self.__x.shape

    def get_output1_shape(self):
        """ Returns the tensor shape of the output 1 data.

        Arguments:
            Nothing.

        Returns:
            An n-tuple containing the integer dimension sizes of the output 1 data.
        """
        return self.__y1.shape

    def get_output2_shape(self):
        """ Returns the tensor shape of the output 2 data.

        Arguments:
            Nothing.

        Returns:
            An n-tuple containing the integer dimension sizes of the output 2 data.
        """
        return self.__y2.shape

    def get_num_epochs(self):
        """Returns the number of epochs of data that have been batched.

        Arguments:
            Nothing.

        Returns:
            An integer number of epochs.
        """
        return self.__num_epochs
