"""This module provides a single class: DataHolder, which manages reading of input files and storage of various data sets.
"""

import numpy as np

import data_batcher
import data_loader

class DataHolder(object):
    """A class to hold various data sets.

    The DataHolder contains the training, testing, and validation data sets.

    It provides functionality for reading from .txt and .npz (binary) files.
    It also provides functionality for writing .npz files for later use.
    Finaly it proves functionality for accessing the data sets described above.
    """

    def __init__(self):
        """Creates a DataLoader.

        All data mebers are initialised as None.

        Arguments:
            None.

        Returns:
            a DataLoader object.
        """
        self.__testing = None
        self.__training = None
        self.__validation = None
        self.__headers = None
        self.__data_loader = None

    def read_from_txt(self, file_name_and_path, test_train_ratio=0.8, valid_train_ratio=0.75):
        """Reads a data set from a .txt file, storing it as three data sets: training, testing, and validation.

        Arguments:
            file_name_and_path: A string describing the file name (and relative path) of the .txt file to read.
            test_train_ratio: A float describing how much of the data to use for training and how much to use for testing.
            valid_train_ratio: A float describing how much of the training data to use for actual training and how much to use for validation.

        Returns:
            Nothing.
        """
        self.__data_loader = data_loader.DataLoader(file_name_and_path, test_train_ratio, valid_train_ratio)
        self.__data_loader.convert_data_to_1_hot()
        self.__data_loader.split_data()
        training_x, training_y1, training_y2 = self.__data_loader.get_training_data()
        self.__training = data_batcher.DataBatcher(training_x, training_y1, training_y2)
        testing_x, testing_y1, testing_y2 = self.__data_loader.get_testing_data()
        self.__testing = data_batcher.DataBatcher(testing_x, testing_y1, testing_y2)
        validation_x, validation_y1, validation_y2 = self.__data_loader.get_validation_data()
        self.__validation = data_batcher.DataBatcher(validation_x, validation_y1, validation_y2)
        self.__headers = self.__data_loader.get_header_data()

    def write_to_binary(self, file_name_and_path):
        """Writes a processed .txt file to a .npz (binary) file.

        Arguments:
            file_name_and_path: A string describing the file name (and relative path) of the .npz file to write.

        Returns:
            Nothing.
        """
        training_x, training_y1, training_y2 = self.__data_loader.get_training_data()
        testing_x, testing_y1, testing_y2 = self.__data_loader.get_testing_data()
        validation_x, validation_y1, validation_y2 = self.__data_loader.get_validation_data()
        headers = self.__data_loader.get_header_data()
        np.savez(file_name_and_path,
                 testing_x=testing_x, testing_y1=testing_y1, testing_y2=testing_y2,
                 training_x=training_x, training_y1=training_y1, training_y2=training_y2,
                 validation_x=validation_x, validation_y1=validation_y1, validation_y2=validation_y2,
                 headers=headers)

    def read_from_npz(self, file_name_and_path):
        """Reads a data set from a .npz (binary) file, storing it as four data sets: training, testing, validation and headers.

        Note that unlike the read_from_txt function the varios data set size ratios do not need to be set. They are stored in the binary.

        Arguments:
            file_name_and_path: A string describing the file name (and relative path) of the .npz file to read.

        Returns:
            Nothing.
        """
        npzfile = np.load(file_name_and_path)
        self.__training = data_batcher.DataBatcher(npzfile['training_x'], npzfile['training_y1'], npzfile['training_y2'])
        self.__testing = data_batcher.DataBatcher(npzfile['testing_x'], npzfile['testing_y1'], npzfile['testing_y2'])
        self.__validation = data_batcher.DataBatcher(npzfile['validation_x'], npzfile['validation_y1'], npzfile['validation_y2'])
        self.__headers = npzfile['headers']

    def get_testing_data(self):
        """Gets the testing data being stored.

        Arguments:
            None

        Returns:
            A DataBatcher object containing the testing data.
        """
        return self.__testing

    def get_training_data(self):
        """Gets the training data being stored.

        Arguments:
            None

        Returns:
            A DataBatcher object containing the training data.
        """
        return self.__training

    def get_validation_data(self):
        """Gets the validation data being stored.

        Arguments:
            None

        Returns:
            A DataBatcher object containing the validation data.
        """
        return self.__validation

    def get_header_data(self):
        """Gets the header data being stored.

        Arguments:
            None

        Returns:
            A numpy array containing the header names
        """
        return self.__headers
