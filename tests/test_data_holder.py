"""This module provides test cases for the DataHolder class."""

import sys
import unittest
from os import path, remove
import numpy as np

sys.path.append("../src/")
sys.path.append("src/")

import data_batcher
import data_holder


class BaseDataHolderTestCase(unittest.TestCase):
    """Provides set up and tear down functions which can be inherited by other test case classes for the DataHolder."""

    def setUp(self):
        """Sets up a DataHolder object initialised with a data set containing 100 samples.

        Also creates a temporary text file from which to read the data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        try:
            input_file = open("tmp.txt", 'w')

        except IOError as excep:
            print('Error writing temp file for testing')
            print(excep)
            sys.exit(2)

        input_file.write("N1\tN2\tN3\tN4\tN5\tN6\tN7\tM8\tM9\tc\n")
        for _ in range(50):
            input_file.write("0\t1\t2\t0\t1\t2\t0\t1\t2\t1\n")
        for _ in range(50):
            input_file.write("2\t1\t0\t2\t1\t0\t2\t1\t0\t0\n")
        input_file.close()

        self.dh = data_holder.DataHolder()
        self.dh.read_from_txt("tmp.txt", 0.8, 0.75)


    def tearDown(self):
        """Removes the temporary text file created in the set up.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        remove("tmp.txt")

class ReadTxtTestCase(BaseDataHolderTestCase):
    """Provides a test for correctly (without any errors) reading a text input file and creating three DataBachers:
    one for each of the training, testing, and validation data sets.

    Inherits from the BaseDataHolderTestCase.
    """
    def runTest(self):
        """Asserts that the DataHolder returns DataBatcher objects and a list object for header after loading the data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertIsInstance(self.dh.get_testing_data(), data_batcher.DataBatcher)
        self.assertIsInstance(self.dh.get_training_data(), data_batcher.DataBatcher)
        self.assertIsInstance(self.dh.get_validation_data(), data_batcher.DataBatcher)
        self.assertIsInstance(self.dh.get_header_data(), list)

class WriteBinaryTestCase(BaseDataHolderTestCase):
    """Provides a test for correctly (without any errors) writing a binary file containing the stored data.

    Inherits from the BaseDataHolderTestCase.
    """
    def runTest(self):
        """Asserts that the DataHolder correctly (without any errors) writes a binary file.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.dh.write_to_binary("tmp2")

        self.assertTrue(path.isfile("tmp2.npz"))

    def tearDown(self):
        """Removes the temporary binary file used for the test.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        remove("tmp2.npz")

class ReadBinaryTestCase(BaseDataHolderTestCase):
    """Provides a test for correctly (without any errors) reading a binary input file and creating three DataBachers:
    one for each of the training, testing, and validation data sets.

    Inherits from the BaseDataHolderTestCase.
    """

    def runTest(self):
        """Asserts that the DataHolder correctly (without any errors) reads from a written binary file and returns DataBatcher objects.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.dh.write_to_binary("tmp2")

        dh2 = data_holder.DataHolder()
        dh2.read_from_npz("tmp2.npz")

        self.assertIsInstance(dh2.get_testing_data(), data_batcher.DataBatcher)
        self.assertIsInstance(dh2.get_training_data(), data_batcher.DataBatcher)
        self.assertIsInstance(dh2.get_validation_data(), data_batcher.DataBatcher)

    def tearDown(self):
        """Removes the temporary binary file used for the test.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        remove("tmp2.npz")

if __name__ == "__main__":
    unittest.main()
