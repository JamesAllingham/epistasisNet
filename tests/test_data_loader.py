"""This module provides test cases for the DataLoader class."""

import sys
import unittest
from os import remove

sys.path.append("../src/")
sys.path.append("src/")

import data_loader

class BaseDataLoaderTestCase(unittest.TestCase):
    """Provides set up and tear down functions which can be inherited by other test case classes for the DataHolder."""

    def setUp(self):
        """Sets up a DataLoader object initialised with a data set containing 100 samples.

        Also creates a temporary text file from which to read the data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        try:
            open_file = open("tmp.txt", 'w')
        except IOError as excep:
            print('Error writing temp file for testing')
            print(excep)
            sys.exit(2)

        open_file.write("N1\tN2\tN3\tN4\tN5\tN6\tN7\tM8\tM9\tc\n")
        for _ in range(50):
            open_file.write("0\t1\t2\t0\t1\t2\t0\t1\t2\t1\n")
        for _ in range(50):
            open_file.write("2\t1\t0\t2\t1\t0\t2\t1\t0\t0\n")
        open_file.close()

        self.dl = data_loader.DataLoader("tmp.txt", 0.8, 0.75)

        self.dl.convert_data_to_1_hot()

        self.dl.split_data()

    def tearDown(self):
        """Removes the temporary text file created in the set up.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        remove("tmp.txt")

class GetInputDataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the input data has been correctly loded.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct input values.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, _, _ = self.dl.get_data()
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 8], 2)
        self.assertEqual(x[49, 2], 2)
        self.assertEqual(x[49, 8], 2)
        self.assertEqual(x[50, 0], 2)
        self.assertEqual(x[50, 8], 0)
        self.assertEqual(x[99, 1], 1)
        self.assertEqual(x[99, 8], 0)

class GetOutput1DataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the output 1 data has been correctly loded.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct output 1 values.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, y1, _ = self.dl.get_data()
        self.assertEqual(y1[0], 1)
        self.assertEqual(y1[49], 1)
        self.assertEqual(y1[50], 0)
        self.assertEqual(y1[99], 0)

class GetOutput2DataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the output 2 data has been correctly loded.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct output 2 values.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, _, y2 = self.dl.get_data()
        self.assertEqual(y2[0, 7], 1)
        self.assertEqual(y2[0, 8], 1)
        self.assertEqual(y2[49, 6], 0)
        self.assertEqual(y2[50, 7], 0)
        self.assertEqual(y2[99, 8], 0)

class GetInputOneHotdataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the input data has been correctly converted to a 1-hot encoding.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct input values in a 1-hot format.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, _, _ = self.dl.get_1_hot_data()
        self.assertEqual(x[0, 0, 0], 1)
        self.assertEqual(x[0, 0, 1], 0)
        self.assertEqual(x[0, 0, 2], 0)
        self.assertEqual(x[49, 2, 0], 0)
        self.assertEqual(x[49, 2, 1], 0)
        self.assertEqual(x[49, 2, 2], 1)
        self.assertEqual(x[99, 1, 0], 0)
        self.assertEqual(x[99, 1, 1], 1)
        self.assertEqual(x[99, 1, 2], 0)

class GetOutput1OneHotdataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the output 1 data has been correctly converted to a 1-hot encoding.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct output 1 values in a 1-hot format.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, y1, _ = self.dl.get_1_hot_data()
        self.assertEqual(y1[0, 0], 0)
        self.assertEqual(y1[0, 1], 1)
        self.assertEqual(y1[49, 0], 0)
        self.assertEqual(y1[49, 1], 1)
        self.assertEqual(y1[50, 0], 1)
        self.assertEqual(y1[50, 1], 0)
        self.assertEqual(y1[99, 0], 1)
        self.assertEqual(y1[99, 1], 0)

class GetOutput2OneHotdataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the output 2 data has been correctly converted to a 1-hot encoding.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns a numpy array with the correct output 2 values in a 1-hot format.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, _, y2 = self.dl.get_1_hot_data()
        self.assertEqual(y2[0, 8, 0], 1)
        self.assertEqual(y2[0, 8, 1], 0)
        self.assertEqual(y2[99, 8, 0], 0)
        self.assertEqual(y2[99, 8, 1], 1)

class GetValidationDataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the validation data is the correct size.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns three numpy arrays (one for each of the input and outputs) of the correct size for the validation set.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, y1, y2 = self.dl.get_validation_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y1), 20)
        self.assertEqual(len(y2), 20)

class GetTestingDataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the testing data is the correct size.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns three numpy arrays (one for each of the input and outputs) of the correct size for the testing set.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, y1, y2 = self.dl.get_testing_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y1), 20)
        self.assertEqual(len(y2), 20)

class GetTrainingDataTestCase(BaseDataLoaderTestCase):
    """Provides a test for checking that the training data is the correct size.

    Inherits from the BaseDataLoaderTestCase.
    """
    def runTest(self):
        """Asserts that the DataLoader returns three numpy arrays (one for each of the input and outputs) of the correct size for the training set.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, y1, y2 = self.dl.get_training_data()
        self.assertEqual(len(x), 60)
        self.assertEqual(len(y1), 60)
        self.assertEqual(len(y2), 60)

if __name__ == "__main__":
    unittest.main()
