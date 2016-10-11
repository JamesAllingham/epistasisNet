"""This module provides test cases for the DataBatcher class."""
import sys
import unittest

import numpy as np

sys.path.append("../src/")
sys.path.append("src/")

import data_batcher

class BaseDataBatcherTestCase(unittest.TestCase):
    """Provides a set up function which can be inherited by other test case classes for the DataBatcher."""

    def setUp(self):
        """Sets up a DataBatcher object initialised with a data set containing 10 samples.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x = np.array([range(10), range(10), range(10)]).T
        y1 = np.array([range(10)]).T
        y2 = np.array([range(10), range(10)]).T
        self.db = data_batcher.DataBatcher(x, y1, y2)

class GetInputShapeTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct input shape.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct size for its input data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(self.db.get_input_shape(), (10, 3))

class GetOutput1ShapeTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output 1 shape.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct size for its output 1 data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(self.db.get_output1_shape(), (10, 1))

class GetOutput2ShapeTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output 2 shape.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct size for its output 2 data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(self.db.get_output2_shape(), (10, 2))

class FirstBatchGivesCorrectInputsTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output 2 shape.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct first batch of input data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(len(x), 5)
        self.assertEqual(x[0, 1], 0)
        self.assertEqual(x[4, 0], 4)
        self.assertEqual(x[4, 2], 4)

class FirstBatchGivesCorrectOutputs1TestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct first output 1 batch.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct first batch of output 1 data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(len(y1), 5)
        self.assertEqual(y1[0, 0], 0)
        self.assertEqual(y1[4, 0], 4)

class FirstBatchGivesCorrectOutputs2TestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct first output 2 batch.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct first batch of output 2 data.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(len(y2), 5)
        self.assertEqual(y2[0, 0], 0)
        self.assertEqual(y2[4, 1], 4)

class InputBatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct input batch after rolling over an epoch.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct input data after an epoch roll over.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0, 0], 0)
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0, 0], 5)
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0, 0], 0)

class Output1BatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output 1 batch after rolling over an epoch.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct output 1 data after an epoch roll over.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0, 0], 0)
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0, 0], 5)
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0, 0], 0)

class Output2BatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output2 batch after rolling over an epoch.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct output 2 data after an epoch roll over.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0, 0], 0)
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0, 0], 5)
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0, 0], 0)

class IfBatchSizeIsNoneReturnAllDataTestCase(BaseDataBatcherTestCase):
    """Provides a test for returning the correct output and input batches if the whole data set it requested.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct input, output 1, and output 2 data if the whole data set is requested.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        _, _, y2 = self.db.next_batch(None)
        self.assertEqual(y2.shape[0], 10)
        _, y1, _ = self.db.next_batch(None)
        self.assertEqual(y1.shape[0], 10)
        x, _, _ = self.db.next_batch(None)
        self.assertEqual(x.shape[0], 10)

class EpochNumberIsCorrectlyReturnedWhenTakingPartialBatches(BaseDataBatcherTestCase):
    """Provides a test for returning the correct epoch number when partial batches cumulatively roll over.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct epoch number when partial batches cause a roll over of the epoch.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(2, self.db.get_num_epochs())

class EpochNumberIsCorrectlyReturnedWhenTakingFullBatches(BaseDataBatcherTestCase):
    """Provides a test for returning the correct epoch number when a full batch is requested.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct epoch number when full batches cause a roll over of the epoch.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(2, self.db.get_num_epochs())

class EpochNumberIsCorrectlyReturnedWhenTakingMixedBatches(BaseDataBatcherTestCase):
    """Provides a test for returning the correct epoch number when partial batches cumulatively roll over and full batches are requested.

    Inherits from the BaseDataBatcherTestCase.
    """
    def runTest(self):
        """Asserts that the DataBatcher returns the correct epoch number when partial and full batches cause a roll over of the epoch.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(2, self.db.get_num_epochs())

if __name__ == "__main__":
    unittest.main()
