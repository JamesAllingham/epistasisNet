import unittest
import sys
sys.path.append("../src/")
sys.path.append("src/")
import data_batcher
import numpy as np

class BaseDataBatcherTestCase(unittest.TestCase):
    
    def setUp(self):
        x = np.array([range(10), range(10), range(10)]).T
        y1 = np.array([range(10)]).T
        y2 = np.array([range(10), range(10)]).T
        self.db = data_batcher.DataBatcher(x, y1, y2)

class GetInputShapeTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(self.db.get_input_shape(), (10,3))

class GetOutput1ShapeTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(self.db.get_output1_shape(), (10,1))

class GetOutput2ShapeTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(self.db.get_output2_shape(), (10,2))

class FirstBatchGivesCorrectInputsTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(len(x), 5)
        self.assertEqual(x[0,1], 0)
        self.assertEqual(x[4,0], 4)
        self.assertEqual(x[4,2], 4)

class FirstBatchGivesCorrectOutputs1TestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(len(y1), 5)
        self.assertEqual(y1[0,0], 0)
        self.assertEqual(y1[4,0], 4)

class FirstBatchGivesCorrectOutputs2TestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(len(y2), 5)
        self.assertEqual(y2[0,0], 0)
        self.assertEqual(y2[4,1], 4)

class InputBatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 0)
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 5)
        x, _, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 0)

class Output1BatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0,0], 0)
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0,0], 5)
        _, y1, _ = self.db.next_batch(5)
        self.assertEqual(y1[0,0], 0)

class Output2BatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0,0], 0)
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0,0], 5)
        _, _, y2 = self.db.next_batch(5)
        self.assertEqual(y2[0,0], 0)

class IfBatchSizeIsNoneReturnAllDataTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, _, y2 = self.db.next_batch(None)
        self.assertEqual(y2.shape[0], 10)
        _, y1, _ = self.db.next_batch(None)
        self.assertEqual(y1.shape[0], 10)
        x, _, _ = self.db.next_batch(None)
        self.assertEqual(x.shape[0], 10)

class EpochNumberIsCorrectlyReturnedWhenTakingPartialBatches(BaseDataBatcherTestCase):

    def runTest(self):
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

    def runTest(self):
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(2, self.db.get_num_epochs())

class EpochNumberIsCorrectlyReturnedWhenTakingMixedBatches(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(0, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(None)
        self.assertEqual(1, self.db.get_num_epochs())
        _, _, _ = self.db.next_batch(5)
        self.assertEqual(2, self.db.get_num_epochs())
        
if __name__ == "__main__":
    unittest.main() 