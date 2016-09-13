import unittest
import sys
sys.path.append("../src/")
import data_batcher
import numpy as np

class BaseDataBatcherTestCase(unittest.TestCase):
    
    def setUp(self):
        x = np.array([range(10)]).T
        y = np.array([range(10)]).T
        self.db = data_batcher.DataBatcher(x,y)

class GetInputShapeTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(self.db.get_input_shape(), (10,1))

class GetOutputShapeTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        self.assertEqual(self.db.get_output_shape(), (10,1))

class FirstBatchGivesCorrectInputsTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        x, _ = self.db.next_batch(5)
        self.assertEqual(len(x), 5)
        self.assertEqual(x[0,0], 0)
        self.assertEqual(x[4,0], 4)

class FirstBatchGivesCorrectOutputsTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        _, y = self.db.next_batch(5)
        self.assertEqual(len(y), 5)
        self.assertEqual(y[0,0], 0)
        self.assertEqual(y[4,0], 4)

class InputBatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        x, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 0)
        x, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 5)
        x, _ = self.db.next_batch(5)
        self.assertEqual(x[0,0], 0)

class OutputBatchesRolloverCorrectlyTestCase(BaseDataBatcherTestCase):

    def runTest(self):
        y, _ = self.db.next_batch(5)
        self.assertEqual(y[0,0], 0)
        y, _ = self.db.next_batch(5)
        self.assertEqual(y[0,0], 5)
        y, _ = self.db.next_batch(5)
        self.assertEqual(y[0,0], 0)
        

if __name__ == "__main__":
    unittest.main() 