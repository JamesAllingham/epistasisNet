import sys
sys.path.append("../src/")
sys.path.append("src/")
import unittest
from os import remove, path

import data_holder
import data_batcher

class BaseDataHolderTestCase(unittest.TestCase):
    
    def setUp(self):
        try:
            input_file = open("tmp.txt", 'w')
            
        except Exception as e:
            print('Error writing temp file for testing')
            print(e)
            sys.exit(2)
        
        input_file.write("N1\tN2\tN3\tN4\tN5\tN6\tN7\tM8\tM9\tc\n")
        for i in range(50):
            input_file.write("0\t1\t2\t0\t1\t2\t0\t1\t2\t1\n")
        for i in range(50):
            input_file.write("2\t1\t0\t2\t1\t0\t2\t1\t0\t0\n")
        input_file.close()

        self.dh = data_holder.DataHolder()
        self.dh.read_from_txt("tmp.txt", 0.8, 0.75)


    def tearDown(self):
        remove("tmp.txt")

class ReadTxtTestCase(BaseDataHolderTestCase):

    def runTest(self):
        self.assertIsInstance(self.dh.get_testing_data(), data_batcher.DataBatcher)
        self.assertIsInstance(self.dh.get_training_data(), data_batcher.DataBatcher)
        self.assertIsInstance(self.dh.get_validation_data(), data_batcher.DataBatcher)

class WriteBinaryTestCase(BaseDataHolderTestCase):

    def runTest(self):
        self.dh.write_to_binary("tmp2")

        self.assertTrue(path.isfile("tmp2.npz"))

    def tearDown(self):
        remove("tmp2.npz")

class ReadBinaryTestCase(BaseDataHolderTestCase):

    def runTest(self):
        self.dh.write_to_binary("tmp2")

        dh2 = data_holder.DataHolder()
        dh2.read_from_npz("tmp2.npz")

        self.assertIsInstance(dh2.get_testing_data(), data_batcher.DataBatcher)
        self.assertIsInstance(dh2.get_training_data(), data_batcher.DataBatcher)
        self.assertIsInstance(dh2.get_validation_data(), data_batcher.DataBatcher)        
        
    def tearDown(self):
        remove("tmp2.npz")

if __name__ == "__main__":
    unittest.main() 