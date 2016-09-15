import unittest
import sys
sys.path.append("../src/")
import data_loader
import numpy as np
from random import sample
from os import remove

class BaseDataLoaderTestCase(unittest.TestCase):
    
    def setUp(self):
        try:
            file = open("tmp.txt",'w')
            
        except:
            print('Error writing temp file for testing')
            sys.exit(2)
        
        file.write("N1\tN2\tN3\tN4\tN5\tN6\tN7\tM8\tM9\tc\n")
        for i in range(50):
            file.write("0\t1\t2\t0\t1\t2\t0\t1\t2\t1\n")
        for i in range(50):
            file.write("2\t1\t0\t2\t1\t0\t2\t1\t0\t0\n")
        file.close()

        self.dl = data_loader.DataLoader("tmp.txt", 0.8, 0.75)
        remove("tmp.txt")

        self.dl.convert_data_to_1_hot()

        self.dl.split_data()

class GetInputDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, _, _ = self.dl.get_data()
        self.assertEqual(x[0,0], 0)
        self.assertEqual(x[0,8], 2)
        self.assertEqual(x[49,2], 2)
        self.assertEqual(x[49,8], 2)
        self.assertEqual(x[50,0], 2)
        self.assertEqual(x[50,8], 0)
        self.assertEqual(x[99,1], 1)
        self.assertEqual(x[99,8], 0)

class GetOutput1DataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, y1, _ = self.dl.get_data()
        self.assertEqual(y1[0], 1)
        self.assertEqual(y1[49], 1)
        self.assertEqual(y1[50], 0)
        self.assertEqual(y1[99], 0)

class GetOutput2DataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, _, y2 = self.dl.get_data()
        self.assertEqual(y2[0,7], 1)
        self.assertEqual(y2[0,8], 1)
        self.assertEqual(y2[49,6], 0)
        self.assertEqual(y2[50,7], 0)
        self.assertEqual(y2[99,8], 0)

class GetInputOneHotdataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, _, _ = self.dl.get_1_hot_data()
        self.assertEqual(x[0,0,0], 1)
        self.assertEqual(x[0,0,1], 0)
        self.assertEqual(x[0,0,2], 0)
        self.assertEqual(x[49,2,0], 0)
        self.assertEqual(x[49,2,1], 0)
        self.assertEqual(x[49,2,2], 1)
        self.assertEqual(x[99,1,0], 0)
        self.assertEqual(x[99,1,1], 1)
        self.assertEqual(x[99,1,2], 0)

class GetOutput1OneHotdataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, y1, _ = self.dl.get_1_hot_data()
        self.assertEqual(y1[0,0], 0)
        self.assertEqual(y1[0,1], 1)
        self.assertEqual(y1[49,0], 0)
        self.assertEqual(y1[49,1], 1)
        self.assertEqual(y1[50,0], 1)
        self.assertEqual(y1[50,1], 0)
        self.assertEqual(y1[99,0], 1)
        self.assertEqual(y1[99,1], 0)

class GetOutput2OneHotdataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, _, y2 = self.dl.get_1_hot_data()
        self.assertEqual(y2[0,8,0], 0)
        self.assertEqual(y2[0,8,1], 1)
        self.assertEqual(y2[99,8,0], 1)
        self.assertEqual(y2[99,8,1], 0)

class GetValidationDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y1, y2 = self.dl.get_validation_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y1), 20)
        self.assertEqual(len(y2), 20)

class GetTestingDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y1, y2 = self.dl.get_testing_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y1), 20)
        self.assertEqual(len(y2), 20)

class GetTrainingDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y1, y2 = self.dl.get_training_data()
        self.assertEqual(len(x), 60)
        self.assertEqual(len(y1), 60)
        self.assertEqual(len(y2), 60)

if __name__ == "__main__":
    unittest.main() 