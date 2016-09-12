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
        
        file.write("x1 x2 x3 x4 x5 x6 x7 x8 x9 c\n")
        for i in range(50):
            file.write("0  1  2  0  1  2  0  1  2  1\n")
        for i in range(50):
            file.write("2  1  0  2  1  0  2  1  0  0\n")
        file.close()

        self.dl = data_loader.DataLoader("tmp.txt", 0.8, 0.75)
        remove("tmp.txt")

        self.dl.convert_data_to_1_hot()

        self.dl.split_data()

class GetInputDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, _ = self.dl.get_data()
        self.assertEqual(x[0,0], 0)
        self.assertEqual(x[0,8], 2)
        self.assertEqual(x[49,2], 2)
        self.assertEqual(x[49,8], 2)
        self.assertEqual(x[50,0], 2)
        self.assertEqual(x[50,8], 0)
        self.assertEqual(x[99,1], 1)
        self.assertEqual(x[99,8], 0)

class GetOutputDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, y = self.dl.get_data()
        self.assertEqual(y[0], 1)
        self.assertEqual(y[49], 1)
        self.assertEqual(y[50], 0)
        self.assertEqual(y[99], 0)

class GetInputOneHotdataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, _ = self.dl.get_1_hot_data()
        self.assertEqual(x[0,0,0], 1)
        self.assertEqual(x[0,0,1], 0)
        self.assertEqual(x[0,0,2], 0)
        self.assertEqual(x[49,2,0], 0)
        self.assertEqual(x[49,2,1], 0)
        self.assertEqual(x[49,2,2], 1)
        self.assertEqual(x[99,1,0], 0)
        self.assertEqual(x[99,1,1], 1)
        self.assertEqual(x[99,1,2], 0)

class GetOutputOneHotdataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        _, y = self.dl.get_1_hot_data()
        self.assertEqual(y[0,0], 0)
        self.assertEqual(y[0,1], 1)
        self.assertEqual(y[49,0], 0)
        self.assertEqual(y[49,1], 1)
        self.assertEqual(y[50,0], 1)
        self.assertEqual(y[50,1], 0)
        self.assertEqual(y[99,0], 1)
        self.assertEqual(y[99,1], 0)

class GetValidationDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y = self.dl.get_validation_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y), 20)

class GetTestingDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y = self.dl.get_testing_data()
        self.assertEqual(len(x), 20)
        self.assertEqual(len(y), 20)

class GetTrainingDataTestCase(BaseDataLoaderTestCase):

    def runTest(self):
        x, y = self.dl.get_training_data()
        self.assertEqual(len(x), 60)
        self.assertEqual(len(y), 60)

if __name__ == "__main__":
    unittest.main() 