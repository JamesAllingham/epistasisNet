from sys import exit
import numpy as np

class DataBatcher:

    def __init__(self, x, y1, y2):
        self.__x = x
        self.__y1 = y1
        self.__y2 = y2
        self.__batch_cursor = 0
        self.__data_size = self.__x.shape[0]

        if self.__data_size != self.__y1.shape[0]:
            print("The input and output sets must have the same number of entries")
            exit(2)

        if self.__y2.shape[0] != self.__y1.shape[0]:
            print("The output sets must have the same number of entries")
            exit(2)

    def next_batch(self, batch_size=50):
        if batch_size > self.__data_size:
            print("Please specify a batch size less than the number of entries in the data set")
            exit(2)

        if batch_size + self.__batch_cursor < self.__data_size:
            # If the batch size is less than the number of entries left in the data:
            # Take the next batch size number of elements and move the cursor forwards.
            x = self.__x[self.__batch_cursor:batch_size + self.__batch_cursor]
            y1 = self.__y1[self.__batch_cursor:batch_size + self.__batch_cursor]
            y2 = self.__y2[self.__batch_cursor:batch_size + self.__batch_cursor]
            self.__batch_cursor = self.__batch_cursor + batch_size
        else:
            # If there is not enough data left then take the remaining data from the end and start again at the begining.
            x = self.__x[self.__batch_cursor:]
            y1 = self.__y1[self.__batch_cursor:]
            y2 = self.__y2[self.__batch_cursor:]
            number_still_required = batch_size - (self.__data_size - self.__batch_cursor)
            x = np.concatenate((x, self.__x[0:number_still_required]))
            y1 = np.concatenate((y1, self.__y1[0:number_still_required]))
            y2 = np.concatenate((y2, self.__y2[0:number_still_required]))
            self.__batch_cursor = number_still_required

        return (x, y1, y2)

    def get_input_shape(self):
        return self.__x.shape

    def get_output1_shape(self):
        return self.__y1.shape

    def get_output2_shape(self):
        return self.__y2.shape