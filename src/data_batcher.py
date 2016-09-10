from sys import exit
import numpy as np

class DataBatcher:

    def __init__(self, path_to_data, data_x_filename, data_y_filename):
        FILE_TYPE = ".npy"
        self.__x = np.load(path_to_data + data_x_filename + FILE_TYPE)
        self.__y = np.load(path_to_data + data_y_filename + FILE_TYPE)
        self.__batch_cursor = 0
        self.__data_size = self.__x.shape[0]

        if self.__data_size != self.__y.shape[0]:
            print("The input and output sets must have the same number of entries")
            exit(2)

    def next_batch(self, batch_size=50):
        if batch_size > self.__data_size:
            print("Please specify a batch size less than the number of entries in the data set")
            exit(2)

        if batch_size + self.__batch_cursor < self.__data_size:
            # If the batch size is less than the number of entries left in the data:
            # Take the next batch size number of elements and move the cursor forwards.
            x = self.__x[self.__batch_cursor:batch_size + self.__batch_cursor]
            y = self.__y[self.__batch_cursor:batch_size + self.__batch_cursor]
            self.__batch_cursor = self.__batch_cursor + batch_size
        else:
            # If there is not enough data left then take the remaining data from the end and start again at the begining.
            x = self.__x[self.__batch_cursor:]
            y = self.__y[self.__batch_cursor:]
            number_still_required = batch_size - (self.__data_size - self.__batch_cursor)
            x = np.concatenate((x, self.__x[0:number_still_required]))
            y = np.concatenate((y, self.__y[0:number_still_required]))
            self.__batch_cursor = number_still_required

        return (x, y)

    def get_input_shape(self):
        return self.__x.shape

    def get_output_shape(self):
        return self.__y.shape


class DataHolder:
    
    def __init__(self):
        self.train = DataBatcher("../data/", "train_x", "train_y")
        self.test = DataBatcher("../data/", "test_x", "test_y")
        self.valid = None

def main():
    dh = DataHolder()
    for i in range(1000):
        batch_xs = dh.train.next_batch(100)
        batch_ys = dh.train.next_batch(100)
        print(i)    

# Testing 
if __name__ == "__main__":
	main()