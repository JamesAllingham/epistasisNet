import data_batcher
import data_loader

import numpy as np

class DataHolder:
    
    def __init__(self):
        self.__testing = None
        self.__training = None
        self.__validation = None
        self.__data_loader = None

    def read_from_txt(self, file_name_and_path, test_train_ratio=0.8, train_valid_ratio=0.75):
        self.__data_loader = data_loader.DataLoader(file_name_and_path, test_train_ratio, train_valid_ratio)
        self.__data_loader.convert_data_to_1_hot()
        self.__data_loader.split_data()
        training_x, training_y1, training_y2 = self.__data_loader.get_training_data()
        self.__training = data_batcher.DataBatcher(training_x, training_y1, training_y2)
        testing_x, testing_y1, testing_y2 = self.__data_loader.get_testing_data()
        self.__testing = data_batcher.DataBatcher(testing_x, testing_y1, testing_y2)
        validation_x, validation_y1, validation_y2 = self.__data_loader.get_validation_data()
        self.__validation = data_batcher.DataBatcher(validation_x, validation_y1, validation_y2)

    def write_to_binary(self, file_name_and_path):
        training_x, training_y1, training_y2 = self.__data_loader.get_training_data()
        testing_x, testing_y1, testing_y2 = self.__data_loader.get_testing_data()
        validation_x, validation_y1, validation_y2 = self.__data_loader.get_validation_data()
        np.savez(file_name_and_path,
                 testing_x=testing_x, testing_y1=testing_y1, testing_y2=testing_y2,
                 training_x=training_x, training_y1=training_y1, training_y2=training_y2,
                 validation_x=validation_x, validation_y1=validation_y1, validation_y2=validation_y2)

    def read_from_binary(self, file_name_and_path):
        npzfile = np.load(file_name_and_path)
        self.__training = data_batcher.DataBatcher(npzfile['training_x'], npzfile['training_y1'], npzfile['training_y2'])
        self.__testing =  data_batcher.DataBatcher(npzfile['testing_x'], npzfile['testing_y1'], npzfile['testing_y2'])
        self.__validation =  data_batcher.DataBatcher(npzfile['validation_x'], npzfile['validation_y1'], npzfile['validation_y2'])

    def get_testing_data(self):
        return self.__testing

    def get_training_data(self):
        return self.__training
    
    def get_validation_data(self):
        return self.__validation