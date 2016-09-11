import data_batcher
import data_loader

class DataHolder:
    
    def __init__(self, file_name_and_path, test_train_ratio=0.2, train_valid_ratio=0.75):
        self.__data_loader = data_loader.DataLoader(file_name_and_path, test_train_ratio, train_valid_ratio)
        self.__data_loader.convert_data_to_1_hot()
        self.__data_loader.split_data()
        training_x, training_y = self.__data_loader.get_training_data()
        self.__training = data_batcher.DataBatcher(training_x, training_y)
        testing_x, testing_y = self.__data_loader.get_testing_data()
        self.__testing = data_batcher.DataBatcher(testing_x, testing_y)
        validation_x, validation_y = self.__data_loader.get_validation_data()
        self.__validation = data_batcher.DataBatcher(validation_x, validation_y)

    def get_testing_data(self):
        return self.__testing

    def get_training_data(self):
        return self.__training
    
    def get_validation_data(self):
        return self.__validation