EpistasisNet
===

# Deep learning to detect gene-gene interactions

ELEN4002/4012 Project by James Allingham and Paul Cresswell. Supervised by Prof. Scott Hazelhurst.

# Prerequisites:
* Python 2.7 or higher
* NumPy
* TensorFlow

Optional (GPU support):
* NVIDIA CUDA 7.5
* cudNN 5.1

# Usage

EpistasisNet can be run from the command line using either the `python` or `python3` commands.

There are a number of command line options which can be specified as shown bellow:

Flag | Default Value | Description
---|---|---
-file_in |  | Input data file location
-tt_ratio| 0.8| test:train ratio
-max_steps| 1000| Maximum steps
-train_batch_size| 100| Training batch size
-test_batch_size| 1000| Testing batch size
-log_dir| /tmp/logs/runx| Directory for storing data
-learning_rate| 0.001| Initial Learning rate
-dropout| 0.5| Keep probability for training dropout
-model_dir| /tmp/tf_models/| Directory for storing the saved models
-write_binary| True| Write the processed numpy array to a binary file
-read_binary| True| Read a binary file rather than a text file
-save_model| True| Save the best model as the training progresses

EpistasisNet expects input text files to be in the format provided by [GAMETES](https://sourceforge.net/projects/gametes/). Note that the text files can be written to binary files by specifying the write_binary flag to be True. 

# Files

The files for EpistasisNet are:

Directory | File | Description
---|---|---
data | convert_from_BEAM_format.py | Converts data in the format used by the BEAM tool to the GAMETES format
     | convert_to_BEAM_format.py | Converts data in the GAMETES format to the BEAM format
docs | style_guide.html | Google's Python style guide
     | MeetingMinutes/\*.pdf | Minutes for various meetings held during the course of the projects
src | GPU_off.sh | A shell script that turns off GPU usage for EpistasisNet (as well as other CUDA applications)
    | GPU_on.sh | A shell script that turns on GPU usage for EpistasisNet (as well as other CUDA applications)
    | convolutional_model.py | Module that supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset
    | data_batcher.py | Module that provides a single class: DataLoader, which manages reading of raw data and formatting is appropriately
    | data_holder.py | Module that provides a single class: DataHolder, which manages reading of input files and storage of various data sets
    | data_loader.py | Module that provides a single class: DataLoader, which manages reading of raw data and formatting appropriately
    | linear_model.py | Module that supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset
    | model.py | Module that supplies a Model class which can be inherited from when creating models representing TensorFlow graphs
    | nonlinear_model.py | Module that supplies a fully connected model with nonlinearities to test for epistasis on a GAMETES dataset
    | pool_conv_model.py | Module that supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset
    | recurrent_model.py | Module that supplies a recurrent model with additional fully connected layers to test for epistasis on a GAMETES dataset
    | run_model.py | Module that trains a TensorFlow model
    | scaling_model | Module that supplies a convolutional model with pooling to test for epistasis on a GAMETES dataset - *Best Model*
    | utilities.py | Module that provides a number of wrapper functions for TensorFlow
tests | test_data_batcher.py | Module that provides test cases for the DataBatcher class
      | test_data_holder.py | Module that provides test cases for the DataHolder class
      | test_data_loader.py | Module that provides test cases for the DataLoader class
      | test_utilities.py | Module provides test cases for the utilities functions for building Tensorflow graphs
