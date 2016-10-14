"""This module provides test cases for the utilities functions for building Tensorflow graphs.

These tests are not meant to retest the Tensorflow functionality, they are only meant to test that the wrapper functions give the expectted tensorflow results.
"""

import sys

import numpy as np
import unittest
import tensorflow as tf

sys.path.append("../src/")
sys.path.append("src/")

import utilities

class TnWeightVariableCorrectSizeTestCase(tf.test.TestCase):
    """Provides a test for checking that the tf_weight_varaible function returns the correctly sized tensor.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the tf_weight_varaible returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            weights = utilities.tn_weight_variable(shape=[4, 2], standard_deviation=0.1)
            self.assertAllEqual(weights.get_shape(), [4, 2])

class TnWeightVariableCorrectStdDevTestCase(tf.test.TestCase):
    """Provides a test for checking that the tf_weight_varaible function returns a tensor whos elements have the correct standard deviation.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the standard deviation of a tensor's elements is approxiamtely correct and that the mean is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            weights = utilities.tn_weight_variable(shape=[1000, 1000], standard_deviation=0.1)
            mean = tf.reduce_mean(weights)
            std_dev = tf.sqrt(tf.reduce_sum(tf.square(weights - mean)))
            self.assertClose(std_dev, 0.1)
            self.assertClose(mean, 0)

class ZerosWeightVariableCorrectSizeTestCase(tf.test.TestCase):
    """Provides a test for checking that the zeros_weight_varaible function returns the correctly sized tensor.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the zeros_weight_varaible returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            weights = utilities.tn_weight_variable(shape=[4, 33])
            self.assertShapeEqual(weights.get_shape(), [4, 33])

class ZerosWeightVariableCorrectValueTestCase(tf.test.TestCase):
    """Provides a test for checking that the zeros_weight_varaible function returns a tensor whos elements are all 0.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the sum of the absolute values of the tensor returned by zeros_weight_variable is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            weights = utilities.zeros_weight_variable(shape=[10, 10])
            self.assertEqual(tf.reduce_sum(tf.abs(weights)), 0)

class BiasVariableCorrectSizeTestCase(tf.test.TestCase):
    """Provides a test for checking that the bias_varaible function returns the correctly sized tensor.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the bias_varaible returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            bias = utilities.bias_variable(shape=[1, 100])
            self.assertShapeEqual(bias.get_shape(), [1, 100])

class BiasVariableCorrectValueTestCase(tf.test.TestCase):
    """Provides a test for checking that the bias_varaible function returns a tensor whos elements are all 0.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the mean value of the bias variable is 0.1 and that the std deviation is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            bias = utilities.bias_varaible(shape=[100, 1])
            mean = tf.reduce_mean(bias)
            std_dev = tf.sqrt(tf.reduce_sum(tf.square(bias - mean)))
            self.assertEqual(std_dev, 0)
            self.assertEqual(mean, 0.1)

class FcLayerCorrectShapeTestCase(tf.test.TestCase):
    """Provides a test for checking that the fc_layer function returns a tensor with the correct values in each dimmension.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the fc_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30])
            output_tensor = utilities.fc_layer(input_tensor, 30, 100)
            self.assertShapeEqual(output_tensor.get_shape(), [20, 100])

class FcLayerAddsCorrectlyNamedOperationsToGraph(tf.test.TestCase):
    """Provides a test for checking that the fc_layer function adds correctly named operations to the graph.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the fc_layer correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30])
            _ = utilities.fc_layer(input_tensor, 30, 100, layer_name="hidden_1")
            op_dict = {"hidden_1/Wx_plus_b/MatMul": "MatMul", "hidden_1/Wx_plus_b/add": "add", "hidden_1/activation": "Relu"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class ReshapeCorrectlyChangesShapeTestCase(tf.test.TestCase):
    """Provides a test for checking that the rehsape function returns a tensor with the correct values in each dimmension.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the rehsape returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30])
            output_tensor = utilities.reshape(input_tensor, [2, 300])
            self.assertAllEqual(output_tensor.get_shape(), [2, 300])

class ReshapeCorrectlyFlattensTestCase(tf.test.TestCase):
    """Provides a test for checking that the rehsape function returns a tensor with the correct values in each dimmension,
    specifically that it is flat in one dimmension if a -1 is given as a parameter for that dimmension.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the rehsape returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30])
            output_tensor = utilities.reshape(input_tensor, [-1, 600])
            self.assertShapeEqual(output_tensor.get_shape(), [1, 600])

class ReshapeAddsCorrectlyNamedOperationsToGraph(tf.test.TestCase):
    """Provides a test for checking that the reshape function adds correctly named operations to the graph.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the reshape correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30])
            _ = utilities.reshape(input_tensor, [2, 300], name_suffix='1')
            op_dict = {"reshape_1/Reshape": "Reshape"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class ConvLayerCorrectShapeSAMEPaddingTestCase(tf.test.TestCase):
    """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'SAME', and the filter stride is set to [1, 1, 1, 1].

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the conv_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30, 1])
            output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5])
            self.assertShapeEqual(output_tensor.get_shape(), [20, 30, 5])

class ConvLayerCorrectShapeVALIDPaddingTestCase(tf.test.TestCase):
    """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'VALID', and the filter stride is set to [1, 1, 1, 1].

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the conv_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([10, 20, 5])
            output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='VALID')
            self.assertShapeEqual(output_tensor.get_shape(), [12, 22, 5])

class ConvLayerCorrectShapeSAMEPaddingAndStridingTestCase(tf.test.TestCase):
    """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'SAME', and the filter stride is set to skip positions.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the conv_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30, 1])
            output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='SAME', strides=[1, 2, 2, 1])
            self.assertShapeEqual(output_tensor.get_shape(), [10, 15, 5])

class ConvLayerAddsCorrectlyNamedOperationsToGraph(tf.test.TestCase):
    """Provides a test for checking that the conv_layer function adds correctly named operations to the graph.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the conv_layer correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 30, 1])
            _ = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='SAME', strides=[1, 2, 2, 1], name_suffix='1')
            op_dict = {"conv_1/convolution/Conv2D": "Conv2D", "conv_1/activation": "Relu"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class PoolLayerCorrectShapeSAMEPAddingTestCase(tf.test.TestCase):
    """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'SAME'.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the pool_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([10, 10, 10])
            output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1])
            self.assertShapeEqual(output_tensor.get_shape(), [5, 5, 10])

class PoolLayerCorrectShapeVALIDPAddingTestCase(tf.test.TestCase):
    """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'VALID'.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the pool_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([10, 10, 10])
            output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1], padding='VALID')
            self.assertShapeEqual(output_tensor.get_shape(), [4, 4, 10])

class PoolLayerCorrectShapeSAMEPAddingAndStridesTestCase(tf.test.TestCase):
    """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
    when padding is set to 'SAME', and the filter stride is set to skip positions.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the pool_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([20, 20, 2])
            output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            self.assertShapeEqual(output_tensor.get_shape(), [5, 5, 2])

class PoolLayerCorrectValueTestCase(tf.test.TestCase):
    """Provides a test for checking that the pool_layer returns the correct values.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the pool_layer returns the correct values. The values should all be maximum values over the pooling kernel.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.convert_to_tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
            output_tensor = utilities.pool_layer(input_tensor)
            self.assertAllEqual(output_tensor, [[[5, 6, 6], [8, 9, 9], [8, 9, 9]]])

class PoolLayerAddsCorrectlyNamedOperationsToGraph(tf.test.TestCase):
    """Provides a test for checking that the pool_layer function adds correctly named operations to the graph.

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the pool_layer correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.convert_to_tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
            _ = utilities.pool_layer(input_tensor)
            op_dict = {"pool_1/max_pooling/MaxPool": "MaxPool"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class DropoutLayerCorrectShapeTestCase(tf.test.TestCase):
    """Provides a test for checking that the dropout_layer does not change the shape of the input.mro

    Inherits from the tf.test.TestCase class.
    """

    def runTest(self):
        """Asserts that the dropout_layer output is the same shape and the input.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session():
            input_tensor = tf.zeros([2, 3, 4, 5, 6])
            output_tensor = utilities.dropout(input_tensor)
            self.assertShapeEqual(output_tensor.get_shape(), [2, 3, 4, 5, 6])

class GetOutputSnpHeaders(unittest.TestCase):
    """Provides a test for checking that the correct snp headers are returned from the snp numbers
    """
    def runTest(self):
        """Asserts that the function get_snp_headers correctly fetches header names for the snps

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        headers = np.array(['N0', 'N1', 'MP01', 'MP02', 'MP03'])
        snp_indices = np.array([2, 3, 4])
        self.assertTrue(np.array_equal(utilities.get_snp_headers(snp_indices, headers), np.array(['MP01', 'MP02', 'MP03'])))

if __name__ == '__main__':
    tf.test.main()
