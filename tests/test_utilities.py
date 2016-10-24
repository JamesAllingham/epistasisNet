"""This module provides test cases for the utilities functions for building Tensorflow graphs.

These tests are not meant to retest the Tensorflow functionality, they are only meant to test that the wrapper functions give the expectted tensorflow results.
"""

import sys

import numpy as np
import tensorflow as tf

sys.path.append("../src/")
sys.path.append("src/")

import utilities

class TnWeightVariableTest(tf.test.TestCase):
    """Tests for the tf_weight_variable function

    Inherits from the tf.test.TestCase class.
    """

    def testTnWeightShape(self):
        """Asserts that the tf_weight_varaible returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        weights_shape = tf.shape(utilities.tn_weight_variable(shape=[4, 2], standard_deviation=0.1))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([4, 2]), sess.run(weights_shape))

    def testStdDev(self):
        """Asserts that the standard deviation of a tensor's elements is approximately correct and that the mean is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        weights = utilities.tn_weight_variable(shape=[1000, 1000], standard_deviation=0.1)
        mean = tf.reduce_mean(weights)
        std_dev = tf.sqrt(tf.reduce_mean(tf.square(weights - mean)))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())

            self.assertNear(sess.run(std_dev), 0.1, err=2e-2)
            self.assertNear(sess.run(mean), 0.0, err=1e-4)

class ZerosWeightVariableTest(tf.test.TestCase):
    """Tests for the zeros_weight_variable function

    Inherits from the tf.test.TestCase class.
    """

    def testZerosWeightShape(self):
        """Asserts that the zeros_weight_varaible returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        weights_shape = tf.shape(utilities.tn_weight_variable(shape=[4, 33]))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([4, 33]), sess.run(weights_shape))

    def testZeroesInitialization(self):
        """Asserts that the sum of the absolute values of the tensor returned by zeros_weight_variable is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        weights = utilities.zeros_weight_variable(shape=[10, 10])
        elem_sum = tf.reduce_sum(tf.abs(weights))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertEqual(sess.run(elem_sum), 0)

class BiasVariableTest(tf.test.TestCase):
    """Tests for the bias_variable function.

    Inherits from the tf.test.TestCase class.
    """

    def testValuesInitialization(self):
        """Asserts that the bias_variable returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        bias_shape = tf.shape(utilities.bias_variable(shape=[1, 100]))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 100]), sess.run(bias_shape))

    def testMeanAndStdDevValues(self):
        """Asserts that the mean value of the bias variable is 0.1 and that the std deviation is 0.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        bias = utilities.bias_variable(shape=[100, 1])
        mean = tf.reduce_mean(bias)
        std_dev = tf.sqrt(tf.reduce_mean(tf.square(bias - mean)))
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertNear(sess.run(std_dev), 0.0, err=1e-4)
            self.assertAlmostEqual(sess.run(mean), 0.1)

class FcLayerTest(tf.test.TestCase):
    """Tests for the fc_layer function.

    Inherits from the tf.test.TestCase class.
    """

    def testFCLayerShape(self):
        """Asserts that the fc_layer returns a result with the correct values in each dimmension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([20, 30])
        output_tensor = utilities.fc_layer(input_tensor, 30, 100)
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([20, 100]), sess.run(output_tensor_shape))

    def testFCOps(self):
        """Asserts that the fc_layer correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor = tf.zeros([20, 30])
            _ = utilities.fc_layer(input_tensor, 30, 100, layer_name="hidden_1")
            op_dict = {"hidden_1/Wx_plus_b/MatMul": "MatMul", "hidden_1/Wx_plus_b/add": "Add", "hidden_1/activation": "Relu"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class ReshapeTest(tf.test.TestCase):
    """Tests for the reshape function.

    Inherits from the tf.test.TestCase class.
    """

    def testReshapeLayerShape(self):
        """Asserts that the reshape returns a result with the correct values in each dimension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([20, 30])
        output_tensor = utilities.reshape(input_tensor, [2, 300])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([2, 300]), sess.run(output_tensor_shape))

    def testFlattening(self):
        """Asserts that the reshape returns a result with the correct values in each dimension.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([20, 30])
        output_tensor = utilities.reshape(input_tensor, [-1, 600])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 600]), sess.run(output_tensor_shape))

    def testReshapeOps(self):
        """Asserts that the reshape correctly adds operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor = tf.zeros([20, 30])
            _ = utilities.reshape(input_tensor, [2, 300], name_suffix='1')
            op_dict = {"reshape_1/Reshape": "Reshape"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class ConvLayerTest(tf.test.TestCase):
    """Tests for the conv_layer function.

    Inherits from the tf.test.TestCase class.
    """

    def testSAMEPadding(self):
        """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'SAME', and the filter stride is set to [1, 1, 1, 1].

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 20, 30, 1])
        output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 20, 30, 5]), sess.run(output_tensor_shape))

    def testVALIDPadding(self):
        """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'VALID', and the filter stride is set to [1, 1, 1, 1].

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 10, 20, 1])
        output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='VALID')
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 6, 16, 5]), sess.run(output_tensor_shape))

    def testSkipStriding(self):
        """Provides a test for checking that the conv_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'SAME', and the filter stride is set to skip positions.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 20, 30, 1])
        output_tensor = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='SAME', strides=[1, 2, 2, 1])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 10, 15, 5]), sess.run(output_tensor_shape))

    def testConvOps(self):
        """Provides a test for checking that the conv_layer function adds correctly named operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor = tf.zeros([1, 20, 30, 1])
            _ = utilities.conv_layer(input_tensor, [5, 5, 1, 5], padding='SAME', strides=[1, 2, 2, 1], name_suffix='1')
            op_dict = {"conv_1/convolution/Conv2D": "Conv2D", "conv_1/activation": "Relu"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class PoolLayerTest(tf.test.TestCase):
    """Tests for the pool_layer function.

    Inherits from the tf.test.TestCase class.
    """

    def testSAMEPaddingPoolLayer(self):
        """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'SAME'.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 10, 10, 10])
        output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 10, 10, 10]), sess.run(output_tensor_shape))

    def testVALIDPaddingPoolLayer(self):
        """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'VALID'.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 10, 10, 10])
        output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1], padding='VALID')
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 9, 9, 10]), sess.run(output_tensor_shape))

    def testSkipStridingPoolLayer(self):
        """Provides a test for checking that the pool_layer function returns a tensor with the correct values in each dimmension,
        when padding is set to 'SAME', and the filter stride is set to skip positions.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([1, 20, 20, 2])
        output_tensor = utilities.pool_layer(input_tensor, shape=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 10, 10, 2]), sess.run(output_tensor_shape))

    def testOutputValues(self):
        """Provides a test for checking that the pool_layer returns the correct values.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected_output = tf.constant([[5, 6, 6], [8, 9, 9], [8, 9, 9]])
        reshaped_input = tf.reshape(input_tensor, [1, 3, 3, 1])
        reshaped_output = tf.reshape(expected_output, [1, 3, 3, 1])
        output_tensor = utilities.pool_layer(reshaped_input, shape=[1, 3, 3, 1])
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(sess.run(reshaped_output), sess.run(output_tensor))

    def testPoolOps(self):
        """Provides a test for checking that the pool_layer function adds correctly named operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor = tf.Variable([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=tf.float32)
            reshaped_input = tf.reshape(input_tensor, [1, 3, 3, 1])
            _ = utilities.pool_layer(reshaped_input)
            op_dict = {"pool_1/max_pooling/MaxPool": "MaxPool"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class DropoutLayerTest(tf.test.TestCase):
    """Tests to check the dropout function

    Inherits from the tf.test.TestCase class.
    """

    def testDropoutShape(self):
        """Provides a test for checking that the dropout_layer does not change the shape of the input.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.zeros([2, 3, 4, 5, 6])
        output_tensor, keep_prob = utilities.dropout(input_tensor)
        output_tensor_shape = tf.shape(output_tensor)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([2, 3, 4, 5, 6]), sess.run(output_tensor_shape, feed_dict={keep_prob : 0.1}))

    def testDropoutOps(self):
        """Provides a test for checking that the dropout_layer function adds correctly named operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor = tf.Variable([2, 3, 4, 5, 6], dtype=tf.float32)
            dropped, keep_prob = utilities.dropout(input_tensor, name_suffix='1')
            sess.run(tf.initialize_all_variables())
            sess.run(dropped, feed_dict={keep_prob : 0.1})
            op_dict = {"dropout_1/Placeholder": "Placeholder", "dropout_1/dropout/mul": "Mul", "dropout_1/dropout/Floor": "Floor", "dropout_1/dropout/add": "Add", "dropout_1/dropout/Shape": "Shape"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class CalculateCrossEntropyTest(tf.test.TestCase):
    """Tests for calculate_cross_entropy function.

    Inherits from the tf.test.TestCase class.
    """

    def testCrossEntropyOps(self):
        """Provides a test for checking that the calculate_cross_entropy function adds correctly named operations to the graph.

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        with self.test_session() as sess:
            input_tensor1 = tf.convert_to_tensor([0.1, 0.9, 0.8])
            input_tensor2 = tf.convert_to_tensor([0.2, 0.3, 0.7])
            _ = utilities.calculate_cross_entropy(input_tensor1, input_tensor2, name_suffix='1')
            op_dict = {"cross_entropy_1/add": "Add", "cross_entropy_1/Log": "Log", "cross_entropy_1/mul": "Mul", "cross_entropy_1/total/Mean": "Mean", "cross_entropy_1/total/Neg": "Neg"}
            tf.python.framework.test_util.assert_ops_in_graph(op_dict, tf.get_default_graph())

class PredictSnpsTest(tf.test.TestCase):
    """Tests for the predict_snps funtion

    Inherits from the tf.test.TestCase class.
    """

    def testCorrectSnpsIndicesFoundForTwoClassifier(self):
        """Provides a test for checking that the function predict_snps() is able to find the snps predicted for a double classifier

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.constant([[[0.5, 0.8], [0.2, 0.8], [0.9, 0.1]], [[0.2, 0.8], [0.49, 0.8], [0.51, 0.49]]])
        output_tensor, _ = utilities.predict_snps(input_tensor, 0.5)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([0, 2]), sess.run(output_tensor))

    def testCorrectSnpsCountFoundForTwoClassifier(self):
        """Provides a test for checking that the function predict_snps() is able to find the count for the snps predicted for a double classifier

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.constant([[[0.5, 0.8], [0.2, 0.8], [0.9, 0.1]], [[0.2, 0.8], [0.49, 0.8], [0.51, 0.49]]])
        _, count = utilities.predict_snps(input_tensor, 0.5)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 2]), sess.run(count))

    def testCorrectSnpsIndicesFoundForSingleClassifier(self):
        """Provides a test for checking that the function predict_snps() is able to find the snps predicted predicted for a single classifier

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.constant([[[0.99], [0.2], [0.99]], [[0.2], [0.97], [0.98]]])
        output_tensor, _ = utilities.predict_snps(input_tensor, 0.98, already_split=True)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([0, 2]), sess.run(output_tensor))

    def testCorrectSnpsCountFoundForSingleClassifier(self):
        """Provides a test for checking that the function predict_snps() is able to find the count for the snps predicted for a single classifier

        Arguments:
            Nothing.

        Returns:
            Nothing.
        """
        input_tensor = tf.constant([[[0.99], [0.2], [0.99]], [[0.2], [0.97], [0.98]]])
        _, count = utilities.predict_snps(input_tensor, 0.98, already_split=True)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            self.assertAllEqual(np.array([1, 2]), sess.run(count))

# class CorrectlyCalculateAccuracyForATwoClassifier(tf.test.TestCase):
#     """Provides a test for checking that the function calculate_snp_accuracy() is able to return a correct accuracy check for a double classifier model

#     Inherits from the tf.test.TestCase class.
#     """
#     def runTest(self):
#         """Asserts that accuracy is summed and meaned over the data

#         Arguments:
#             Nothing.

#         Returns:
#             Nothing.
#         """
#         with self.test_session() as sess:
#             input_tensor = tf.constant([[[0.5, 0.8], [0.2, 0.8], [0.9, 0.1]], [0.2, 0.8], [0.49, 0.8], [0.51, 0.49]])
#             labels = tf.constant([[[0, 1], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]])
#             self.assertEqual(utilities.calculate_snp_accuracy(input_tensor, labels), 1)

# class CorrectlyCalculateAccuracyForASingleClassifier(tf.test.TestCase):
#     """Provides a test for checking that the function calculate_snp_accuracy() is able to return a correct accuracy check for a single classifier model

#     Inherits from the tf.test.TestCase class.
#     """
#     def runTest(self):
#         """Asserts that accuracy is correctly summed and meaned over the predicted snp data

#         Arguments:
#             Nothing.

#         Returns:
#             Nothing.
#         """
#         with self.test_session() as sess:
#             input_tensor = tf.constant([[[0.99], [0.2], [0.99]], [[0.2], [0.49], [0.99]]])
#             labels = tf.constant([[[0, 1], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]]])
#             self.assertEqual(utilities.calculate_snp_accuracy(input_tensor, labels, cut_off_prob=0.98), 0.75)

# class GetOutputSnpHeaders(unittest.TestCase):
#     """Provides a test for checking that the correct snp headers are returned from the snp numbers

#     Inherits from the unittest.TestCase class.
#     """
#     def runTest(self):
#         """Asserts that the function get_snp_headers correctly fetches header names for the snps

#         Arguments:
#             Nothing.

#         Returns:
#             Nothing.
#         """
#         headers = np.array(['N0', 'N1', 'MP01', 'MP02', 'MP03'])
#         snp_indices = np.array([2, 3, 4])
#         self.assertTrue(np.array_equal(utilities.get_snp_headers(snp_indices, headers), np.array(['MP01', 'MP02', 'MP03'])))

if __name__ == '__main__':
    tf.test.main()
