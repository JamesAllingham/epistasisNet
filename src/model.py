"""This module supplies a Model class which can be inherited from when creating models representing TensorFlow graphs.
"""

class Model(object):
    """A class which can be inherited from when building a TensorFlow graph.
    """

    def __init__(self):
        """Creates a Model object.

        Initialises data members to None.

        Arguments:
            Nothing.

        Returns:
            A Model object.
        """
        self._accuracy1 = None
        self._accuracy2 = None
        self._loss1 = None
        self._loss2 = None
        self._merged = None
        self._train_step = None
        self._keep_prob = None

    def get_accuracies(self):
        """Returns sessions to run in order to get the accuracies for each of the outputs.

        Arguments:
            Nothing.

        Returns:
            (accuracy1, accuracy2) - TensorFlow sessions which return the accuracies for output 1 and output 2 respectively.
        """
        return self._accuracy1, self._accuracy2

    def get_losses(self):
        """Returns sessions to run in order to get the lesses for each of the outputs.

        Arguments:
            Nothing.

        Returns:
            (loss1, loss2) - TensorFlow sessions which return the losses for output 1 and output 2 respectively.
        """
        return self._loss1, self._loss2

    def get_merged(self):
        """Returns a session to run in order to get the merged graph summary.

        Arguments:
            Nothing.

        Returns:
            The merged TensorFlow session.
        """
        return self._merged

    def get_train_step(self):
        """Returns a session to run in order to train the network.

        Arguments:
            Nothing.

        Returns:
            The training TensorFlow session.
        """
        return self._train_step

    def get_keep_prob(self):
        """Returns a TensorFlow variable for the keep probability of the Dropout layer(s).

        Arguments:
            Nothing.

        Returns:
            The keep probability tensorflow varaible.
        """
        return self._keep_prob
