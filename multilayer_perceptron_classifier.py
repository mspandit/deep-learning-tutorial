import numpy

import theano
import theano.tensor as Tensor

from classifier import Classifier
from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer

class MultilayerPerceptronClassifier(Classifier):
    """
    """

    def initialize_l1(self, L1_reg):
        """
        L1 norm ; one regularization option is to enforce L1 norm to be small
        """

        self.L1 = (
            abs(self.hiddenLayer.weights).sum()
            + abs(self.logRegressionLayer.weights).sum()
        )
        self.L1_reg = L1_reg


    def initialize_l2(self, L2_reg):
        """
        square of L2 norm ; one regularization option is to enforce square of
        L2 norm to be small
        """

        self.L2_sqr = (
            (self.hiddenLayer.weights ** 2).sum()
            + (self.logRegressionLayer.weights ** 2).sum()
        )
        self.L2_reg = L2_reg


    def __init__(self, rng, n_in, n_hidden, n_out, L1_reg=0.00, L2_reg=0.0001):
        """
        """
        super(MultilayerPerceptronClassifier, self).__init__()

        self.hiddenLayer = HiddenLayer(
            rng=rng, 
            n_in=n_in, 
            n_out=n_hidden,
            nonlinear_function=Tensor.tanh
        )

        self.logRegressionLayer = LogisticClassifier(
            n_in=n_hidden,
            n_out=n_out
        )

        self.initialize_l1(L1_reg)
        self.initialize_l2(L2_reg)

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
    
    def cost_function(self, inputs, outputs):
        """docstring for cost"""
        return self.logRegressionLayer.negative_log_likelihood(self.hiddenLayer.output_probabilities_function(inputs), outputs) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

    def evaluation_function(self, inputs, outputs):
        """docstring for errors"""
        return self.logRegressionLayer.errors(self.hiddenLayer.output_probabilities_function(inputs), outputs)
