import numpy

import theano
import theano.tensor as Tensor

from classifier import Classifier
from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer

class MultilayerPerceptronClassifier(Classifier):
    """
    """

    def initialize_l1(self):
        """
        L1 norm ; one regularization option is to enforce L1 norm to be small
        """

        self.L1 = (
            abs(self.hiddenLayer.weights).sum()
            + abs(self.logRegressionLayer.weights).sum()
        )


    def initialize_l2(self):
        """
        square of L2 norm ; one regularization option is to enforce square of
        L2 norm to be small
        """

        self.L2_sqr = (
            (self.hiddenLayer.weights ** 2).sum()
            + (self.logRegressionLayer.weights ** 2).sum()
        )


    def __init__(self, rng, input, n_in, n_hidden, n_out, L1_reg=0.00, L2_reg=0.0001):
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
            input=self.hiddenLayer.output_probabilities_function(input),
            n_in=n_hidden,
            n_out=n_out
        )

        self.initialize_l1()
        self.initialize_l2()

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
    
    def cost_function(self, outputs):
        """docstring for cost"""
        return self.logRegressionLayer.negative_log_likelihood(outputs) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

    def evaluation_function(self, outputs):
        """docstring for errors"""
        return self.logRegressionLayer.errors(outputs)
