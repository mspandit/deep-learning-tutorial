"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys

import numpy

import theano
import theano.tensor as Tensor


from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer

class MultilayerPerceptronClassifier(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, L1_reg=0.00, L2_reg=0.0001):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)

        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng = rng, 
            n_in = n_in, 
            n_out = n_hidden,
            nonlinear_function = Tensor.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticClassifier(
            input = self.hiddenLayer.output_probabilities_function(input),
            n_in = n_hidden,
            n_out = n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.weights).sum() + abs(self.logRegressionLayer.weights).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.weights ** 2).sum() + (self.logRegressionLayer.weights ** 2).sum()

        # negative log likelihood of the MultilayerPerceptron is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
    
    def cost(self, outputs):
        """docstring for cost"""
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        return self.negative_log_likelihood(outputs) + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr
        
    def params_gradient(self, outputs):
        """compute the gradient of cost with respect to theta (stored in params). the resulting gradients will be stored in a list gparams"""
        return [Tensor.grad(self.cost(outputs), param) for param in self.params]
        
    def updates(self, outputs, learning_rate):
        """
        specify how to update the parameters of the model as a list of (variable, update expression) pairs
        given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        same length, zip generates a list C of same size, where each element
        is a pair formed from the two lists :
           C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        """
        return [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, self.params_gradient(outputs))]
