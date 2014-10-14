"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time

import numpy

import theano
import theano.tensor as Tensor

from data_set import DataSet

class LogisticClassifier(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.weights = theano.shared(
            value = numpy.zeros((n_in, n_out), dtype = theano.config.floatX), 
            name = 'weights', 
            borrow = True
        )
        # initialize the baises as a vector of n_out 0s
        self.biases = theano.shared(
            value = numpy.zeros((n_out,), dtype = theano.config.floatX), 
            name = 'biases', 
            borrow = True
        )

        # parameters of the model
        self.parameters = [self.weights, self.biases]
    
    def output_probabilities(self, input):
        """function to compute vector of class-membership probabilities"""
        return Tensor.nnet.softmax(Tensor.dot(input, self.weights) + self.biases)
        
    def predicted_output(self, input):
        """function to compute prediction as class whose probability is maximal"""
        return Tensor.argmax(self.output_probabilities(input), axis = 1)

    def weights_gradient(self, inputs, outputs):
        """docstring for weights_gradient"""
        return Tensor.grad(cost = self.cost_function(inputs, outputs), wrt = self.weights)
        
    def biases_gradient(self, inputs, outputs):
        """docstring for biases_gradient"""
        return Tensor.grad(cost = self.cost_function(inputs, outputs), wrt = self.biases)
        
    def updates(self, inputs, outputs, learning_rate):
        """Specify how to update the parameters of the model as a list of (variable, update expression) pairs."""
        return [
            (self.weights, self.weights - learning_rate * self.weights_gradient(inputs, outputs)),
            (self.biases, self.biases - learning_rate * self.biases_gradient(inputs, outputs))
        ]
    
    def cost_function(self, inputs, outputs):
        """
        """
 
        return -Tensor.mean(Tensor.log(self.output_probabilities(inputs))[Tensor.arange(outputs.shape[0]), outputs])

    def evaluation_function(self, inputs, outputs):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type outputs: theano.tensor.TensorType
        :param outputs: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if outputs has same dimension of predicted_output
        if outputs.ndim != self.predicted_output(inputs).ndim:
            raise TypeError('outputs should have the same shape as self.predicted_output',
                ('outputs', target.type, 'predicted_output', self.predicted_output(inputs).type))
        # check if outputs is of the correct datatype
        if outputs.dtype.startswith('int'):
            # the Tensor.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return Tensor.mean(Tensor.neq(self.predicted_output(inputs), outputs))
        else:
            raise NotImplementedError()
