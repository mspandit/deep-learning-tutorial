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

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
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

        # function to compute vector of class-membership probabilities
        self.output_probabilities = Tensor.nnet.softmax(Tensor.dot(input, self.weights) + self.biases)

        # function to compute prediction as class whose probability is maximal
        self.predicted_output = Tensor.argmax(self.output_probabilities, axis = 1)

        # parameters of the model
        self.params = [self.weights, self.biases]
        self.negative_log_likelihood_fn = None

    def weights_gradient(self, outputs):
        """docstring for weights_gradient"""
        return Tensor.grad(cost = self.negative_log_likelihood(outputs), wrt = self.weights)
        
    def biases_gradient(self, outputs):
        """docstring for biases_gradient"""
        return Tensor.grad(cost = self.negative_log_likelihood(outputs), wrt = self.biases)
        
    def updates(self, outputs, learning_rate):
        """Specify how to update the parameters of the model as a list of (variable, update expression) pairs."""
        return [
            (self.weights, self.weights - learning_rate * self.weights_gradient(outputs)),
            (self.biases, self.biases - learning_rate * self.biases_gradient(outputs))
        ]
    
    def negative_log_likelihood(self, outputs):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type outputs: theano.tensor.TensorType
        :param outputs: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # outputs.shape[0] is (symbolically) the number of rows in outputs, i.e.,
        # number of examples (call it n) in the minibatch
        # Tensor.arange(outputs.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] Tensor.log(self.output_probabilities) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[Tensor.arange(outputs.shape[0]),outputs] is a vector
        # v containing [LP[0,outputs[0]], LP[1,outputs[1]], LP[2,outputs[2]], ...,
        # LP[n-1,outputs[n-1]]] and Tensor.mean(LP[Tensor.arange(outputs.shape[0]),outputs]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        if self.negative_log_likelihood_fn == None:
            self.negative_log_likelihood_fn = -Tensor.mean(Tensor.log(self.output_probabilities)[Tensor.arange(outputs.shape[0]), outputs])
        return self.negative_log_likelihood_fn

    def errors(self, outputs):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type outputs: theano.tensor.TensorType
        :param outputs: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if outputs has same dimension of predicted_output
        if outputs.ndim != self.predicted_output.ndim:
            raise TypeError('outputs should have the same shape as self.predicted_output',
                ('outputs', target.type, 'predicted_output', self.predicted_output.type))
        # check if outputs is of the correct datatype
        if outputs.dtype.startswith('int'):
            # the Tensor.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return Tensor.mean(Tensor.neq(self.predicted_output, outputs))
        else:
            raise NotImplementedError()

class LogisticClassifier(object):
    """docstring for LogisticClassifier"""
    def __init__(self, dataset, batch_size=600, n_epochs=1000):
        """
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        """
        super(LogisticClassifier, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_valid_batches = self.dataset.valid_set_input.get_value(borrow=True).shape[0] / self.batch_size
        self.n_test_batches = self.dataset.test_set_input.get_value(borrow=True).shape[0] / self.batch_size
        
    def mean_validation_loss(self):
        """docstring for mean_validation_loss"""
        return numpy.mean([self.validation_errors(batch_index) for batch_index in xrange(self.n_valid_batches)])
        
    def mean_test_loss(self):
        """docstring for mean_test_loss"""
        return numpy.mean([self.test_errors(batch_index) for batch_index in xrange(self.n_test_batches)])
        
    def train(self):
        """docstring for train"""

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.dataset.train_set_input.get_value(borrow=True).shape[0] / self.batch_size

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.

        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        done_looping = False
        epoch_losses = []
        epoch = 0
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                self.train_model(minibatch_index)
                
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = self.mean_validation_loss()
                    epoch_losses.append([this_validation_loss, iter])

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        test_score = self.mean_test_loss()

                if patience <= iter:
                    done_looping = True
                    break
        
        return [epoch_losses, best_validation_loss, best_iter, test_score]
    
        
    def initialize(self, learning_rate = 0.13):
        """
        Demonstrate stochastic gradient descent optimization of a log-linear
        model

        This is demonstrated on MNIST.

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
        """

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        # allocate symbolic variables for the data
        index = Tensor.lscalar()  # index to a [mini]batch
        inputs = Tensor.matrix('inputs')  # the data is presented as rasterized images
        outputs = Tensor.ivector('outputs')  # the labels are presented as 1D vector of
                               # [int] labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticRegression(input = inputs, n_in = 28 * 28, n_out = 10)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.test_errors = theano.function(
            inputs = [index],
            outputs = classifier.errors(outputs),
            givens = {
                inputs: self.dataset.test_set_input[index * self.batch_size: (index + 1) * self.batch_size],
                outputs: self.dataset.test_set_output[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.validation_errors = theano.function(
            inputs = [index],
            outputs = classifier.errors(outputs),
            givens = {
                inputs: self.dataset.valid_set_input[index * self.batch_size:(index + 1) * self.batch_size],
                outputs: self.dataset.valid_set_output[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        # compute the gradient of cost with respect to theta = (W,b)

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs = [index],
            updates = classifier.updates(outputs, learning_rate),
            givens = {
                inputs: self.dataset.train_set_input[index * self.batch_size:(index + 1) * self.batch_size],
                outputs: self.dataset.train_set_output[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    classifier = LogisticClassifier(dataset)
    classifier.initialize()

    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = classifier.train()
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100.0, test_score * 100.))
    