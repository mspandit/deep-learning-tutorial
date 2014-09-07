"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`weights` and a bias vector :math:`biases`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|input, weights,biases) &= softmax_i(weights input + biases) \\
                &= \frac {e^{weights_i input + biases_i}} {\sum_j e^{weights_j input + biases_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|input).

.. math::

  output_{pred} = argmax_i P(Y=i|input,weights,biases)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import sys
import time
import os

import numpy

import theano
import theano.tensor

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`weights`
    and bias vector :math:`biases`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.weights = theano.shared(
            value = numpy.zeros(
                (n_in, n_out), 
                dtype = theano.config.floatX
            ), 
            name = 'weights', 
            borrow = True)
            
        # initialize the biases as a vector of n_out 0s
        self.biases = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'biases', 
            borrow = True)

    def output_probability(self, input):
        """symbolic form of function computing vector of class-membership probabilities"""
        return theano.tensor.nnet.softmax(theano.tensor.dot(input, self.weights) + self.biases)

    def output_prediction(self, input):
        """symbolic form of function computing prediction as class whose probability is maximal"""
        return theano.tensor.argmax(self.output_probability(input), axis=1)

    def negative_log_likelihood(self, input, output):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{weights,biases\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=output^{(i)}|input^{(i)}, weights,biases)) \\
                \ell (\theta=\{weights,biases\}, \mathcal{D})

        :type output: theano.tensor.TensorType
        :param output: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # output.shape[0] is (symbolically) the number of rows in output, i.e.,
        # number of examples (call it n) in the batch
        # theano.tensor.arange(output.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] theano.tensor.log(self.output_probability) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[theano.tensor.arange(output.shape[0]),output] is a vector
        # v containing [LP[0,output[0]], LP[1,output[1]], LP[2,output[2]], ...,
        # LP[n-1,output[n-1]]] and theano.tensor.mean(LP[theano.tensor.arange(output.shape[0]),output]) is
        # the mean (across batch examples) of the elements in v,
        # i.e., the mean log-likelihood across the batch.
        return -theano.tensor.mean(theano.tensor.log(self.output_probability(input))[theano.tensor.arange(output.shape[0]), output])

    def errors(self, input, output):
        """Return a float representing the number of errors in the batch
        over the total number of examples of the batch ; zero one
        loss over the size of the batch

        :type output: theano.tensor.TensorType
        :param output: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if output has same dimension of output_pred
        if output.ndim != self.output_prediction(input).ndim:
            raise TypeError('output should have the same shape as self.output_pred',
                ('output', target.type, 'output_pred', self.output_pred.type))
        # check if output is of the correct datatype
        if output.dtype.startswith('int'):
            # the theano.tensor.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return theano.tensor.mean(theano.tensor.neq(self.output_prediction(input), output))
        else:
            raise NotImplementedError()

from data_set import DataSet
from trainer import Trainer

IMPROVEMENT_THRESHOLD = 0.995  # a relative improvement of this much is considered significant
def decreasing_rapidly(quantity, reference):
    """docstring for decreasing_rapidly"""
    return (quantity < reference * IMPROVEMENT_THRESHOLD)
        
PATIENCE_INCREASE = 2  # wait this many times longer when a new best is found

def sgd_optimization_mnist(
    learning_rate = 0.13, 
    n_epochs = 1000,
    dataset = 'mnist.pkl.gz',
    batch_size = 600
):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = DataSet(dataset, batch_size)
    datasets.load()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(n_in = 28 * 28, n_out = 10)
    trainer = Trainer(classifier, datasets, learning_rate)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    validation_interval = min(datasets.n_train_batches, patience / 2) # check validation set periodically
    best_validation_loss = numpy.inf
    start_time = time.clock()

    done_looping = False
    epoch = 0
    iter = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for batch_index in xrange(datasets.n_train_batches):

            batch_avg_cost = trainer.train(batch_index)

            if (iter + 1) % validation_interval == 0:
                print('epoch %i, batch %i/%i, ' % (epoch, batch_index + 1, datasets.n_train_batches))
                # compute zero-one loss on validation set
                this_validation_loss = trainer.validation.loss()
                print('          validation error %f%%' % (this_validation_loss * 100.))
    
                if decreasing_rapidly(this_validation_loss, best_validation_loss):
                    patience = max(patience, iter * PATIENCE_INCREASE)

                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_loss = trainer.test.loss()
                    print('          test error of best model %f%%' % (test_loss * 100.))
                else:
                    test_loss = 0.

                validation_interval = min(datasets.n_train_batches, patience / 2)
                
            if patience <= iter:
                done_looping = True
                break
            
            iter += 1

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%, with test performance %f %%') % (best_validation_loss * 100., test_loss * 100.))
    print 'The code ran for %d epochs at %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()