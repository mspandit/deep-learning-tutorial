"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import numpy

import theano
import theano.tensor as Tensor
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from classifier import Classifier
from logistic_classifier import LogisticClassifier
from convolutional_classifier import ConvolutionalMultilayerPerceptronClassifier
from hidden_layer import HiddenLayer
from pooling_layer import PoolingLayer

from trainer import Trainer


class ConvolutionalMultilayerPerceptronTrainer(Trainer):
    """docstring for ConvolutionalMultilayerPerceptronTrainer"""

    def __init__(self, dataset, n_epochs=200, batch_size=500):
        """
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        """
        super(ConvolutionalMultilayerPerceptronTrainer, self).__init__(
            dataset,
            batch_size,
            n_epochs
        )

    def initialize(self, learning_rate=0.1, nkerns=[20, 50]):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        minibatch_index = Tensor.lscalar()
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        classifier = ConvolutionalMultilayerPerceptronClassifier(
            self.batch_size,
            nkerns
        )

        # create a function to compute the mistakes that are made by the model
        self.test_eval_function = self.compiled_test_function(
            classifier,
            minibatch_index,
            inputs,
            outputs
        )

        self.validation_eval_function = self.compiled_validation_function(
            classifier,
            minibatch_index,
            inputs,
            outputs
        )

        self.training_function = self.compiled_training_function(
            classifier,
            minibatch_index,
            inputs,
            outputs,
            learning_rate
        )


import time
from data_set import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    lenet5 = ConvolutionalMultilayerPerceptronTrainer(dataset)
    lenet5.initialize()
    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = lenet5.train()
    end_time = time.clock()
    print >> sys.stderr, (
        'The code for file '
        + os.path.split(__file__)[1]
        + ' ran for %.2fm'
        % ((end_time - start_time) / 60.)
    )
    print(
        'Best validation score of %f %% obtained at iteration %i, with '
        'test performance %f %%'
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
