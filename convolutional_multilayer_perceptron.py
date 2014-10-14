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
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as Tensor
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from classifier import Classifier
from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer
from pooling_layer import PoolingLayer


from data_set import DataSet
from trainer import Trainer


class ConvolutionalMultilayerPerceptronClassifier(Classifier):
    """docstring for ConvolutionalMultilayerPerceptronClassifier"""
    def __init__(self, batch_size, nkerns=[20, 50]):
        super(ConvolutionalMultilayerPerceptronClassifier, self).__init__()
        self.nkerns = nkerns
        self.batch_size = batch_size
        rng = numpy.random.RandomState(23455)

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        # Reshape matrix of rasterized images of shape (self.batch_size,28*28)
        # to a 4D tensor, compatible with our PoolingLayer

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (self.batch_size,nkerns[0],12,12)
        self.layer0 = PoolingLayer(
            rng, 
            image_shape = (self.batch_size, 1, 28, 28),
            filter_shape = (nkerns[0], 1, 5, 5), 
            poolsize = (2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        self.layer1 = PoolingLayer(
            rng, 
            image_shape = (self.batch_size, nkerns[0], 12, 12),
            filter_shape = (nkerns[1], nkerns[0], 5, 5), 
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (self.batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng, 
            n_in = nkerns[1] * 4 * 4,
            n_out = 500, 
            nonlinear_function=Tensor.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticClassifier(n_in = 500, n_out = 10)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params


    def cost_function(self, inputs, outputs):
        """docstring for cost_function"""
        return self.layer3.negative_log_likelihood(
            self.layer2.output_probabilities_function(
                self.layer1.output_probabilities_function(
                    self.layer0.output_probabilities_function(
                        inputs.reshape((self.batch_size, 1, 28, 28))
                    )
                ).flatten(2)
            ),
            outputs
        )
        
    def evaluation_function(self, inputs, outputs):
        """docstring for evaluation_function"""
        return self.layer3.errors(
            self.layer2.output_probabilities_function(
                self.layer1.output_probabilities_function(
                    self.layer0.output_probabilities_function(
                        inputs.reshape((self.batch_size, 1, 28, 28))
                    )
                ).flatten(2)
            ), 
            outputs
        )

class ConvolutionalMultilayerPerceptronTrainer(Trainer):
    """docstring for ConvolutionalMultilayerPerceptronTrainer"""
    def __init__(self, dataset, n_epochs = 200, batch_size = 500):
        """
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        """
        super(ConvolutionalMultilayerPerceptronTrainer, self).__init__(dataset, batch_size, n_epochs)
        

    def initialize(self, learning_rate=0.1, nkerns=[20,50]):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        classifier = ConvolutionalMultilayerPerceptronClassifier(self.batch_size, nkerns)

        # allocate symbolic variables for the data
        index = Tensor.lscalar()  # index to a [mini]batch
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        ishape = (28, 28)  # this is the size of MNIST images

        # create a function to compute the mistakes that are made by the model
        self.test_errors = self.compiled_test_function(classifier, index, inputs, outputs)

        self.validation_errors = self.compiled_validation_function(classifier, index, inputs, outputs)

        self.train_model = self.compiled_training_function(classifier, index, inputs, outputs, learning_rate)

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    lenet5 = ConvolutionalMultilayerPerceptronTrainer(dataset)
    lenet5.initialize()
    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = lenet5.train()
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print('Best validation score of %f %% obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
