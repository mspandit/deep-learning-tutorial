"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from classifier import Classifier
from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer
from restricted_boltzmann_machine import RestrictedBoltzmannMachine


class DBN(Classifier):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """
        """
        super(DBN, self).__init__()
        
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.parameters = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.inputs = T.matrix('inputs')
        self.outputs = T.ivector('outputs')

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.inputs
            else:
                layer_input = (
                    self.sigmoid_layers[-1]
                    .output_probabilities_function(layer_input)
                )

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input_units=input_size,
                output_units=hidden_layers_sizes[i],
                nonlinear_function=T.nnet.sigmoid
            )

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.parameters.extend(sigmoid_layer.parameters)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RestrictedBoltzmannMachine(
                numpy_rng = numpy_rng,
                theano_rng = theano_rng,
                input = layer_input,
                n_visible = input_size,
                n_hidden = hidden_layers_sizes[i],
                W = sigmoid_layer.weights,
                hbias = sigmoid_layer.biases
            )
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticClassifier(
            input_units=hidden_layers_sizes[-1],
            output_units=n_outs)
        self.parameters.extend(self.logLayer.parameters)


    def cost_function(self, inputs, outputs):
        """
        compute the cost for second phase of training, defined as the
        negative log likelihood of the logistic regression (output) layer
        """
        prev_input = inputs
        for i in xrange(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            prev_input = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )
        return self.logLayer.cost_function(prev_input, outputs)


    def evaluation_function(self, inputs, outputs):
        """docstring for evaluation_function"""
        prev_input = inputs
        for i in xrange(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            prev_input = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )
        return self.logLayer.evaluation_function(prev_input, outputs)

    def pretraining_functions(self, inputs, train_set_input, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_input: theano.tensor.TensorType
        :param train_set_input: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_input.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        prev_inputs = inputs
        for i in xrange(self.n_layers):

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = self.rbm_layers[i].get_cost_updates(prev_inputs, learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={inputs: train_set_input[batch_begin: batch_end]}
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            prev_inputs = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_inputs)
            )
        return pretrain_fns
