"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

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
from denoising_autoencoder import DenoisingAutoencoder


class StackedDenoisingAutoencoder(Classifier):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self, 
        numpy_rng, 
        theano_rng = None, 
        n_ins = 784,
        hidden_layers_sizes = [500, 500], 
        n_outs = 10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.parameters = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input_units=input_size,
                output_units=hidden_layers_sizes[i],
                nonlinear_function=T.nnet.sigmoid
            )
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.parameters.extend(sigmoid_layer.parameters)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = DenoisingAutoencoder(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.weights,
                bhid=sigmoid_layer.biases
            )
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticClassifier(
            input_units=hidden_layers_sizes[-1], 
            output_units=n_outs
        )

        self.parameters.extend(self.logLayer.parameters)
        # construct a function that implements one step of finetunining



    def cost_function(self, inputs, outputs):
        """docstring for finetune_cost"""
        # compute the cost for second phase of training,
        # defined as the negative log likelihood
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
        """docstring for errors"""
        prev_input = inputs
        for i in xrange(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            prev_input = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        return self.logLayer.evaluation_function(prev_input, outputs)


    def pretraining_functions(self, train_set_input, batch_size, inputs, outputs):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_input: theano.tensor.TensorType
        :param train_set_input: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_input.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_models = []
        prev_input = inputs
        for i in xrange(self.n_layers):
            # append `fn` to the list of functions
            pretrain_models.append(
                theano.function(
                    inputs=[
                        index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param(learning_rate, default=0.1)
                    ],
                    outputs=self.dA_layers[i].cost(prev_input, corruption_level),
                    updates=self.dA_layers[i].updates(prev_input, learning_rate, corruption_level),
                    givens={
                        inputs: train_set_input[batch_begin : batch_end]
                    }
                )
            )
            prev_input = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )

        return pretrain_models
