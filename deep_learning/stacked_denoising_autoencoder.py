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

from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer
from denoising_autoencoder import dA


class SdA(object):
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
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

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

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = (
                    self.sigmoid_layers[-1]
                    .output_probabilities_function(layer_input)
                )

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                # input=layer_input,
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
            self.params.extend(sigmoid_layer.parameters)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.weights,
                bhid=sigmoid_layer.biases
            )
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticClassifier(
            # input=self.sigmoid_layers[-1].output_probabilities_function(layer_input),
            n_in=hidden_layers_sizes[-1], 
            n_out=n_outs
        )

        self.params.extend(self.logLayer.parameters)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        prev_input = self.x
        for i in xrange(self.n_layers):
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            prev_input = (
                self.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )
        self.finetune_cost = self.logLayer.cost_function(prev_input, self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.evaluation_function(prev_input, self.y)

    def pretraining_functions(self, train_set_input, batch_size):
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
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost = dA.cost(corruption_level)
            updates = dA.updates(learning_rate, corruption_level)
            # compile the theano function
            fn = theano.function(
                inputs = [
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs = cost,
                updates = updates,
                givens = {
                    self.x: train_set_input[batch_begin : batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_models.append(fn)

        return pretrain_models

    def build_finetune_functions(self, dataset, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        # compute number of minibatches for training, validation and testing
        n_valid_batches = dataset.valid_set_input.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = dataset.test_set_input.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_model = theano.function(inputs=[index],
            outputs = self.finetune_cost,
            updates = updates,
            givens = {
                self.x: dataset.train_set_input[index * batch_size : (index + 1) * batch_size],
                self.y: dataset.train_set_output[index * batch_size : (index + 1) * batch_size]
            },
            name = 'train'
        )

        test_score_i = theano.function(
            inputs = [index], 
            outputs = self.errors,
            givens = {
                self.x: dataset.test_set_input[index * batch_size : (index + 1) * batch_size],
                self.y: dataset.test_set_output[index * batch_size : (index + 1) * batch_size]
            },
            name = 'test'
        )

        valid_score_i = theano.function(
            inputs = [index], 
            outputs = self.errors,
            givens = {
                self.x: dataset.valid_set_input[index * batch_size : (index + 1) * batch_size],
                self.y: dataset.valid_set_output[index * batch_size : (index + 1) * batch_size]
            },
            name = 'valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_model, valid_score, test_score


from data_set import DataSet
from trainer import Trainer

class StackedDenoisingAutoencoder(Trainer):
    """docstring for StackedDenoisingAutoencoder"""
    def __init__(self, dataset, pretraining_epochs = 15, n_epochs = 1000, batch_size = 1, pretrain_lr = 0.001):
        """
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining

        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        """
        super(StackedDenoisingAutoencoder, self).__init__(dataset, batch_size, n_epochs)
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr

    def preinitialize(self):
        """docstring for preinitialize"""
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.dataset.train_set_input.get_value(borrow=True).shape[0] / self.batch_size

        # numpy random generator
        self.numpy_rng = numpy.random.RandomState(89677)

        # construct the stacked denoising autoencoder class
        self.sda = SdA(
            numpy_rng = self.numpy_rng, 
            n_ins = 28 * 28,
            hidden_layers_sizes = [1000, 1000, 1000],
            n_outs=10
        )

        #########################
        # PRETRAINING THE MODEL #
        #########################
        self.pretraining_fns = self.sda.pretraining_functions(
            train_set_input = self.dataset.train_set_input,
            batch_size = self.batch_size
        )
    
    def pretrain(self):
        """TODO: Factor this into Trainer."""
    
        ## Pre-train layer-wise
        corruption_levels = [.1, .2, .3]
        layer_epoch_costs = []
        for layer_index in xrange(self.sda.n_layers):
            # go through pretraining epochs
            epoch_costs = []
            epoch = 0
            while epoch < self.pretraining_epochs:
                # go through the training set
                layer_costs = []
                for batch_index in xrange(self.n_train_batches):
                    layer_costs.append(
                        self.pretraining_fns[layer_index](
                            index = batch_index,
                            corruption = corruption_levels[layer_index],
                            lr = self.pretrain_lr
                        )
                    )
                # print 'Pretraining layer %d, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
                epoch_costs.append(numpy.mean(layer_costs))
                epoch += 1
            layer_epoch_costs.append(epoch_costs)
        return layer_epoch_costs

    def mean_validation_loss(self):
        """docstring for mean_validation_loss"""
        return numpy.mean(self.validation_eval_function())
        
    def mean_test_loss(self):
        """docstring for mean_test_loss"""
        return numpy.mean(self.test_eval_function())
    
    def train(self, patience, patience_increase = 2.0, improvement_threshold = 0.995):
        """docstring for train"""
        patience = 10 * self.n_train_batches  # look as this many examples regardless
        return super(StackedDenoisingAutoencoder, self).train(patience, patience_increase, improvement_threshold)
    
    def initialize(self, finetune_lr=0.1):
        """
        Demonstrates how to train and test a stochastic denoising autoencoder.

        This is demonstrated on MNIST.

        :type learning_rate: float
        :param learning_rate: learning rate used in the finetune stage
        (factor for the stochastic gradient)

        :type n_iter: int
        :param n_iter: maximal number of iterations ot run the optimizer
        """
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model
        self.training_function, self.validation_eval_function, self.test_eval_function = self.sda.build_finetune_functions(
            dataset = self.dataset, 
            batch_size = self.batch_size,
            learning_rate = finetune_lr
        )
        
if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    sda = StackedDenoisingAutoencoder(dataset)
    sda.preinitialize()
    start_time = time.clock()
    layer_epoch_costs = sda.pretrain()
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    sda.initialize()
    start_time = time.clock()
    epoch_validation_losses, best_validation_loss, best_iter, test_score = sda.train(None)
    end_time = time.clock()
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
