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

from logistic_classifier import LogisticClassifier
from hidden_layer import HiddenLayer
from restricted_boltzmann_machine import RBM


class DBN(object):
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
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

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
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                # input=layer_input,
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
            self.params.extend(sigmoid_layer.parameters)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(
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
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_input, batch_size, k):
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
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_input[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, dataset, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type dataset: list of pairs of theano.tensor.TensorType
        :param dataset: It is a list that contain all the datasets;
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

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: dataset.train_set_input[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: dataset.train_set_output[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: dataset.test_set_input[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: dataset.test_set_output[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: dataset.valid_set_input[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: dataset.valid_set_output[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

from trainer import Trainer

class DeepBeliefNetworkTrainer(Trainer):
    """docstring for DeepBeliefNetwork"""
    def __init__(self, dataset, pretraining_epochs=100, training_epochs=1000, 
                 pretrain_lr=0.01, finetune_lr = 0.1, 
                 batch_size=10):
        """
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining

        :type training_epochs: int
        :param training_epochs: maximal number of iterations ot run the optimizer

        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage

        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training

        :type batch_size: int
        :param batch_size: the size of a minibatch
        """
        super(DeepBeliefNetworkTrainer, self).__init__(dataset, batch_size, training_epochs)
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.dataset.train_set_input.get_value(borrow=True).shape[0] / self.batch_size
        
    def pretrain(self):
        """TODO: Factor this into Trainer."""

        layer_epoch_costs = []

        ## Pre-train layer-wise
        for i in xrange(self.dbn.n_layers):
            epoch_costs = []
            # go through pretraining epochs
            for epoch in xrange(self.pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_train_batches):
                    c.append(
                        self.pretraining_fns[i](
                            index = batch_index,
                            lr = self.pretrain_lr
                        )
                    )
                epoch_costs.append(numpy.mean(c))
                # print 'Pre-training layer %d, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
            layer_epoch_costs.append(epoch_costs)
    
        return layer_epoch_costs
    
    def mean_validation_loss(self):
        """docstring for mean_validation_loss"""
        return numpy.mean(self.validate_model())
        
    def mean_test_loss(self):
        """docstring for mean_test_loss"""
        return numpy.mean(self.test_model())

    def train(self, patience_increase = 2.0, improvement_threshold = 0.995):
        """docstring for train"""
        # early-stopping parameters
        patience = 4 * self.n_train_batches  # look as this many examples regardless
        return super(DeepBeliefNetworkTrainer, self).train(patience, patience_increase, improvement_threshold)

    def initialize(self, k = 1):
        """
        Demonstrates how to train and test a Deep Belief Network.

        This is demonstrated on MNIST.
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        """

        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)

        # construct the Deep Belief Network
        self.dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10)

        self.pretraining_fns = self.dbn.pretraining_functions(
            train_set_input = self.dataset.train_set_input,
            batch_size = self.batch_size,
            k = k
        )

        self.train_model, self.validate_model, self.test_model = self.dbn.build_finetune_functions(
            dataset = self.dataset, 
            batch_size = self.batch_size,
            learning_rate = self.finetune_lr
        )

from data_set import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    dbn = DeepBeliefNetwork(dataset)
    dbn.initialize()

    start_time = time.clock()
    layer_epoch_costs = dbn.pretrain()
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = dbn.train()
    end_time = time.clock()
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
