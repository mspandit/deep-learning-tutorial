import os
import sys
import time
import argparse

import numpy
import theano
import theano.tensor as Tensor
from data_set import DataSet
from trainer import Trainer
from stacked_denoising_autoencoder import StackedDenoisingAutoencoder


class StackedDenoisingAutoencoderTrainer(Trainer):
    """docstring for StackedDenoisingAutoencoder"""

    def __init__(
        self,
        dataset,
        pretraining_epochs=15,
        n_epochs=1000,
        batch_size=1,
        pretrain_lr=0.001
    ):
        """
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining

        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        """
        super(StackedDenoisingAutoencoderTrainer, self).__init__(
            dataset,
            batch_size,
            n_epochs
        )
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr


    def compiled_pretraining_functions(
        self,
        classifier,
        minibatch_index,
        train_set_input,
        inputs,
        outputs
    ):
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

        corruption_level = Tensor.scalar('corruption')  # % of corruption to use
        learning_rate = Tensor.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = minibatch_index * self.batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + self.batch_size

        pretrain_models = []
        prev_input = inputs
        for i in xrange(classifier.n_layers):
            # append `fn` to the list of functions
            pretrain_models.append(
                theano.function(
                    inputs=[
                        minibatch_index,
                        theano.Param(corruption_level, default=0.2),
                        theano.Param(learning_rate, default=0.1)
                    ],
                    outputs=classifier.dA_layers[i].cost(prev_input, corruption_level),
                    updates=classifier.dA_layers[i].updates(prev_input, learning_rate, corruption_level),
                    givens={
                        inputs: train_set_input[batch_begin : batch_end]
                    }
                )
            )
            prev_input = (
                classifier.sigmoid_layers[i]
                .output_probabilities_function(prev_input)
            )

        return pretrain_models


    def preinitialize(self):
        """docstring for preinitialize"""

        minibatch_index = Tensor.lscalar('minibatch_index')
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        # construct the stacked denoising autoencoder class
        self.classifier = StackedDenoisingAutoencoder(
            numpy_rng = numpy.random.RandomState(89677), 
            n_ins = 28 * 28,
            hidden_layers_sizes = [1000, 1000, 1000],
            n_outs=10
        )

        self.pretraining_fns = self.compiled_pretraining_functions(
            self.classifier,
            minibatch_index,
            self.dataset.train_set_input,
            inputs,
            outputs
        )

    class State(object):
        """docstring for StackedDenoisingAutoencoderTrainer.State"""
        def __init__(self, n_layers, pretraining_epochs):
            super(StackedDenoisingAutoencoderTrainer.State, self).__init__()
            self.corruption_levels = [(i + 1) * 0.1 for i in xrange(n_layers)]
            self.layer_epoch_combos = [[i, j] for i in xrange(n_layers) for j in xrange(pretraining_epochs)]
            self.layer_epoch_costs = []
            self.layer_epoch_index = 0

        def continue_pretraining(self):
            """docstring for continue_pretraining"""
            return self.layer_epoch_index < len(self.layer_epoch_combos)

        def current_layer(self):
            """docstring for current_layer"""
            return self.layer_epoch_combos[self.layer_epoch_index][0]


    def start_pretraining(self):
        """docstring for start_pretraining"""
        return StackedDenoisingAutoencoderTrainer.State(self.classifier.n_layers, self.pretraining_epochs)

    def continue_pretraining(self, state):
        """docstring for continue_pretraining"""
        if state.continue_pretraining():
            layer_costs = [
                self.pretraining_fns[state.current_layer()](
                    minibatch_index=batch_index,
                    corruption=state.corruption_levels[state.current_layer()],
                    lr=self.pretrain_lr
                )
                for batch_index in xrange(self.n_train_batches)
            ]
            state.layer_epoch_costs.append(numpy.mean(layer_costs))
            state.layer_epoch_index += 1
            return True
        else:
            return False

    def pretrain(self):
        """TODO: Factor this into Trainer."""
    
        ## Pre-train layer-wise
        corruption_levels = [(i + 1) * 0.1 for i in xrange(self.classifier.n_layers)]
        layer_epoch_costs = []
        for layer_index in xrange(self.classifier.n_layers):
            # go through pretraining epochs
            epoch_costs = []
            epoch = 0
            while epoch < self.pretraining_epochs:
                # go through the training set
                layer_costs = []
                for batch_index in xrange(self.n_train_batches):
                    layer_costs.append(
                        self.pretraining_fns[layer_index](
                            minibatch_index=batch_index,
                            corruption=corruption_levels[layer_index],
                            lr=self.pretrain_lr
                        )
                    )
                # print 'Pretraining layer %d, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
                epoch_costs.append(numpy.mean(layer_costs))
                epoch += 1
            layer_epoch_costs.append(epoch_costs)
        return layer_epoch_costs
    

    def train(self, patience, patience_increase = 2.0, improvement_threshold = 0.995):
        """docstring for train"""
        patience = 10 * self.n_train_batches  # look as this many examples regardless
        return super(StackedDenoisingAutoencoderTrainer, self).train(patience, patience_increase, improvement_threshold)
    

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

        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')
        minibatch_index = Tensor.lscalar('minibatch_index')

        self.training_function = self.compiled_training_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs,
            finetune_lr
        )
        self.validation_eval_function = self.compiled_validation_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs
        )
        self.test_eval_function = self.compiled_test_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs
        )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Demonstrate Stacked Denoising Autoencoder'
    )
    argparser.add_argument(
        '--pretraining-epochs',
        dest='pretraining_epochs',
        type=int,
        default='15',
        help="number of epochs to run the pre-training (default: 15)"
    )
    argparser.add_argument(
        '--training-epochs',
        dest='training_epochs',
        type=int,
        default='1000',
        help="number of epochs to run the training (default: 1000)"
    )
    
    dataset = DataSet()
    dataset.load()
    trainer = StackedDenoisingAutoencoderTrainer(
        dataset,
        pretraining_epochs=argparser.parse_args().pretraining_epochs,
        n_epochs=argparser.parse_args().training_epochs
    )
    trainer.preinitialize()
    start_time = time.clock()
    state = trainer.start_pretraining()
    while (trainer.continue_pretraining(state)):
        print(
            'Pretraining layer %d epoch %d, cost %f'
            % (
                state.layer_epoch_combos[state.layer_epoch_index - 1][0],
                state.layer_epoch_combos[state.layer_epoch_index - 1][1],
                state.layer_epoch_costs[-1]
            )
        ) 
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    trainer.initialize()
    start_time = time.clock()
    state = trainer.start_training()
    while (trainer.continue_training(state)):
        print (
            'Training epoch %d, validation error %f%%'
            % (state.epoch, state.epoch_losses[-1][0] * 100.0)
        )
    end_time = time.clock()
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(('Optimization complete with best validation score of %f%%,'
           'with test performance %f%%') %
                 (best_validation_loss * 100., test_score * 100.))
