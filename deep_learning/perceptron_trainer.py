import os
import sys
import time
import argparse

import numpy
import theano.tensor as Tensor
from trainer import Trainer
from perceptron_classifier import MultilayerPerceptronClassifier

class MultilayerPerceptronTrainer(Trainer):
    """docstring for MultilayerPerceptron"""


    def __init__(self, dataset, n_epochs = 1000, batch_size = 20):
        """
        """
        super(MultilayerPerceptronTrainer, self).__init__(
            dataset,
            batch_size,
            n_epochs
        )


    def initialize(self, learning_rate=0.01, n_hidden=500):
        """
        """

        minibatch_index = Tensor.lscalar() 
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        rng = numpy.random.RandomState(1234)

        self.classifier = MultilayerPerceptronClassifier(
            rng=rng,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )

        self.test_eval_function = self.compiled_test_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs
        )
        
        self.validation_eval_function = self.compiled_validation_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs
        )
        
        self.training_function = self.compiled_training_function(
            self.classifier,
            minibatch_index,
            inputs,
            outputs,
            learning_rate
        )


from data_set import DataSet

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Demonstrate Multilayer Perceptron'
    )
    argparser.add_argument(
        '--training-epochs',
        dest='epochs',
        type=int,
        default='1000',
        help='number of epochs to run the training (default: 1000)'
    )

    dataset = DataSet()
    dataset.load()
    trainer = MultilayerPerceptronTrainer(dataset, n_epochs=argparser.parse_args().epochs)
    trainer.initialize()
    state = trainer.start_training(
        patience=10000,
        patience_increase=2,
        improvement_threshold=0.995
    )
    start_time = time.clock()
    while (trainer.continue_training(state)):
        print (
            'epoch %d, validation error %f%%'
            % (state.epoch, state.epoch_losses[-1][0] * 100.0)
        )
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(
        (
            'Optimization complete. Best validation score of %f%% '
            'obtained at iteration %i, with test performance %f%%'
        ) 
        % (
            state.best_validation_loss * 100.,
            state.best_iter + 1,
            state.test_score * 100.
        )
    )
