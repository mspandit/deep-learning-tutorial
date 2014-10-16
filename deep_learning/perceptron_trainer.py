import os
import sys
import time
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

        classifier = MultilayerPerceptronClassifier(
            rng=rng,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )

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


from data_set import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    trainer = MultilayerPerceptronTrainer(dataset, n_epochs=1000)
    trainer.initialize()
    trainer.start_training(
        patience=10000,
        patience_increase=2,
        improvement_threshold=0.995
    )
    start_time = time.clock()
    while (trainer.continue_training()):
        print (
            'epoch %d, validation error %f%%'
            % (trainer.epoch, trainer.epoch_losses[-1][0] * 100.0)
        )
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(
        (
            'Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i, with test performance %f %%'
        ) 
        % (
            trainer.best_validation_loss * 100.,
            trainer.best_iter + 1,
            trainer.test_score * 100.
        )
    )
