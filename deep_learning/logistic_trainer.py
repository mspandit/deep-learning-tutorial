import theano
import theano.tensor as Tensor

from logistic_classifier import LogisticClassifier
from trainer import Trainer

class LogisticTrainer(Trainer):
    """docstring for LogisticClassifier"""
    def __init__(self, dataset, batch_size = 600, n_epochs = 1000):
        """
        """
        super(LogisticTrainer, self).__init__(dataset, batch_size, n_epochs)
    
    def initialize(self, learning_rate = 0.13):
        """
        """

        minibatch_index = Tensor.lscalar()
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        # Each MNIST image has size 28*28
        classifier = LogisticClassifier(input_units=28 * 28, output_units=10)

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

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    trainer = LogisticClassifierTrainer(dataset)
    trainer.initialize()

    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = trainer.train()
    end_time = time.clock()
    for epoch_index in xrange(len(epoch_losses)):
        print (
            'epoch %d, validation error %f%%'
            % (epoch_index, epoch_losses[epoch_index][0] * 100.0)
        )
        
    print >> sys.stderr, (
        'The code for file '
        + os.path.split(__file__)[1]
        + ' ran for %.1fs' % ((end_time - start_time))
    )
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100.0, test_score * 100.))
    