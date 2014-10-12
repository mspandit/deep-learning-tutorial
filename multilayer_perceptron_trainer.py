import time
import numpy
import theano.tensor as Tensor
from trainer import Trainer
from multilayer_perceptron import MultilayerPerceptron
        
class MultilayerPerceptronTrainer(Trainer):
    """docstring for MultilayerPerceptron"""
    def __init__(self, dataset, n_epochs = 1000, batch_size = 20):
        """
        """
        super(MultilayerPerceptronTrainer, self).__init__(dataset, batch_size, n_epochs)

    def initialize(self, learning_rate=0.01, n_hidden=500):
        """
        """

        minibatch_index = Tensor.lscalar()
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        rng = numpy.random.RandomState(1234)

        classifier = MultilayerPerceptron(
            rng=rng,
            input=inputs,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )

        self.test_errors = self.initialize_test_function(
            classifier,
            minibatch_index,
            inputs,
            outputs
        )

        self.validation_errors = self.initialize_validation_function(
            classifier,
            minibatch_index,
            inputs,
            outputs
        )

        # compiling a Theano function `train_model` that updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = self.initialize_training_function(
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
    mlp = MultilayerPerceptronTrainer(dataset)
    mlp.initialize()
    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = mlp.train(patience = 10000, patience_increase = 2, improvement_threshold = 0.995)
    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
