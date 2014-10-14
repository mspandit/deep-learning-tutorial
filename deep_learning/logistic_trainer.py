import theano
import theano.tensor as Tensor

from logistic_classifier import LogisticClassifier
from trainer import Trainer

class LogisticTrainer(Trainer):
    """docstring for LogisticClassifier"""
    def __init__(self, dataset, batch_size = 600, n_epochs = 1000):
        """
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        """
        super(LogisticTrainer, self).__init__(dataset, batch_size, n_epochs)
    
    def initialize(self, learning_rate = 0.13):
        """
        Demonstrate stochastic gradient descent optimization of a log-linear
        model

        This is demonstrated on MNIST.

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
        """

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        # allocate symbolic variables for the data
        index = Tensor.lscalar()  # index to a [mini]batch
        inputs = Tensor.matrix('inputs')  # the data is presented as rasterized images
        outputs = Tensor.ivector('outputs')  # the labels are presented as 1D vector of
                               # [int] labels

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticClassifier(n_in = 28 * 28, n_out = 10)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.test_eval_function = theano.function(
            inputs = [index],
            outputs = classifier.errors(inputs, outputs),
            givens = {
                inputs: self.dataset.test_set_input[index * self.batch_size: (index + 1) * self.batch_size],
                outputs: self.dataset.test_set_output[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.validation_eval_function = theano.function(
            inputs = [index],
            outputs = classifier.errors(inputs, outputs),
            givens = {
                inputs: self.dataset.valid_set_input[index * self.batch_size:(index + 1) * self.batch_size],
                outputs: self.dataset.valid_set_output[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        # compute the gradient of cost with respect to theta = (W,b)

        # compiling a Theano function `train_model` that updates the parameter of the model based on the rules
        # defined in `updates`
        self.training_function = theano.function(
            inputs = [index],
            updates = classifier.updates(inputs, outputs, learning_rate),
            givens = {
                inputs: self.dataset.train_set_input[index * self.batch_size:(index + 1) * self.batch_size],
                outputs: self.dataset.train_set_output[index * self.batch_size:(index + 1) * self.batch_size]
            }
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
        print 'epoch %d, validation error %f%%' % (epoch_index, epoch_losses[epoch_index][0] * 100.0)
        
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100.0, test_score * 100.))
    