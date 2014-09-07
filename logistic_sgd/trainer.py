import numpy
import theano
import theano.tensor

class DataModel(object):
    def __init__(self, index, input_set, output_set, classifier, input, n_batches):
        self.loss_function = theano.function(
            inputs = [index],
            outputs = classifier.errors(
                input_set, 
                output_set
            ), 
            givens = {
                input: input_set
            }
        )
        self.n_batches = n_batches
        
    def loss(self):
        return numpy.mean([self.loss_function(i) for i in xrange(self.n_batches)])

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, classifier, datasets, learning_rate):
        super(Trainer, self).__init__()
        self.classifier = classifier
        self.datasets = datasets
        self.index = theano.tensor.lscalar()  # index to a [mini]batch
        self.input = theano.tensor.matrix('input')  # the data is presented as rasterized images
        self.output = theano.tensor.ivector('output')  # the labels are presented as 1D vector of [int] labels
        
        self.test = DataModel(
            self.index, 
            datasets.test_set_input[self.index * datasets.batch_size: (self.index + 1) * datasets.batch_size], 
            datasets.test_set_output[self.index * datasets.batch_size: (self.index + 1) * datasets.batch_size], 
            classifier, 
            self.input, 
            datasets.test_set_input.get_value(borrow=True).shape[0] / datasets.batch_size
        )
        
        self.validation = DataModel(
            self.index, 
            datasets.valid_set_input[self.index * datasets.batch_size: (self.index + 1) * datasets.batch_size], 
            datasets.valid_set_output[self.index * datasets.batch_size: (self.index + 1) * datasets.batch_size], 
            classifier, 
            self.input, 
            datasets.valid_set_input.get_value(borrow=True).shape[0] / datasets.batch_size
        )

        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train = theano.function(
            inputs = [self.index],
            outputs = classifier.negative_log_likelihood(self.input, self.output),
            # specify how to update the parameters of the model as a list of
            # (variable, update expression) pairs.
            updates = [
                (
                    classifier.weights, 
                    classifier.weights - learning_rate * theano.tensor.grad(cost = classifier.negative_log_likelihood(self.input, self.output), wrt = classifier.weights)
                ),
                (
                    classifier.biases, 
                    classifier.biases - learning_rate * theano.tensor.grad(cost = classifier.negative_log_likelihood(self.input, self.output), wrt = classifier.biases)
                )
            ],
            givens = {
                    self.input: datasets.train_set_input[self.index * datasets.batch_size:(self.index + 1) * datasets.batch_size],
                    self.output: datasets.train_set_output[self.index * datasets.batch_size:(self.index + 1) * datasets.batch_size]
            }
        )