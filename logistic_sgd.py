"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`weights` and a bias vector :math:`biases`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|input, weights,biases) &= softmax_i(weights input + biases) \\
                &= \frac {e^{weights_i input + biases_i}} {\sum_j e^{weights_j input + biases_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|input).

.. math::

  output_{pred} = argmax_i P(Y=i|input,weights,biases)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`weights`
    and bias vector :math:`biases`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.weights = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='weights', borrow=True)
        # initialize the biases as a vector of n_out 0s
        self.biases = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='biases', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_output_given_input = theano.tensor.nnet.softmax(theano.tensor.dot(input, self.weights) + self.biases)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.output_pred = theano.tensor.argmax(self.p_output_given_input, axis=1)

    def negative_log_likelihood(self, output):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{weights,biases\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=output^{(i)}|input^{(i)}, weights,biases)) \\
                \ell (\theta=\{weights,biases\}, \mathcal{D})

        :type output: theano.tensor.TensorType
        :param output: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # output.shape[0] is (symbolically) the number of rows in output, i.e.,
        # number of examples (call it n) in the minibatch
        # theano.tensor.arange(output.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] theano.tensor.log(self.p_output_given_input) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[theano.tensor.arange(output.shape[0]),output] is a vector
        # v containing [LP[0,output[0]], LP[1,output[1]], LP[2,output[2]], ...,
        # LP[n-1,output[n-1]]] and theano.tensor.mean(LP[theano.tensor.arange(output.shape[0]),output]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -theano.tensor.mean(theano.tensor.log(self.p_output_given_input)[theano.tensor.arange(output.shape[0]), output])

    def errors(self, output):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type output: theano.tensor.TensorType
        :param output: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if output has same dimension of output_pred
        if output.ndim != self.output_pred.ndim:
            raise TypeError('output should have the same shape as self.output_pred',
                ('output', target.type, 'output_pred', self.output_pred.type))
        # check if output is of the correct datatype
        if output.dtype.startswith('int'):
            # the theano.tensor.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return theano.tensor.mean(theano.tensor.neq(self.output_pred, output))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_inputoutput, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_input, data_output = data_inputoutput
        shared_input = theano.shared(numpy.asarray(data_input,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_output = theano.shared(numpy.asarray(data_output,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_output`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_output`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_input, theano.tensor.cast(shared_output, 'int32')

    test_set_input, test_set_output = shared_dataset(test_set)
    valid_set_input, valid_set_output = shared_dataset(valid_set)
    train_set_input, train_set_output = shared_dataset(train_set)

    rval = [(train_set_input, train_set_output), (valid_set_input, valid_set_output),
            (test_set_input, test_set_output)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_input, train_set_output = datasets[0]
    valid_set_input, valid_set_output = datasets[1]
    test_set_input, test_set_output = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_input.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_input.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_input.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = theano.tensor.lscalar()  # index to a [mini]batch
    input = theano.tensor.matrix('input')  # the data is presented as rasterized images
    output = theano.tensor.ivector('output')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=input, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(output)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(output),
            givens={
                input: test_set_input[index * batch_size: (index + 1) * batch_size],
                output: test_set_output[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(output),
            givens={
                input: valid_set_input[index * batch_size:(index + 1) * batch_size],
                output: valid_set_output[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (weights,biases)
    g_weights = theano.tensor.grad(cost=cost, wrt=classifier.weights)
    g_biases = theano.tensor.grad(cost=cost, wrt=classifier.biases)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.weights, classifier.weights - learning_rate * g_weights),
               (classifier.biases, classifier.biases - learning_rate * g_biases)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                input: train_set_input[index * batch_size:(index + 1) * batch_size],
                output: train_set_output[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    PATIENCE_INCREASE = 2  # wait this much longer when a new best is
                                  # found
    IMPROVEMENT_THRESHOLD = 0.995  # a relative improvement of this much is
                                  # considered significant
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % min(n_train_batches, patience / 2) == 0: # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       IMPROVEMENT_THRESHOLD:
                        patience = max(patience, iter * PATIENCE_INCREASE)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()