import os
import gzip
import cPickle

import numpy

import theano
import theano.tensor

class DataSet(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def load(self):
        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''

        #############
        # LOAD DATA #
        #############

        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(self.dataset)
        if data_dir == "" and not os.path.isfile(self.dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(os.path.split(__file__)[0], "..", "data", self.dataset)
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                self.dataset = new_path

        if (not os.path.isfile(self.dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)

        print '... loading data'

        # Load the dataset
        f = gzip.open(self.dataset, 'rb')
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

        self.test_set_input, self.test_set_output = shared_dataset(test_set)
        self.valid_set_input, self.valid_set_output = shared_dataset(valid_set)
        self.train_set_input, self.train_set_output = shared_dataset(train_set)
        self.n_train_batches = self.train_set_input.get_value(borrow=True).shape[0] / self.batch_size
