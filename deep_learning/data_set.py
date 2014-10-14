import cPickle
import gzip
import os
import theano
import theano.tensor as Tensor
import numpy

class DataSet(object):
    """docstring for DataSet"""


    def __init__(self, filename='mnist.pkl.gz'):
        super(DataSet, self).__init__()
        self.filename = filename


    def load(self, limit=None):
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(self.filename)
        if data_dir == "" and not os.path.isfile(self.filename):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                self.filename = new_path

        if (not os.path.isfile(self.filename)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)

        # Load the dataset
        f = gzip.open(self.filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        if (limit != None):
            train_set = [train_set[0][0:limit], train_set[1][0:limit]]
            valid_set = [valid_set[0][0:limit], valid_set[1][0:limit]]
            test_set = [test_set[0][0:limit], test_set[1][0:limit]]
        
        f.close()
        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.

        def shared_dataset(data_xy, borrow=True):
            """ 
            Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch
            everytime is needed (the default behaviour if the data is not in a
            shared variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(
                numpy.asarray(
                    data_x,
                    dtype=theano.config.floatX
                ),
                borrow=borrow
            )
            shared_y = theano.shared(
                numpy.asarray(
                    data_y,
                    dtype=theano.config.floatX
                ),
                borrow=borrow
            )
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, Tensor.cast(shared_y, 'int32')

        self.test_set_input, self.test_set_output = shared_dataset(test_set)
        self.valid_set_input, self.valid_set_output = shared_dataset(valid_set)
        self.train_set_input, self.train_set_output = shared_dataset(train_set)
