import numpy
import theano
import theano.tensor as Tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class PoolingLayer(object):
    """Pool Layer of a convolutional network """


    def __fan_in(self):
        """
        there are "num input feature maps * filter height * filter width"
        inputs to each hidden unit
        """
        return numpy.prod(self.filter_shape[1:])


    def __fan_out(self):
        """
        each unit in the lower layer receives a gradient from:
        "num output feature maps * filter height * filter width" /
        pooling size
        """
        return (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) /
                   numpy.prod(self.pool_size))


    def __W_bound(self):
        """docstring for __W_bound"""
        return numpy.sqrt(6. / (self.__fan_in() + self.__fan_out()))


    def initialize_weights(self, rng):
        """docstring for initialize_weights"""

        # initialize weights with random weights
        self.weights = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__W_bound(), high=self.__W_bound(), size=self.filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )


    def initialize_biases(self):
        """docstring for initialize_biases"""

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.biases = theano.shared(value=b_values, borrow=True)
    
        

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = poolsize
        
        self.initialize_weights(rng)
        self.initialize_biases()

        # store parameters of this layer
        self.parameters = [self.weights, self.biases]


    def output_probabilities_function(self, inputs):
        """docstring for output_probabilities_function"""
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=inputs, filters=self.weights,
                filter_shape=self.filter_shape, image_shape=self.image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return Tensor.tanh(pooled_out + self.biases.dimshuffle('x', 0, 'x', 'x'))
