import numpy
import theano.tensor as Tensor
from pooling_layer import PoolingLayer
from hidden_layer import HiddenLayer
from classifier import Classifier
from logistic_classifier import LogisticClassifier


class ConvolutionalMultilayerPerceptronClassifier(Classifier):
    """docstring for ConvolutionalMultilayerPerceptronClassifier"""

    def __init__(self, batch_size, nkerns=[20, 50]):
        """
        """
        super(ConvolutionalMultilayerPerceptronClassifier, self).__init__()

        self.batch_size = batch_size
        rng = numpy.random.RandomState(23455)

        # Reshape matrix of rasterized images of shape (self.batch_size,28*28)
        # to a 4D tensor, compatible with our PoolingLayer

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (self.batch_size,nkerns[0],12,12)
        self.layer0 = PoolingLayer(
            rng,
            image_shape=(self.batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        self.layer1 = PoolingLayer(
            rng,
            image_shape=(self.batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (self.batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input_units=nkerns[1] * 4 * 4,
            output_units=500,
            nonlinear_function=Tensor.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticClassifier(input_units=500, output_units=10)

        # create a list of all model parameters to be fit by gradient descent
        self.parameters = (
            self.layer3.parameters
            + self.layer2.parameters
            + self.layer1.parameters
            + self.layer0.parameters
        )

    def cost_function(self, inputs, outputs):
        """docstring for cost_function"""
        prev_outputs = inputs.reshape((self.batch_size, 1, 28, 28))
        prev_outputs = self.layer0.output_probabilities_function(
            prev_outputs
        )
        prev_outputs = self.layer1.output_probabilities_function(
            prev_outputs
        ).flatten(2)
        prev_outputs = self.layer2.output_probabilities_function(
            prev_outputs
        )
        return self.layer3.cost_function(
            prev_outputs,
            outputs
        )

    def evaluation_function(self, inputs, outputs):
        """docstring for evaluation_function"""
        return self.layer3.evaluation_function(
            self.layer2.output_probabilities_function(
                self.layer1.output_probabilities_function(
                    self.layer0.output_probabilities_function(
                        inputs.reshape((self.batch_size, 1, 28, 28))
                    )
                ).flatten(2)
            ),
            outputs
        )
