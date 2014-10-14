import numpy
import theano
import theano.tensor as Tensor
from classifier import Classifier

class HiddenLayer(Classifier):
    
    def initialize_weights(self, input_units, output_units, weights, name):
        """docstring for initialize_weights"""

        if weights is None:
            weights_values = numpy.asarray(
                self.rng.uniform(
                    low = -numpy.sqrt(6.0 / (input_units + output_units)),
                    high = numpy.sqrt(6.0 / (input_units + output_units)),
                    size = (input_units, output_units)
                ), 
                dtype = theano.config.floatX
            )
            if self.nonlinear_function == theano.tensor.nnet.sigmoid:
                weights_values *= 4

            self.weights = theano.shared(
                value=weights_values,
                name='name',
                borrow=True
            )
        else:
            self.weights = weights
    
    def output_probabilities_function(self, inputs):
        """docstring for output_probabilities_function"""
        
        if (self.nonlinear_function is None):
            return Tensor.nnet.softmax(
                Tensor.dot(inputs, self.weights) + self.biases
            )
        else:
            return self.nonlinear_function(
                Tensor.dot(inputs, self.weights) + self.biases
            )

    def __init__(
        self,
        rng,
        input_units,
        output_units,
        weights=None,
        biases=None,
        nonlinear_function=Tensor.tanh
    ):
        super(HiddenLayer, self).__init__()

        self.rng = rng
        self.nonlinear_function = nonlinear_function
        self.initialize_weights(input_units, output_units, weights, 'hidden_layer_weights')
        self.initialize_biases(output_units, biases, 'hidden_layer_biases')
        self.parameters = [self.weights, self.biases]