import numpy
import theano

class Classifier(object):
    """docstring for Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        
    def initialize_biases(self, output_units, biases, name):
        """docstring for initialize_biases"""

        if biases is None:
            biases_values = numpy.zeros((output_units,), dtype=theano.config.floatX)
            self.biases = theano.shared(value = biases_values, name=name, borrow=True)
        else:
            self.biases = biases
