import numpy
import theano
import theano.tensor as Tensor

class Classifier(object):
    """docstring for Classifier"""


    def __init__(self):
        super(Classifier, self).__init__()


    def initialize_biases(self, output_units, biases, name):
        """docstring for initialize_biases"""

        if biases is None:
            biases_values = numpy.zeros(
                (output_units,), 
                dtype=theano.config.floatX
            )
            self.biases = theano.shared(
                value=biases_values, 
                name=name, 
                borrow=True
            )
        else:
            self.biases = biases


    def params_gradient(self, inputs, outputs):
        """
        compute the gradient of cost with respect to theta (stored in params).
        the resulting gradients will be stored in a list gparams
        """
        
        return [
            Tensor.grad(self.cost_function(inputs, outputs), param) 
            for param in self.parameters
        ]


    def updates(self, inputs, outputs, learning_rate):
        """
        specify how to update the parameters of the model as a list of
        (variable, update expression) pairs given two list the zip A = [a1, a2,
        a3, a4] and B = [b1, b2, b3, b4] of same length, zip generates a list C
        of same size, where each element is a pair formed from the two lists :
        C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

        train_model is a function that updates the model parameters by
        SGD Since this model has many parameters, it would be tedious to
        manually create an update rule for each model parameter. We thus
        create the updates list by automatically looping over all
        (params[i],grads[i]) pairs.
        """
        return [
            (param, param - learning_rate * gparam) 
            for param, gparam in zip(
                self.parameters,
                self.params_gradient(inputs, outputs)
            )
        ]


    def output_probabilities_function(self, input):
        """
        Simple classifiers will implement this, but compoosite classifiers
        will use their components' implementations.
        """
        raise NotImplementedError()


    def cost_function(self, inputs, outputs):
        """
        Composite classifiers will implement this using the implementation of
        the top layer (typically LogisticRegression)
        """
        raise NotImplementedError()


    def evaluation_function(self, inputs, outputs):
        """
        Composite classifiers will implement this using the implementation of
        the top layer (typically LogisticRegression)
        """
        raise NotImplementedError()