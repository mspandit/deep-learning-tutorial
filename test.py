import unittest

from logistic_classifier import sgd_optimization_mnist
from multilayer_perceptron import test_mlp
from convolutional_multilayer_perceptron import evaluate_lenet5
from denoising_autoencoder import test_dA
from stacked_denoising_autoencoder import test_SdA
from restricted_boltzmann_machine import test_rbm
from deep_belief_network import test_DBN
from data_set import DataSet

class TestTutorials(unittest.TestCase):
    """docstring for TestTutorials"""
    @classmethod
    def setUpClass(self):
        self.dataset = DataSet()
        self.dataset.load(100)

    def test_convolutional_multilayer_perceptron(self):
        evaluate_lenet5(self.dataset, n_epochs = 1, batch_size = 2, nkerns = [2, 5])
        
    def test_deep_belief_network(self):
        test_DBN(self.dataset, pretraining_epochs = 1, training_epochs = 1, batch_size = 2)

    def test_denoising_autoencoder(self):
        test_dA(self.dataset, training_epochs = 1, batch_size = 2)
        
    def test_logistic_stochastic_gradient_descent(self):
        sgd_optimization_mnist(self.dataset, n_epochs = 1, batch_size = 2)

    def test_multilayer_perceptron(self):
        test_mlp(self.dataset, n_epochs = 1, batch_size = 2)
        
    def test_restricted_boltzmann_machine(self):
        test_rbm(self.dataset, training_epochs = 1, batch_size = 2, n_chains = 2, n_samples = 2, n_hidden = 5)

    def test_stacked_denoising_autoencoder(self):
        test_SdA(self.dataset, pretraining_epochs = 1, training_epochs = 1, batch_size = 2)
        
if __name__ == '__main__':
    unittest.main()
