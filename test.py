import unittest

from logistic_sgd import sgd_optimization_mnist
from mlp import test_mlp
from convolutional_mlp import evaluate_lenet5
from dA import test_dA
from SdA import test_SdA
from rbm import test_rbm
from DBN import test_DBN

class TestTutorials(unittest.TestCase):
    def test_logistic_stochastic_gradient_descent(self):
        sgd_optimization_mnist(n_epochs = 1)

    def test_multilayer_perceptron(self):
        test_mlp(n_epochs = 1)

    def test_convolutional_multilayer_perceptron(self):
        evaluate_lenet5(n_epochs = 1, batch_size = 1, nkerns = [2, 5])

    def test_denoising_autoencoder(self):
        # test_dA(training_epochs = 1, batch_size = 1)
        pass

    def test_stacked_denoising_autoencoder(self):
        test_SdA(pretraining_epochs = 1, training_epochs = 1, batch_size = 1)
        
    def test_restricted_boltzmann_machine(self):
        test_rbm(training_epochs = 1, batch_size = 2, n_chains = 2, n_samples = 2, n_hidden = 5)
        
    def test_deep_belief_network(self):
        test_DBN(pretraining_epochs = 1, training_epochs = 1)
        
if __name__ == '__main__':
    unittest.main()
