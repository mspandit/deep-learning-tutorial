import unittest

from logistic_classifier import LogisticClassifier
from multilayer_perceptron import MultilayerPerceptron
from convolutional_multilayer_perceptron import ConvolutionalMultilayerPerceptron
from denoising_autoencoder import DenoisingAutoencoder
from stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from restricted_boltzmann_machine import RestrictedBoltzmannMachine
from deep_belief_network import DeepBeliefNetwork
from data_set import DataSet

class TestTutorials(unittest.TestCase):
    """docstring for TestTutorials"""
    @classmethod
    def setUpClass(self):
        self.dataset = DataSet()
        self.dataset.load(100)

    def test_convolutional_multilayer_perceptron(self):
        lenet5 = ConvolutionalMultilayerPerceptron(self.dataset)
        lenet5.evaluate(n_epochs = 1, batch_size = 2, nkerns = [2, 5])
        
    def test_deep_belief_network(self):
        dbn = DeepBeliefNetwork(self.dataset)
        dbn.evaluate(pretraining_epochs = 1, training_epochs = 1, batch_size = 2)

    def test_denoising_autoencoder(self):
        da = DenoisingAutoencoder(self.dataset)
        da.evaluate(training_epochs = 1, batch_size = 2)
        
    def test_logistic_stochastic_gradient_descent(self):
        lc = LogisticClassifier(self.dataset)
        lc.evaluate(n_epochs = 1, batch_size = 2)

    def test_multilayer_perceptron(self):
        mp = MultilayerPerceptron(self.dataset)
        mp.evaluate(n_epochs = 1, batch_size = 2)
        
    def test_restricted_boltzmann_machine(self):
        rbm = RestrictedBoltzmannMachine(self.dataset)
        rbm.evaluate(training_epochs = 1, batch_size = 2, n_chains = 2, n_samples = 2, n_hidden = 5)

    def test_stacked_denoising_autoencoder(self):
        sda = StackedDenoisingAutoencoder(self.dataset)
        sda.evaluate(pretraining_epochs = 1, training_epochs = 1, batch_size = 2)
        
if __name__ == '__main__':
    unittest.main()
