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
        best_validation_loss, best_iter, test_score = lenet5.evaluate(n_epochs = 1, batch_size = 2, nkerns = [2, 5])
        self.assertEqual(best_validation_loss, 0.52000000000000002)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.45000000000000001)
        
    def test_deep_belief_network(self):
        dbn = DeepBeliefNetwork(self.dataset)
        best_validation_loss, best_iter, test_score = dbn.evaluate(pretraining_epochs = 1, training_epochs = 1, batch_size = 2)
        self.assertEqual(best_validation_loss, 0.79)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.76)

    def test_denoising_autoencoder(self):
        da = DenoisingAutoencoder(self.dataset)
        da.evaluate(training_epochs = 1, batch_size = 2)
        
    def test_logistic_stochastic_gradient_descent(self):
        lc = LogisticClassifier(self.dataset)
        best_validation_loss, best_iter, test_score = lc.evaluate(n_epochs = 1, batch_size = 2)
        self.assertEqual(best_validation_loss, 0.40)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.30)

    def test_multilayer_perceptron(self):
        mp = MultilayerPerceptron(self.dataset)
        best_validation_loss, best_iter, test_score = mp.evaluate(n_epochs = 1, batch_size = 2)
        self.assertEqual(best_validation_loss, 0.54)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.52)
        
    def test_restricted_boltzmann_machine(self):
        rbm = RestrictedBoltzmannMachine(self.dataset)
        rbm.evaluate(training_epochs = 1, batch_size = 2, n_chains = 2, n_samples = 2, n_hidden = 5)

    def test_stacked_denoising_autoencoder(self):
        sda = StackedDenoisingAutoencoder(self.dataset)
        best_validation_loss, best_iter, test_score = sda.evaluate(pretraining_epochs = 1, training_epochs = 1, batch_size = 2)
        self.assertEqual(best_validation_loss, 0.73)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.67)
        
if __name__ == '__main__':
    unittest.main()
