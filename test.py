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
        lenet5 = ConvolutionalMultilayerPerceptron(self.dataset, n_epochs = 1, batch_size = 2)
        lenet5.initialize(nkerns = [2, 5])
        epoch_losses, test_score = lenet5.evaluate()
        self.assertEqual(epoch_losses, [[0.52000000000000002, 49]])
        self.assertEqual(test_score, 0.45000000000000001)
        
    def test_deep_belief_network(self):
        dbn = DeepBeliefNetwork(self.dataset, batch_size = 2, pretraining_epochs = 1, training_epochs = 1)
        dbn.initialize()
        
        layer_epoch_costs = dbn.pretrain()
        self.assertEqual(layer_epoch_costs, [[-229.57465974291591], [-724.56407666785856], [-237.06892045897598]])
        
        best_validation_loss, best_iter, test_score = dbn.train()
        self.assertEqual(best_validation_loss, 0.79)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.76)

    def test_denoising_autoencoder(self):
        da = DenoisingAutoencoder(self.dataset, training_epochs = 1, batch_size = 2)
        uncorrupt_costs, corrupt_costs = da.evaluate()
        self.assertEqual(uncorrupt_costs, [149.16503228187111])
        self.assertEqual(corrupt_costs, [173.66499408829787])
        
    def test_logistic_stochastic_gradient_descent(self):
        lc = LogisticClassifier(self.dataset, batch_size = 2, n_epochs = 1)
        lc.initialize()
        epoch_losses, best_validation_loss, best_iter, test_score = lc.train()
        self.assertEqual(epoch_losses, [[0.40000000000000002, 49]])
        self.assertEqual(test_score, 0.30)

    def test_multilayer_perceptron(self):
        mp = MultilayerPerceptron(self.dataset, n_epochs = 1, batch_size = 2)
        mp.initialize()
        epoch_losses, test_score = mp.train()
        self.assertEqual(epoch_losses, [[0.54, 49]])
        self.assertEqual(test_score, 0.52)
        
    def test_restricted_boltzmann_machine(self):
        rbm = RestrictedBoltzmannMachine(self.dataset, training_epochs = 1, batch_size = 2)
        rbm.initialize(n_chains = 2, n_samples = 2, n_hidden = 5)
        epoch_costs, plotting_time = rbm.train()
        self.assertEqual(epoch_costs, [-174.86070176730175])

    def test_stacked_denoising_autoencoder(self):
        sda = StackedDenoisingAutoencoder(self.dataset, pretraining_epochs = 1, training_epochs = 1, batch_size = 2)
        sda.preinitialize()
        layer_epoch_costs = sda.pretrain()
        self.assertEqual(layer_epoch_costs, [[328.15852933515004], [771.56755018914123], [661.65193991637716]])
        sda.initialize()
        validation_losses, best_validation_loss, best_iter, test_score = sda.train()
        self.assertEqual(validation_losses, [[0.73, 49]])
        self.assertEqual(best_validation_loss, 0.73)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.67)
        
if __name__ == '__main__':
    unittest.main()
