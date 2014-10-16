import unittest

from logistic_trainer import LogisticTrainer
from perceptron_trainer import MultilayerPerceptronTrainer
from convolutional_trainer import ConvolutionalMultilayerPerceptronTrainer
from denoising_autoencoder_trainer import DenoisingAutoencoderTrainer
from stacked_denoising_autoencoder_trainer import StackedDenoisingAutoencoderTrainer
from restricted_boltzmann_machine_trainer import RestrictedBoltzmannMachineTrainer
from deep_belief_trainer import DeepBeliefNetworkTrainer
from data_set import DataSet

class TestTutorials(unittest.TestCase):
    """docstring for TestTutorials"""
    @classmethod
    def setUpClass(self):
        self.dataset = DataSet()
        self.dataset.load(100)

    def test_convolutional_multilayer_perceptron(self):
        lenet5 = ConvolutionalMultilayerPerceptronTrainer(self.dataset, n_epochs = 1, batch_size = 2)
        lenet5.initialize(nkerns = [2, 5])
        epoch_losses, best_validation_loss, best_iter, test_score = lenet5.train(patience = 10000, patience_increase = 2, improvement_threshold = 0.995)
        self.assertEqual(epoch_losses, [[0.52000000000000002, 49]])
        self.assertEqual(test_score, 0.45000000000000001)

    def test_convolutional_multilayer_perceptron_incremental(self):
        lenet5 = ConvolutionalMultilayerPerceptronTrainer(self.dataset, n_epochs = 1, batch_size = 2)
        lenet5.initialize(nkerns = [2, 5])
        state = lenet5.start_training(patience = 10000, patience_increase = 2, improvement_threshold = 0.995)
        while lenet5.continue_training(state):
            pass
        self.assertEqual(state.epoch_losses, [[0.52000000000000002, 49]])
        self.assertEqual(state.test_score, 0.45000000000000001)
        
    def test_deep_belief_network(self):
        dbn = DeepBeliefNetworkTrainer(self.dataset, batch_size = 2, pretraining_epochs = 1, training_epochs = 1)
        dbn.initialize()
        
        layer_epoch_costs = dbn.pretrain()
        self.assertTrue(layer_epoch_costs[0][0] > -229.574659742916 and layer_epoch_costs[0][0] < -229.574659742915)
        self.assertTrue(layer_epoch_costs[1][0] > -724.564076667859 and layer_epoch_costs[1][0] < -724.564076667856)
        self.assertTrue(layer_epoch_costs[2][0] > -237.068920458976 and layer_epoch_costs[2][0] < -237.068920458975)
        
        epoch_losses, best_validation_loss, best_iter, test_score = dbn.train()
        self.assertEqual(best_validation_loss, 0.79)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.76)

    def test_denoising_autoencoder(self):
        da = DenoisingAutoencoderTrainer(self.dataset, training_epochs = 1, batch_size = 2)
        da.initialize()
        uncorrupt_costs = da.train()
        self.assertEqual(uncorrupt_costs, [149.16503228187111])
        da.initialize(corruption_level = 0.3)
        corrupt_costs = da.train()
        self.assertTrue(
            corrupt_costs[0] > 173.6649940882978 
            and corrupt_costs[0] < 173.6649940882979
        )

    def test_denoising_autoencoder_incremental(self):
        da = DenoisingAutoencoderTrainer(self.dataset, training_epochs = 1, batch_size = 2)
        da.initialize()
        state = da.start_training()
        while da.continue_training(state):
            pass
        self.assertEqual(state.costs, [149.16503228187111])
        da.initialize(corruption_level = 0.3)
        state = da.start_training()
        while da.continue_training(state):
            pass
        self.assertTrue(
            state.costs[0] > 173.6649940882978
            and state.costs[0] < 173.6649940882979
        )
        
    def test_logistic(self):
        lc = LogisticTrainer(self.dataset, batch_size = 2, n_epochs = 1)
        lc.initialize()
        epoch_losses, best_validation_loss, best_iter, test_score = lc.train(patience = 5000, patience_increase = 2, improvement_threshold = 0.995)
        self.assertEqual(epoch_losses, [[0.40000000000000002, 49]])
        self.assertEqual(test_score, 0.30)
        
    def test_logistic_incremental(self):
        lc = LogisticTrainer(self.dataset, batch_size=2, n_epochs=1)
        lc.initialize()
        state = lc.start_training(patience=5000, patience_increase=2, improvement_threshold=0.995)
        while lc.continue_training(state):
            pass
        self.assertEqual(state.epoch_losses, [[0.40000000000000002, 49]])
        self.assertEqual(state.test_score, 0.30)

    def test_multilayer_perceptron(self):
        mp = MultilayerPerceptronTrainer(self.dataset, n_epochs = 1, batch_size = 2)
        mp.initialize()
        epoch_losses, best_validation_loss, best_iter, test_score = mp.train(patience = 10000, patience_increase = 2, improvement_threshold = 0.995)
        self.assertEqual(epoch_losses, [[0.54, 49]])
        self.assertEqual(test_score, 0.52)

    def test_multilayer_perceptron_incremental(self):
        mp = MultilayerPerceptronTrainer(self.dataset, n_epochs = 1, batch_size = 2)
        mp.initialize()
        state = mp.start_training(patience = 10000, patience_increase = 2, improvement_threshold = 0.995)
        while mp.continue_training(state):
            pass
        self.assertEqual(state.epoch_losses, [[0.54, 49]])
        self.assertEqual(state.test_score, 0.52)
        
    def test_restricted_boltzmann_machine(self):
        rbm = RestrictedBoltzmannMachineTrainer(self.dataset, training_epochs = 1, batch_size = 2)
        rbm.initialize(n_chains = 2, n_samples = 2, n_hidden = 5)
        epoch_costs = rbm.train()
        self.assertEqual(epoch_costs, [-174.86070176730175])
        
    def test_restricted_boltzmann_machine_incremental(self):
        rbm = RestrictedBoltzmannMachineTrainer(self.dataset, training_epochs = 1, batch_size = 2)
        rbm.initialize(n_chains = 2, n_samples = 2, n_hidden = 5)
        state = rbm.start_training()
        while rbm.continue_training(state):
            pass
        self.assertEqual(state.epoch_losses, [-174.86070176730175])

    def test_stacked_denoising_autoencoder(self):
        sda = StackedDenoisingAutoencoderTrainer(self.dataset, pretraining_epochs = 1, n_epochs = 1, batch_size = 2)
        sda.preinitialize()
        layer_epoch_costs = sda.pretrain()
        self.assertEqual(layer_epoch_costs, [[328.15852933515004], [771.56755018914123], [661.65193991637716]])
        sda.initialize()
        epoch_losses, best_validation_loss, best_iter, test_score = sda.train(None)
        self.assertEqual(epoch_losses, [[0.73, 49]])
        self.assertEqual(best_validation_loss, 0.73)
        self.assertEqual(best_iter, 49)
        self.assertEqual(test_score, 0.67)

    def test_stacked_denoising_autoencoder_incremental(self):
        sda = StackedDenoisingAutoencoderTrainer(self.dataset, pretraining_epochs = 1, n_epochs = 1, batch_size = 2)
        sda.preinitialize()
        state = sda.start_pretraining()
        while sda.continue_pretraining(state):
            pass
        self.assertEqual(state.layer_epoch_costs, [[328.15852933515004], [771.56755018914123], [661.65193991637716]])
        sda.initialize()
        state = sda.start_training()
        while sda.continue_training(state):
            pass
        self.assertEqual(state.epoch_losses, [[0.73, 49]])
        self.assertEqual(state.best_validation_loss, 0.73)
        self.assertEqual(state.best_iter, 49)
        self.assertEqual(state.test_score, 0.67)
        
if __name__ == '__main__':
    unittest.main()
