import os
import sys
import time
import Image

import numpy
import theano
import theano.tensor as Tensor
from theano.tensor.shared_randomstreams import RandomStreams
from utilities import tile_raster_images

from data_set import DataSet
from trainer import Trainer
from denoising_autoencoder import DenoisingAutoencoder


class DenoisingAutoencoderTrainer(Trainer):
    """docstring for DenoisingAutoencoder"""


    def __init__(self, dataset, training_epochs=15, learning_rate=0.1,
                batch_size=20):
        """
        :type training_epochs: int
        :param training_epochs: number of epochs used for training

        :type learning_rate: float
        :param learning_rate: learning rate used for training the DeNosing
                              AutoEncoder
        """
        super(DenoisingAutoencoderTrainer, self).__init__(
            dataset,
            batch_size,
            training_epochs
        )
        self.learning_rate = learning_rate
        

    def train(self):
        """TODO: Factor this into Trainer"""
        ############
        # TRAINING #
        ############

        # go through training epochs
        costs = []
        epoch = 0
        while epoch < self.n_epochs:
            # go through trainng set
            c = [
                self.train_da(batch_index)
                for batch_index in xrange(self.n_train_batches)
            ]
            # print 'Training epoch %d, cost %f' % (epoch, numpy.mean(c))
            costs.append(numpy.mean(c))
        
            epoch += 1
        return costs


    def build_model(self, corruption_level = 0.0):
        """docstring for build_model_0"""

        minibatch_index = Tensor.lscalar('minibatch_index')
        inputs = Tensor.matrix('denoising_autoencoder_inputs')

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        da = DenoisingAutoencoder(numpy_rng = rng, theano_rng = theano_rng, n_visible = 28 * 28, n_hidden = 500)

        cost = da.cost(inputs, corruption_level = corruption_level)
        updates = da.updates(inputs, corruption_level = corruption_level, learning_rate = self.learning_rate)

        self.train_da = theano.function(
            inputs = [minibatch_index], 
            outputs = cost, 
            updates = updates,
            givens = {
                inputs: self.dataset.train_set_input[minibatch_index * self.batch_size : (minibatch_index + 1) * self.batch_size]
            }
        )
        
        costs = self.train()
        image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save('filters_corruption_0.png')
        
        return costs


    def evaluate(self, output_folder='dA_plots'):
        """
        This demo is tested on MNIST
        """

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
        
        uncorrupt_costs = self.build_model()
        corrupt_costs = self.build_model(corruption_level = 0.3)

        os.chdir('../')
        
        return [uncorrupt_costs, corrupt_costs]


if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    da = DenoisingAutoencoderTrainer(dataset)
    uncorrupt_costs, corrupt_costs = da.evaluate()
