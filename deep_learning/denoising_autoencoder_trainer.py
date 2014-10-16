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

    def start_training(self):
        """docstring for start_training"""
        self.costs = []
        self.epoch = 0

    def continue_training(self):
        """docstring for continue_training"""
        if self.epoch < self.n_epochs:
            c = [
                self.training_function(batch_index)
                for batch_index in xrange(self.n_train_batches)
            ]
            self.costs.append(numpy.mean(c))
            self.epoch += 1
            return True
        else:
            return False

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
                self.training_function(batch_index)
                for batch_index in xrange(self.n_train_batches)
            ]
            # print 'Training epoch %d, cost %f' % (epoch, numpy.mean(c))
            costs.append(numpy.mean(c))
        
            epoch += 1
        return costs


    def compiled_training_function(self, classifier, minibatch_index, inputs, learning_rate, corruption_level):
        """docstring for compiled_training_function"""

        return theano.function(
            inputs=[minibatch_index],
            outputs=classifier.cost(inputs, corruption_level=corruption_level),
            updates=classifier.updates(inputs, corruption_level=corruption_level,
                learning_rate=learning_rate
            ),
            givens={
                inputs: self.dataset.train_set_input[minibatch_index * self.batch_size : (minibatch_index + 1) * self.batch_size]
            }
        )


    def initialize(self, learning_rate=0.1, corruption_level = 0.0):
        """docstring for build_model_0"""

        minibatch_index = Tensor.lscalar('minibatch_index')
        inputs = Tensor.matrix('denoising_autoencoder_inputs')

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        classifier = DenoisingAutoencoder(
            numpy_rng=rng,
            theano_rng=theano_rng,
            n_visible=28 * 28,
            n_hidden=500
        )

        self.training_function = self.compiled_training_function(
            classifier,
            minibatch_index,
            inputs,
            learning_rate,
            corruption_level
        )

        image = Image.fromarray(tile_raster_images(X=classifier.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save('filters_corruption_0.png')


if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    trainer = DenoisingAutoencoderTrainer(dataset)

    if not os.path.isdir('dA_plots'):
        os.makedirs('dA_plots')
    os.chdir('dA_plots')
    
    trainer.initialize()
    trainer.start_training()

    start_time = time.clock()
    while (trainer.continue_training()):
        print 'Training epoch %d, cost %f' % (trainer.epoch, trainer.costs[-1])
    end_time = time.clock()
    print >> sys.stderr, (
        'The code for file '
        + os.path.split(__file__)[1]
        + ' ran for %.2fm'
        % ((end_time - start_time) / 60.)
    )

    trainer.initialize(corruption_level = 0.3)
    trainer.start_training()

    start_time = time.clock()
    while (trainer.continue_training()):
        print 'Training epoch %d, cost %f' % (trainer.epoch, trainer.costs[-1])
    end_time = time.clock()
    print >> sys.stderr, (
        'The code for file '
        + os.path.split(__file__)[1]
        + ' ran for %.2fm'
        % ((end_time - start_time) / 60.)
    )

    os.chdir('../')
