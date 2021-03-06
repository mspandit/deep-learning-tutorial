import os
import time
import Image

import numpy
import theano
import theano.tensor as Tensor
from trainer import Trainer
from trainer import TrainingState
from theano.tensor.shared_randomstreams import RandomStreams

from restricted_boltzmann_machine import RestrictedBoltzmannMachine
from utilities import tile_raster_images


class RestrictedBoltzmannMachineTrainer(Trainer):
    """docstring for RestrictedBoltzmannMachine"""

    def __init__(self, dataset, training_epochs = 15,
                 batch_size = 20):
        super(RestrictedBoltzmannMachineTrainer, self).__init__(
            dataset,
            batch_size,
            training_epochs
        )

    def continue_training(self, state):
        """docstring for continue_training"""
        if state.continue_training():
            # go through the training set
            costs = [
                self.training_function(batch_index)
                for batch_index in xrange(self.n_train_batches)
            ]

            state.epoch_losses.append(numpy.mean(costs))

            state.epoch += 1
            
            return True
        else:
            return False

    def train(self):
        """TODO: Factor this into Trainer"""
        state = TrainingState(
            0,
            None,
            None,
            self.n_train_batches,
            self.n_epochs
        )
        plotting_time = 0.

        while state.continue_training():
            # go through the training set
            costs = [
                self.training_function(batch_index)
                for batch_index in xrange(self.n_train_batches)
            ]

            state.epoch_losses.append(numpy.mean(costs))

            state.epoch += 1
        return state.epoch_losses


    def compiled_training_function(self, classifier, minibatch_index, inputs, persistent_chain, learning_rate):
        """docstring for compiled_training_function"""

        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = classifier.get_cost_updates(
            inputs,
            lr=learning_rate,
            persistent=persistent_chain,
            k=15
        )
        
        return theano.function(
            [minibatch_index], 
            cost,
            updates=updates,
            givens={
                inputs: self.dataset.train_set_input[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ]
            },
            name = 'train_rbm'
        )


    def initialize(
        self,
        learning_rate=0.1,
        n_chains=20,
        n_samples=10,
        output_folder='rbm_plots',
        n_hidden=500
    ):
        """
        """

        minibatch_index = Tensor.lscalar('minibatch_index')
        inputs = Tensor.matrix('inputs')

        rng = numpy.random.RandomState(123)

        self.classifier = RestrictedBoltzmannMachine(
            n_visible=28 * 28,
            n_hidden=n_hidden,
            numpy_rng=rng,
            theano_rng=RandomStreams(rng.randint(2 ** 30))
        )

        self.training_function = self.compiled_training_function(
            self.classifier,
            minibatch_index,
            inputs,
            # initialize storage for the persistent chain (state = hidden
            # layer of chain)
            theano.shared(
                numpy.zeros(
                    (self.batch_size, n_hidden),
                    dtype=theano.config.floatX
                ),
                borrow=True
            ),
            learning_rate
        )


    def sample(self):
        """docstring for sample"""
        #################################
        #     Sampling from the RBM     #
        #################################
        # find out the number of test samples
        number_of_test_samples = self.dataset.test_set_input.get_value(borrow=True).shape[0]

        # pick random test examples, with which to initialize the persistent chain
        test_idx = rng.randint(number_of_test_samples - n_chains)
        persistent_vis_chain = theano.shared(numpy.asarray(
                self.dataset.test_set_input.get_value(borrow=True)[test_idx:test_idx + n_chains],
                dtype=theano.config.floatX))

        plot_every = 1000
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `plot_every` steps before returning the
        # sample for plotting
        [presig_hids, hid_mfs, hid_samples, presig_vis,
         vis_mfs, vis_samples], updates =  \
                            theano.scan(self.classifier.gibbs_vhv,
                                    outputs_info=[None,  None, None, None,
                                                  None, persistent_vis_chain],
                                    n_steps=plot_every)

        # add to updates the shared variable that takes care of our persistent
        # chain :.
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                    updates=updates,
                                    name='sample_fn')

        # create a space to store the image for plotting ( we need to leave
        # room for the tile_spacing as well)
        image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
                                 dtype='uint8')
        for idx in xrange(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()
            image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                    X=vis_mf,
                    img_shape=(28, 28),
                    tile_shape=(1, n_chains),
                    tile_spacing=(1, 1))
            # construct image

        image = Image.fromarray(image_data)
        image.save('samples.png')
        os.chdir('../')


from data_set import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    rbm = RestrictedBoltzmannMachineTrainer(dataset)

    if not os.path.isdir('rbm_plots'):
        os.makedirs('rbm_plots')
    os.chdir('rbm_plots')

    rbm.initialize()

    start_time = time.clock()
    plotting_time = 0
    state = rbm.start_training()
    while rbm.continue_training(state):
        plotting_start = time.clock()
        image = Image.fromarray(
            tile_raster_images(
                X=state.classifier.weights.get_value(borrow = True).T,
                img_shape=(28, 28), 
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % state.epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)
    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
