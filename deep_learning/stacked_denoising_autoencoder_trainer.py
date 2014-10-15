import numpy
import theano.tensor as Tensor
from data_set import DataSet
from trainer import Trainer
from stacked_denoising_autoencoder import StackedDenoisingAutoencoder


class StackedDenoisingAutoencoderTrainer(Trainer):
    """docstring for StackedDenoisingAutoencoder"""


    def __init__(self, dataset, pretraining_epochs = 15, n_epochs = 1000, batch_size = 1, pretrain_lr = 0.001):
        """
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining

        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        """
        super(StackedDenoisingAutoencoderTrainer, self).__init__(dataset, batch_size, n_epochs)
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr


    def preinitialize(self):
        """docstring for preinitialize"""
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.dataset.train_set_input.get_value(borrow=True).shape[0] / self.batch_size

        # numpy random generator
        self.numpy_rng = numpy.random.RandomState(89677)

        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        # construct the stacked denoising autoencoder class
        self.sda = StackedDenoisingAutoencoder(
            numpy_rng = self.numpy_rng, 
            n_ins = 28 * 28,
            hidden_layers_sizes = [1000, 1000, 1000],
            n_outs=10
        )

        #########################
        # PRETRAINING THE MODEL #
        #########################
        self.pretraining_fns = self.sda.pretraining_functions(
            train_set_input=self.dataset.train_set_input,
            batch_size=self.batch_size,
            inputs=inputs,
            outputs=outputs
        )
    

    def pretrain(self):
        """TODO: Factor this into Trainer."""
    
        ## Pre-train layer-wise
        corruption_levels = [.1, .2, .3]
        layer_epoch_costs = []
        for layer_index in xrange(self.sda.n_layers):
            # go through pretraining epochs
            epoch_costs = []
            epoch = 0
            while epoch < self.pretraining_epochs:
                # go through the training set
                layer_costs = []
                for batch_index in xrange(self.n_train_batches):
                    layer_costs.append(
                        self.pretraining_fns[layer_index](
                            index = batch_index,
                            corruption = corruption_levels[layer_index],
                            lr = self.pretrain_lr
                        )
                    )
                # print 'Pretraining layer %d, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
                epoch_costs.append(numpy.mean(layer_costs))
                epoch += 1
            layer_epoch_costs.append(epoch_costs)
        return layer_epoch_costs
    

    def train(self, patience, patience_increase = 2.0, improvement_threshold = 0.995):
        """docstring for train"""
        patience = 10 * self.n_train_batches  # look as this many examples regardless
        return super(StackedDenoisingAutoencoderTrainer, self).train(patience, patience_increase, improvement_threshold)
    

    def initialize(self, finetune_lr=0.1):
        """
        Demonstrates how to train and test a stochastic denoising autoencoder.

        This is demonstrated on MNIST.

        :type learning_rate: float
        :param learning_rate: learning rate used in the finetune stage
        (factor for the stochastic gradient)

        :type n_iter: int
        :param n_iter: maximal number of iterations ot run the optimizer
        """

        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        # get the training, validation and testing function for the model
        self.training_function, self.validation_eval_function, self.test_eval_function = self.sda.build_finetune_functions(
            dataset=self.dataset, 
            batch_size=self.batch_size,
            inputs=inputs,
            outputs=outputs,
            learning_rate=finetune_lr
        )
        

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    sda = StackedDenoisingAutoencoderTrainer(dataset)
    sda.preinitialize()
    start_time = time.clock()
    layer_epoch_costs = sda.pretrain()
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    sda.initialize()
    start_time = time.clock()
    epoch_validation_losses, best_validation_loss, best_iter, test_score = sda.train(None)
    end_time = time.clock()
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
