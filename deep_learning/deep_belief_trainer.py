import numpy
import theano.tensor as Tensor
from trainer import Trainer
from deep_belief_network import DBN

class DeepBeliefNetworkTrainer(Trainer):
    """docstring for DeepBeliefNetwork"""
    def __init__(
        self,
        dataset,
        pretraining_epochs=100,
        training_epochs=1000, 
        pretrain_lr=0.01,
        finetune_lr = 0.1, 
        batch_size=10
    ):
        """
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining

        :type training_epochs: int
        :param training_epochs: maximal number of iterations ot run the optimizer

        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage

        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training

        :type batch_size: int
        :param batch_size: the size of a minibatch
        """
        super(DeepBeliefNetworkTrainer, self).__init__(
            dataset,
            batch_size,
            training_epochs
        )
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.dataset.train_set_input.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        
    def pretrain(self):
        """TODO: Factor this into Trainer."""

        layer_epoch_costs = []

        ## Pre-train layer-wise
        for i in xrange(self.dbn.n_layers):
            epoch_costs = []
            # go through pretraining epochs
            for epoch in xrange(self.pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_train_batches):
                    c.append(
                        self.pretraining_fns[i](
                            index = batch_index,
                            lr = self.pretrain_lr
                        )
                    )
                epoch_costs.append(numpy.mean(c))
                # print 'Pre-training layer %d, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
            layer_epoch_costs.append(epoch_costs)
    
        return layer_epoch_costs


    def train(self, patience_increase = 2.0, improvement_threshold = 0.995):
        """docstring for train"""
        # early-stopping parameters
        patience = 4 * self.n_train_batches  # look as this many examples regardless
        return super(DeepBeliefNetworkTrainer, self).train(
            patience,
            patience_increase,
            improvement_threshold
        )


    def initialize(self, k = 1):
        """
        Demonstrates how to train and test a Deep Belief Network.

        This is demonstrated on MNIST.
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        """

        minibatch_index = Tensor.lscalar()
        inputs = Tensor.matrix('inputs')
        outputs = Tensor.ivector('outputs')

        self.dbn = DBN(
            numpy_rng=numpy.random.RandomState(123),
            n_ins=28 * 28,
            hidden_layers_sizes=[1000, 1000, 1000],
            n_outs=10
        )

        self.pretraining_fns = self.dbn.pretraining_functions(
            inputs,
            train_set_input = self.dataset.train_set_input,
            batch_size = self.batch_size,
            k = k
        )

        self.training_function = self.compiled_training_function(
            self.dbn,
            minibatch_index,
            inputs,
            outputs,
            self.finetune_lr
        )
        self.validation_eval_function = self.compiled_validation_function(
            self.dbn,
            minibatch_index,
            inputs,
            outputs
        )
        self.test_eval_function = self.compiled_test_function(
            self.dbn,
            minibatch_index, 
            inputs,
            outputs
        )


from data_set import DataSet

if __name__ == '__main__':
    dataset = DataSet()
    dataset.load()
    dbn = DeepBeliefNetwork(dataset)
    dbn.initialize()

    start_time = time.clock()
    layer_epoch_costs = dbn.pretrain()
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    start_time = time.clock()
    epoch_losses, best_validation_loss, best_iter, test_score = dbn.train()
    end_time = time.clock()
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
