import theano
import numpy

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, dataset, batch_size, n_epochs):
        super(Trainer, self).__init__()
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_valid_batches = self.dataset.valid_set_input.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        self.n_test_batches = self.dataset.test_set_input.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        self.n_train_batches = self.dataset.train_set_input.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        
    def train(
        self,
        patience=5000,
        patience_increase=2,
        improvement_threshold=0.995
    ):
        """docstring for train"""
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        validation_frequency = min(self.n_train_batches, patience / 2)
        epoch = 0
        epoch_losses = []
        done_looping = False
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
                self.training_function(minibatch_index)
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = self.mean_validation_loss()
                    print (
                        'epoch %d validation error %f%%'
                        % (epoch, this_validation_loss * 100.0)
                    )
                    epoch_losses.append([this_validation_loss, iter])
                    if this_validation_loss < best_validation_loss:
                        if (
                            this_validation_loss
                            < best_validation_loss * improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        test_score = self.mean_test_loss()

                if patience <= iter:
                    done_looping = True
                    break
        return [epoch_losses, best_validation_loss, best_iter, test_score]
        
    def mean_validation_loss(self):
        """docstring for mean_validation_loss"""
        return numpy.mean(
            [
                self.validation_eval_function(batch_index) 
                for batch_index in xrange(self.n_valid_batches)
            ]
        )
        
    def mean_test_loss(self):
        """docstring for mean_test_loss"""
        return numpy.mean(
            [
                self.test_eval_function(batch_index) 
                for batch_index in xrange(self.n_test_batches)
            ]
        )

    def compiled_validation_function(
        self,
        classifier,
        minibatch_index,
        inputs,
        outputs
    ):
        """docstring for initialize_validation_function"""
        return theano.function(
            inputs = [minibatch_index],
            outputs = classifier.evaluation_function(inputs, outputs),
            givens = {
                inputs: self.dataset.valid_set_input[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ],
                outputs: self.dataset.valid_set_output[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ]
            }
        )

    def compiled_test_function(
        self,
        classifier,
        minibatch_index,
        inputs,
        outputs
    ):
        """docstring for initialize_test_function"""
        return theano.function(
            inputs=[minibatch_index],
            outputs=classifier.evaluation_function(inputs, outputs),
            givens={
                inputs: self.dataset.test_set_input[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ],
                outputs: self.dataset.test_set_output[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ]
            }
        )

    def compiled_training_function(
        self,
        classifier,
        minibatch_index,
        inputs,
        outputs,
        learning_rate
    ):
        """docstring for initialize_training_function"""

        return theano.function(
            inputs=[minibatch_index],
            outputs=classifier.cost_function(inputs, outputs),
            updates=classifier.updates(inputs, outputs, learning_rate),
            givens={
                inputs: self.dataset.train_set_input[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ],
                outputs: self.dataset.train_set_output[
                    minibatch_index * self.batch_size:
                    (minibatch_index + 1) * self.batch_size
                ]
            }
        )
