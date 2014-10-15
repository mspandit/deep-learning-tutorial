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

    def start_training(
        self,
        patience=5000,
        patience_increase=2,
        improvement_threshold=0.995
    ):
        """docstring for start_training"""
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.best_validation_loss = numpy.inf
        self.best_iter = 0
        self.test_score = 0.0
        self.validation_frequency = min(self.n_train_batches, self.patience / 2)
        self.epoch = 0
        self.epoch_losses = []
        self.done_looping = False

    def continue_training(self):
        """
        """
        if (self.epoch < self.n_epochs) and (not self.done_looping):
            self.epoch += 1
            for minibatch_index in xrange(self.n_train_batches):
                self.training_function(minibatch_index)
                self.iter = (
                    (self.epoch - 1) * self.n_train_batches + minibatch_index
                )
                if (self.iter + 1) % self.validation_frequency == 0:
                    this_validation_loss = self.mean_validation_loss()
                    self.epoch_losses.append(
                        [
                            this_validation_loss,
                            self.iter
                        ]
                    )
                    if this_validation_loss < self.best_validation_loss:
                        if (
                            this_validation_loss
                            < self.best_validation_loss
                            * self.improvement_threshold
                        ):
                            self.patience = max(
                                self.patience, 
                                self.iter * self.patience_increase
                            )
                        self.best_validation_loss = this_validation_loss
                        self.best_iter = iter
                        self.test_score = self.mean_test_loss()

                if self.patience <= self.iter:
                    self.done_looping = True
                    break
            return True
        else:
            return False
        
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
