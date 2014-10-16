import theano
import numpy


class TrainingState(object):
    """docstring for TrainingState"""
    def __init__(self, patience, patience_increase, improvement_threshold, n_train_batches, n_epochs):
        super(TrainingState, self).__init__()
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.best_validation_loss = numpy.inf
        self.iter = None
        self.best_iter = 0
        self.test_score = 0.0
        self.validation_frequency = min(n_train_batches, self.patience / 2)
        self.epoch = 0
        self.n_epochs = n_epochs
        self.epoch_losses = []
        self.done_looping = False

    def continue_training(self):
        """docstring for continue_training"""
        return (self.epoch < self.n_epochs) and (not self.done_looping)

    def do_validate(self):
        """docstring for do_validate"""
        return (self.iter + 1) % self.validation_frequency == 0

    def consider(self, mean_validation_loss, mean_test_loss):
        """docstring for validate"""
        if (self.iter + 1) % self.validation_frequency == 0:
            this_validation_loss = mean_validation_loss()
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
                self.best_iter = self.iter
                self.test_score = mean_test_loss()

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
        return TrainingState(
            patience,
            patience_increase,
            improvement_threshold,
            self.n_train_batches,
            self.n_epochs
        )

    def continue_training(self, state):
        """
        """
        if state.continue_training():
            state.epoch += 1
            for minibatch_index in xrange(self.n_train_batches):
                self.training_function(minibatch_index)
                state.iter = (
                    (state.epoch - 1) * self.n_train_batches + minibatch_index
                )
                state.consider(self.mean_validation_loss, self.mean_test_loss)

                if state.patience <= state.iter:
                    state.done_looping = True
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
        state = TrainingState(
            patience,
            patience_increase,
            improvement_threshold,
            self.n_train_batches,
            self.n_epochs
        )
        while state.continue_training():
            state.epoch += 1
            for minibatch_index in xrange(self.n_train_batches):
                self.training_function(minibatch_index)
                state.iter = (
                    (state.epoch - 1) * self.n_train_batches + minibatch_index
                )
                state.consider(self.mean_validation_loss, self.mean_test_loss)

                if state.patience <= state.iter:
                    state.done_looping = True
                    break
        return [
            state.epoch_losses,
            state.best_validation_loss,
            state.best_iter,
            state.test_score
        ]
        
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
