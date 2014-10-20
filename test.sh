#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./deep_learning
python ./deep_learning/logistic_trainer.py --training-epochs 1
echo "You should have seen 'epoch 1, validation error 12.458333%'"
python ./deep_learning/perceptron_trainer.py --training-epochs 1
echo "You should have seen 'epoch 1, validation error 9.620000%'"
python ./deep_learning/convolutional_trainer.py --training-epochs 1
echo "You should have seen 'epoch 1, validation error 9.230000%'"
python ./deep_learning/denoising_autoencoder_trainer.py --training-epochs 1
echo "You should have seen 'Training epoch 1, cost 63.289169'"
echo "You should have seen 'Training epoch 1, cost 81.771419'"
python ./deep_learning/stacked_denoising_autoencoder_trainer.py --pretraining-epochs 1 --training-epochs 1
echo "You should have seen 'Pretraining layer 0 epoch 0, cost 71.648650'"
echo "You should have seen 'Pretraining layer 1 epoch 0, cost 475.566207'"
echo "You should have seen 'Pretraining layer 2 epoch 0, cost 234.241923'"
echo "You should have seen 'Training epoch 1, validation error 10.950000%'"
python ./deep_learning/restricted_boltzmann_machine_trainer.py --training-epochs 1
python ./deep_learning/deep_belief_trainer.py --pretraining-epochs 1 --training-epochs 1
