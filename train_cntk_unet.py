from __future__ import print_function

import numpy as np

import cntk as C
from cntk.learners import learning_rate_schedule, UnitType

import simulation
import cntk_unet

from cntk.device import set_default_device, gpu
set_default_device(gpu(0))

def train(input_images, target_masks, use_existing=False):
    shape = input_images[0].shape
    data_size = input_images.shape[0]

    x = C.input_variable(shape)
    y = C.input_variable(shape)

    z = cntk_unet.create_model(x)
    dice_coef = cntk_unet.dice_coefficient(z, y)

    checkpoint_file = "cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 2
    num_epochs = 10
    num_mb_per_epoch = int(data_size / minibatch_size)

    for e in range(0, num_epochs):
        for i in range(0, num_mb_per_epoch):
            training_x = input_images[i * minibatch_size:(i + 1) * minibatch_size]
            training_y = target_masks[i * minibatch_size:(i + 1) * minibatch_size]

            trainer.train_minibatch({x: training_x, y: training_y})

        trainer.save_checkpoint(checkpoint_file)

    return trainer

if __name__ == '__main__':
    shape = (1, 128, 128)
    data_size = 500

    input_images, target_masks = simulation.generate_random_data(shape, data_size)

    train(input_images, target_masks, False)
