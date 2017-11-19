## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate
from keras.optimizers import SGD
import keras.backend as K

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, num_epochs=50, batch_size=128,
          brelu=False, gaussian=False):
    """
    Standard neural network training procedure.
    """

    activation = (lambda x: K.relu(x, max_value=1)) if brelu else 'relu'

    if 'mnist' in file_name:
        model = Sequential()
        model.add(Conv2D(64, (8, 8),
                         input_shape=data.train_data.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (6, 6)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(10))
        
    else:
        ins = Input(shape=data.train_data.shape[1:])
        res = Conv2D(64, (8, 8), padding='same')(ins)
        res = Activation('relu')(res)

        res = Conv2D(128, (6, 6), padding='same')(res)
        keep = res = Activation('relu')(res)

        res = Conv2D(64, (1, 1))(res)
        res = Activation('relu')(res)

        res = Conv2D(64, (1, 1))(res)
        res = Activation('relu')(res)

        res = Conv2D(128, (1, 1))(res)
        res = Activation('relu')(res)

        res = Concatenate()([res, keep])

        res = MaxPooling2D((3,3))(res)

        res = Flatten()(res)
        res = Dense(10)(res)
        model = Model(outputs=res, inputs=ins)


    model.summary()
    
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])


    for i in range(num_epochs):
        train_data = np.array(data.train_data)
        if gaussian:
            if "mnist" in file_name:
                train_data += np.random.normal(0, 0.3, size=train_data.shape)
            else:
                train_data += np.random.normal(0, 0.05, size=train_data.shape)
        model.fit(train_data, data.train_labels,
                  batch_size=batch_size,
                  validation_data=(data.validation_data, data.validation_labels),
                  nb_epoch=1,
                  shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model

    
if not os.path.isdir('models'):
    os.makedirs('models')

train(MNIST(), "models/mnist", num_epochs=30, brelu=False)
train(MNIST(), "models/mnist_brelu", num_epochs=30, brelu=True)
train(MNIST(), "models/mnist_gaussian", num_epochs=30, gaussian=True)
train(MNIST(), "models/mnist_gaussian_brelu", num_epochs=30, gaussian=True, brelu=True)

train(CIFAR(), "models/cifar", num_epochs=100, brelu=False)
train(CIFAR(), "models/cifar_brelu", num_epochs=100, brelu=True)
train(CIFAR(), "models/cifar_gaussian", num_epochs=100, gaussian=True)
train(CIFAR(), "models/cifar_gaussian_brelu", num_epochs=100, gaussian=True, brelu=True)
