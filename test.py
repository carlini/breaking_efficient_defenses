## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from l2_attack import CarliniL2
import keras.backend as K
from keras.models import load_model
import tensorflow as tf

import numpy as np
from setup_mnist import MNIST
from setup_cifar import CIFAR

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
def get_labs(y):
    l = np.zeros((len(y),10))
    for i in range(len(y)):
        r = np.random.random_integers(0,9)
        while r == np.argmax(y[i]):
            r = np.random.random_integers(0,9)
        l[i,r] = 1
    return l

def attack(data, name):
    sess = K.get_session()
    model = load_model("models/"+name, custom_objects={'fn': fn})
    class Wrap:
        image_size = 28 if "mnist" in name else 32
        num_labels = 10
        num_channels = 1 if "mnist" in name else 3
        def predict(self, x):
            return model(x)
    attack = CarliniL2(sess, Wrap(), batch_size=100,
                       max_iterations=10000, binary_search_steps=5,
                       initial_const=1, targeted=True)
    adv = attack.attack(data.test_data[:100],
                        get_labs(data.test_labels[:100]))
    np.save("/tmp/"+name, adv)
    print(np.mean(np.sum((adv-data.test_data[:100])**2,axis=(1,2,3))**.5))
    
attack(MNIST(), "mnist")
attack(MNIST(), "mnist_brelu")
attack(MNIST(), "mnist_gaussian")
attack(MNIST(), "mnist_gaussian_brelu")

attack(CIFAR(), "cifar")
attack(CIFAR(), "cifar_brelu")
attack(CIFAR(), "cifar_gaussian")
attack(CIFAR(), "cifar_gaussian_brelu")
