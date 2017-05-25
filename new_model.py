# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import os
from scipy import misc
import numpy as np
# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Convolutional network building
global network
global model

network = input_data(shape=[None, 50, 50, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu',regularizer='L2')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu',regularizer='L2')
network = conv_2d(network, 64, 3, activation='relu',regularizer='L2')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu',regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 25, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network)
model.load('./epochs/9/my_model.tflearn')

#getting our data
def train_model(X,Y,flag, i):
    global network
    global model

    if(len(X) > 9000):
        X_test = X[9000:]
        Y_test = Y[9000:]
        X = X[:9000]
        Y = Y[:9000]
    else:
        X_test = X[5000:]
        Y_test = Y[5000:]
        X = X[:5000]
        Y = Y[:5000]

    # Train using classifier
    if(flag == 0):
        model = tflearn.DNN(network, tensorboard_verbose=3)

    model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=96, run_id='asl_cnn')

    if not os.path.exists('./epochs/' + str(i) + '/'):
        os.makedirs('./epochs/' + str(i) + '/')
    # Save a model
    model.save('./epochs/' + str(i) + '/' + 'my_model.tflearn')


def test_model():
    global network
    global model

    img = misc.imread('test_image.png')
    x = []
    x.append(img)
    x = np.asarray(x).astype('float32')
    y = model.predict(x)
    answ = np.argsort(y[0])
    print(answ)
