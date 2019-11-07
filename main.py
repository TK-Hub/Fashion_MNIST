#==================================================================================================
#                       Simple image classification MNIST Fashion
#                       Author: Tim
#                       Created: 07.09.2019
#==================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    '''plt.figure()
    plt.imshow(train_images[3])
    plt.colorbar()
    plt.grid(False)
    plt.show()'''

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Set up the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=18)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
