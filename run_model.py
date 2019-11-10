#==================================================================================================
#                       Apply MNIST Fashion model
#                       Author: Tim
#                       Created: 08.09.2019
#==================================================================================================
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def preprocess_images(path):
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    imgs=[]
    for img_name in onlyfiles:
        print(path, img_name)
        path_to_img = os.path.join(path, img_name)
        img = Image.open(path_to_img).convert('L')
        img = img.resize((28, 28))
        arr = np.array(img)
        arr = 1 - (arr/255)
        arr=arr.tolist()
        #print(arr)

        plt.figure()
        plt.imshow(arr)
        plt.colorbar()
        plt.grid(False)
        plt.show()
        imgs.append(arr)
    return imgs


#==================================================================================================
#                       Main Code
#==================================================================================================

if __name__ == "__main__":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    path_images='C:/Users/Tim/Desktop/Documents/02_Python_Projects/07_TF_Tutorials/01_Basic_Image_Classification/testdata'
    path_model='C:/Users/Tim/Desktop/Documents/02_Python_Projects/07_TF_Tutorials/01_Basic_Image_Classification/Fashion_MNIST/trained_model.h5'
    images=preprocess_images(path_images)

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model(path_model)

    # Show the model architecture
    new_model.summary()
    #print(images)
    predictions = new_model.predict(images)
    for i in predictions:
        percentages, prediction = i, np.argmax(i),
        print(class_names[prediction])


