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
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for path_to_img in onlyfiles:
        img = Image.open(path_to_img).convert('L')
        img.resize(28, 28)


        
        print(img)

        '''plt.figure()
        plt.imshow()
        plt.colorbar()
        plt.grid(False)
        plt.show()'''

if __name__ == "__main__":
    
    path='C:/Users/Tim/Desktop/Documents/02_Python_Projects/07_TF_Tutorials/01_Basic_Image_Classification/testdata'

    preprocess_images(path)
