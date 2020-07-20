import keras
from keras.layers import *
import tensorflow as tf
import numpy as np
from dataloader import *

class Model:
    def __init__(self, config):
        self.config = config

    def model(self):
        inp = Input(shape=[105,105,1])
        x = Conv2D(filters=32, padding='same', kernel_size=(2,2), activation='relu')
        x = MaxPooling2D()
        x = Conv2D(filters=128, padding='same', kernel_size=(2,2), activation='relu')
        x = MaxPooling2D()

