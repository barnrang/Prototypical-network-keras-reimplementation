from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import matplotlib.pyplot as plt
eps = 1e-12

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in tensorflow.keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (84,84, 3)

#build convnet to use in each siamese 'leg'
def conv_net():
    convnet = Sequential()
    for i in range(4):
        convnet.add(Conv2D(64,(3,3),padding='same',input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation('relu'))
        convnet.add(MaxPooling2D())
    convnet.add(Flatten())
    return convnet

def l1_distance(x,y):
    return tf.reduce_sum(tf.maximum(tf.abs(x-y),eps), axis=1, keep_dims=True)

def l2_distance(x,y):
    return tf.sqrt(tf.reduce_sum(tf.maximum(tf.square(x-y),eps), axis=1, keep_dims=True))

def hinge_loss(target, pred, h=1.):
    loss = tf.reduce_mean(tf.maximum(pred + h, 0.))
    return loss

def acc(target, pred):
    result = tf.cast(tf.less(pred, target), dtype=tf.float32)
    return tf.reduce_mean(result)
