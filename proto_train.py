import argparse
import os
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_way', dest='train_way', type=int, default=60)
    parser.add_argument('--train_query', dest='train_query', type=int, default=5)
    parser.add_argument('--val_way', dest='val_way', type=int, default=20)
    parser.add_argument('--shot', dest='shot', type=int, default=1)
    parser.add_argument('--gpu', dest='gpu', type=int, default=0)

    return parser.parse_args()

args = parser()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

from tensorflow.keras import callbacks as cb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers as rg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import backend as K


import numpy.random as rng

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from python.dataloader import loader
from protoloader import DataGenerator
from proto_model import conv_net, hinge_loss, l2_distance, acc, l1_distance
#from transform import transform_gate
from util.tensor_op import *
from util.loss import *
input_shape = (None,28,28,1)
batch_size = 20
train_way = args.train_way
train_query = args.train_query
val_way = args.val_way
shot = args.shot
lr = 0.002

def scheduler(epoch):
    global lr
    if epoch % 100 == 0:
        lr /= 2
    return lr

class SaveConv(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            save_model(conv, f"model/omniglot_conv_{epoch}_{shot}_{val_way}")

if __name__ == "__main__":
    conv = conv_net()
    sample = Input(input_shape)
    conv_5d = TimeDistributed(conv)
    out_feature = conv_5d(sample)
    out_feature = Lambda(reduce_tensor)(out_feature)
    inp = Input(input_shape)
    map_feature = conv_5d(inp)
    map_feature = Lambda(reshape_query)(map_feature)
    pred = Lambda(proto_dist)([out_feature, map_feature]) #negative distance
    combine = Model([sample, inp], pred)

    optimizer = Adam(0.001)
    combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    train_loader = DataGenerator(way=train_way, query=train_query, shot=shot, num_batch=1000)
    val_loader = DataGenerator(data_type='val',way=val_way, shot=shot)

    (x,y), z = train_loader[0]
    print(x.shape, y.shape, z.shape)
    print(combine.summary())

    save_conv = SaveConv()
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    lr_sched = cb.LearningRateScheduler(scheduler)
    tensorboard = cb.TensorBoard()

    combine.fit_generator(train_loader,epochs=1000,validation_data=val_loader,  use_multiprocessing=False, workers=4, shuffle=False, callbacks=[save_conv, lr_sched, tensorboard])

    save_model(conv, "model/omniglot_conv")
