
import argparse
import os
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_way', dest='train_way', type=int, default=30)
    parser.add_argument('--train_query', dest='train_query', type=int, default=15)
    parser.add_argument('--val_way', dest='val_way', type=int, default=5)
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
from mini_protoloader import DataGenerator
from mini_proto_model import conv_net, hinge_loss, l2_distance, acc, l1_distance
#from transform import transform_gate
from util.tensor_op import *
from util.loss import *
input_shape = (None,84,84,3)
batch_size = 20
train_way = args.train_way
train_query = args.train_query
val_way = args.val_way
shot = args.shot
lr = 0.002

def scheduler(epoch):
    global lr
    if epoch % 15 == 0:
        lr /= 2
    return lr

class SaveConv(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            save_model(conv, f"model/miniimage_conv_{epoch}_{shot}_{val_way}")


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
    combine.compile(loss='categorical_crossentropy', optimizer=optimizer,
        metrics=['categorical_accuracy'])

    train_loader = DataGenerator(way=train_way, query=train_query, shot=shot, num_batch=1000)
    val_loader = DataGenerator(data_type='val',way=val_way, shot=shot)
    test_loader = DataGenerator(data_type='test',way=val_way, shot=shot, num_batch=1000)

    save_conv = SaveConv()
    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    lr_sched = cb.LearningRateScheduler(scheduler)
    tensorboard = cb.TensorBoard()


    combine.fit_generator(train_loader,epochs=50,validation_data=val_loader,
        use_multiprocessing=True, workers=4, shuffle=False,
        callbacks=[save_conv, lr_sched, tensorboard])
    combine.evaluate(test_loader)

    save_model(conv, "model/miniimage_conv_{epoch}_{shot}_{val_way}")
    combine.evaluate(test_loader)


# images, labels = zip(*list(loader('python/images_background')))
# images = np.expand_dims(images, axis=-1)
# images = np.repeat(images, repeats=3, axis=-1)
# print(images.shape)
# main_labels, sub_labels= [x[0] for x in labels], [x[1] for x in labels]
# encoder = LabelBinarizer()
# enc_main_labels = encoder.fit_transform(main_labels)
# output_num = len(np.unique(main_labels))
# bottleneck_model = conv_model()
# bottleneck_model.trainable = False
# inp = Input(shape=(105,105,3))
# features = bottleneck_model(inp)
# prediction = class_model(features)
# full_model = Model(inputs=inp, outputs=prediction)
# adam = Adam(1e-3)
# full_model.compile(optimizer=adam,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# full_model.fit(x=images, y=enc_main_labels, batch_size=32, epochs=100, validation_split=0.2)

# def class_model(inp):
#     x = Flatten()(inp)
#     x = BatchNormalization()(x)
#     x = Dense(256, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dense(output_num, activation='softmax')(x)
#     return x
