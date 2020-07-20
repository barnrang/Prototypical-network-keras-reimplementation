import numpy as np
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import keras
import random
from python.dataloader import loader

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_type='train', dim=(84,84), n_channels=3,
                way=5, shot=1, query=5, num_batch=500):
        'Initialization'
        self.type = data_type
        # if self.type == 'train':
        #     self.is_training = np.array([True for _ in range(batch_size)])
        # else:
        #     self.is_training = np.array([False for _ in range(batch_size)])
        self.dim = dim
        #self.batch_size = batch_size
        self.n_channels = n_channels
        self.num_per_class = 600
        self.num_batch = num_batch
        #self.y_target = np.zeros(self.batch_size)
        self.build_data(self.type)
        self.on_epoch_end()
        self.way = way
        self.shot = shot
        self.query = query
        self.transformer = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=True,
            shear_range=0.1

        )
        #TODO!!!!
        #self.hard_batch = np.zeros(batch_size, *dim, n_channels)

    def build_data(self, data_type):
        if data_type == 'train':
            self.class_data = np.load('python/mini_train.npy')
        elif data_type == 'val':
            self.class_data = np.load('python/mini_eval.npy')
        else:
            self.class_data = np.load('python/mini_test.npy')

        self.n_classes = len(self.class_data)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X_sample, X_query, label = self.__data_generation()
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_sample = np.empty((self.way, self.shot, *self.dim, self.n_channels))
        X_query = np.empty((self.way, self.query, *self.dim, self.n_channels))
        chosen_class = random.sample(range(self.n_classes), self.way)
        label = np.empty(self.way * self.query)
        # print(pos, neg)
        # print(self.class_data[pos][0].shape)
        # Generate data
        for i in range(self.way):
            sample_idx = random.sample(range(self.num_per_class), self.shot + self.query)
            sample_data = self.class_data[chosen_class[i]][sample_idx]/255.
            if True:
            #if self.type != 'train':
                X_sample[i] = sample_data[:self.shot]
                X_query[i] = sample_data[self.shot:self.shot + self.query]
            else:
                for j in range(self.shot):
                    params = self.transformer.get_random_transform(self.dim + (self.n_channels,))
                    x = self.transformer.apply_transform(sample_data[j], params)
                    X_sample[i][j] = x

                for j in range(self.shot, self.shot + self.query):
                    params = self.transformer.get_random_transform(self.dim + (self.n_channels,))
                    x = self.transformer.apply_transform(sample_data[j], params)
                    X_query[i][j-self.shot] = x

            label[i * self.query: (i+1) * self.query] = i
        return X_sample, X_query, np_utils.to_categorical(label)
        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
