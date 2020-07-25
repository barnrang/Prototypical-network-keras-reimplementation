import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize as imresize
import os

train_label = pd.read_csv('train.csv')
val_label = pd.read_csv('val.csv')
test_label = pd.read_csv('test.csv')

train_images = []

PATH = 'images'

for name, df in train_label['filename'].groupby(train_label['label']):
    images = []
    for image_name in df.values:
        image = imread(os.path.join(PATH, image_name))
        image = (imresize(image, (84,84)) * 255.).astype(np.uint8)
        images.append(image)

    train_images.append(images)

val_images = []

PATH = 'images'

for name, df in val_label['filename'].groupby(val_label['label']):
    images = []
    for image_name in df.values:
        image = imread(os.path.join(PATH, image_name))
        image = (imresize(image, (84,84)) * 255.).astype(np.uint8)
        images.append(image)

    val_images.append(images)

test_images = []

PATH = 'images'

for name, df in test_label['filename'].groupby(test_label['label']):
    images = []
    for image_name in df.values:
        image = imread(os.path.join(PATH, image_name))
        image = (imresize(image, (84,84)) * 255.).astype(np.uint8)
        images.append(image)

    test_images.append(images)

train_images = np.array(train_images)

val_images = np.array(val_images)
test_images = np.array(test_images)

np.save('mini_train', train_images)
np.save('mini_val', val_images)
np.save('mini_test', test_images)

