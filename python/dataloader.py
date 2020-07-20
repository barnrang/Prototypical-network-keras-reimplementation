import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
#from scipy.misc import imresize
from skimage.transform import resize as imresize
from skimage.transform import rotate
#from scipy.ndimage.interpolation import rotate
#import pandas as pd
BASE_PATH = "images_background"
#TRAIN_CLASS = 25
TRAIN_CLASS = 1200


def loader(path=None):
    "TODO!!!!"
    index = 0
    train_images = []
    eval_images = []
    current_save = train_images
    if path is None:
        path = BASE_PATH
    folders_list = os.listdir(path)
    folders_list.sort()
    count = 0
    loading_eval = False
    for folder in tqdm(folders_list):
        path1 = os.path.join(path, folder)
        try: #In case of invalid folder
            for char_type in os.listdir(path1):
                if not loading_eval and count >= 1200:
                    loading_eval = True
                    current_save = eval_images
                    print("Start to collect eval")

                path2 = os.path.join(path1, char_type)
                try:
                    for rot in [0,90,180,270]:
                        class_image = []
                        for image_name in os.listdir(path2):
                            image = plt.imread(os.path.join(path2, image_name))
                            image = imresize(image,(28,28), anti_aliasing=False)
                            image = rotate(image, rot)
                            image = np.expand_dims(image, axis=-1)
                            class_image.append(image)
                            current_save.append(class_image)
                    count += 1
                except NotADirectoryError:
                    print(f"Cannot load from {path2}")
        except NotADirectoryError:
            print(f"cannot load from {path1}")
            continue

    np.save(f"train.npy", (np.array(train_images) * 255).astype(np.uint8))
    np.save(f"test.npy", (np.array(eval_images) * 255).astype(np.uint8))


if __name__ == "__main__":
    images = loader()
