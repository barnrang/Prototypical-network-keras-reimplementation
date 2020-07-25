# Reimplementation of Prototypical Network using Keras (TF 2.0)
Blog post: https://medium.com/@barnrang/re-implementation-of-the-prototypical-network-for-few-shot-learning-using-tensorflow-2-0-keras-b2adac8e49e0

This repository is a reimplementation of the paper "Prototypical Networks for Few-shot Learning"

### Requirement
Please install the following dependency
```
numpy
skimage
tensorflow >= 2.0.0
```


### Make dataset
Assume that you redirect to the root of the project, run the following command to build the omniglot dataset
```
cd python
unzip images_background.zip; unzip images_evaluation.zip; mv images_evaluation/* images_background/
python dataloader.py
```
Or run
```
./make_omniglot.sh
```



Note that we split (1200 * 4(rotate 4 direction)) classes for training and the rest for the test set. The dataset will be collected into a numpy file `.npy`

For miniimagenet, please download the file https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view to the folder `python/` and run the following
```
cd python
unzip images.zip
python dataloader_mini.py
```
Or run

```
./make_miniimagenet.sh
```

### Train
After the dataset was created, please redirect back to the root and train the model by the following command

```
cd .. #(back to root)

# For omniglot
python proto_train.py

# For miniimagenet
python proto_train_mini.py
```

Possible arguments are
```
python proto_train.py --train_way 60 --train_query 5 --val_way 20 --shot 1 --gpu 0[for specify the gpu]
```

### Test
```
# Omniglot
python proto_test.py --model model/omniglot_conv_45_1_20 --shot 1 --test_way 20

# Miniimagenet
python mini_proto_test.py --model model/miniimage_conv_45_1_5 --shot 1 --test_way 5
```


### Reference
[1] Jake Snell and Kevin Swersky and Richard S. Zemel (2017). Prototypical Networks for Few-shot LearningCoRR, abs/1703.05175. https://arxiv.org/abs/1703.05175
