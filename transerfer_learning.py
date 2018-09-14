#-*- coding:utf-8 -*-
#author:zhangwei

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import np_utils , plot_model
from keras.datasets import imdb
from keras.models import Sequential , Model
from keras.layers import  Input , Conv2D , MaxPool2D , Dense
from keras.layers import Reshape , Flatten , Activation , Dropout , BatchNormalization
from keras.optimizers import SGD , Adam , RMSprop
from keras import backend as K
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import os , shutil

conv_base = VGG16(weights='imagenet' ,
                  include_top=False ,
                  input_shape=[150 , 150 , 3])
conv_base.trainable = True
set_trainable =False
for layer in conv_base.layers:
    # print(layer.name)
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# conv_base.summary()
# print(len(conv_base.trainable_weights))

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256 , activation='relu'))
model.add(Dense(1 , activation='sigmoid'))
model.summary()
print(len(model.trainable_weights))


train_datagen = ImageDataGenerator(rescale=1. / 255 ,
                                   rotation_range=40 ,
                                   width_shift_range=0.2 ,
                                   height_shift_range=0.2 ,
                                   shear_range=0.2 ,
                                   zoom_range=0.2 ,
                                   horizontal_flip=True ,
                                   fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1. / 255)

original_dataset = 'F:/Data/kaggle/train/'
base_dir = 'F:/Data/cats_and_dogs_small/'
# os.mkdir(base_dir)
train_dir = os.path.join(base_dir , 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir , 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir , 'test')
# os.mkdir(test_dir)
#
train_cats_dir = os.path.join(train_dir , 'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir , 'dogs')
# os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir , 'cats')
# os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir , 'dogs')
# os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir , 'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir , 'dogs')
# os.mkdir(test_dogs_dir)

train_generator = train_datagen.flow_from_directory(train_dir ,
                                                    target_size=[150 , 150] ,
                                                    batch_size=20 ,
                                                    class_mode='binary')
validation_generator = valid_datagen.flow_from_directory(validation_dir ,
                                                         target_size=[150 , 150] ,
                                                         batch_size=20 ,
                                                         class_mode='binary')
adam = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy' ,
              optimizer=adam ,
              metrics=['accuracy'])
model.fit_generator(train_generator ,
                    steps_per_epoch=100 ,
                    epochs=1 ,
                    validation_data=validation_generator ,
                    validation_steps=50)

test_generator = valid_datagen.flow_from_directory(test_dir ,
                                                   target_size=[150 , 150] ,
                                                   batch_size=20 ,
                                                   class_mode='binary')
test_loss , test_acc = model.evaluate_generator(test_generator , steps=50)
print('Test_acc' , test_acc)

