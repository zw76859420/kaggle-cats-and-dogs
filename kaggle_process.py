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
from keras.preprocessing.image import ImageDataGenerator
import os , shutil

original_dataset = 'F:/Data/kaggle/train/'
base_dir = 'F:/Data/cats_and_dogs_small/'
# os.mkdir(base_dir)
train_dir = os.path.join(base_dir , 'train/')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir , 'validation/')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir , 'test/')
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

# fnames = ['dog.{}.jpg'.format(i) for i in range(1500 , 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset , fname)
#     dst = os.path.join(test_dogs_dir , fname)
#     shutil.copyfile(src , dst)

# print(len(os.listdir(train_cats_dir)))

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir ,
                                                    target_size=(150 , 150) ,
                                                    batch_size=20 ,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir ,
                                                              target_size=[150 , 150] ,
                                                              batch_size=20 ,
                                                              class_mode='binary')


# i = 0
# for data_batch , labels_batch in validation_generator:
#     print(type(labels_batch))
#     i += 1
#     if i >= 100:
#         break

model = Sequential()
model.add(Conv2D(32 , (3 , 3) , activation='relu' , input_shape=(150 , 150 , 3)))
model.add(MaxPool2D(2 , 2))
model.add(Conv2D(64 , (3 , 3) , activation='relu'))
model.add(MaxPool2D(2 , 2))
model.add(Conv2D(128 , (3 , 3) , activation='relu'))
model.add(MaxPool2D((2 , 2)))
model.add(Conv2D(128 , (3 ,3) , activation='relu'))
model.add(MaxPool2D(2 , 2))
model.add(Flatten())
model.add(Dense(512 , activation='relu'))
model.add(Dense(1 , activation='sigmoid'))
model.summary()

adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy' ,
              optimizer=adam ,
              metrics=['accuracy'])

history = model.fit_generator(train_generator ,
                              steps_per_epoch=100 ,
                              epochs=30 ,
                              validation_data=validation_generator ,
                              validation_steps=50)

model.save('F:/Data/cats_and_dogs_small_1.h5')

