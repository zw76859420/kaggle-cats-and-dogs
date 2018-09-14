#-*- coding:utf-8 -*-
#author:zhangwei

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential , Model
from keras.layers import  Input , Conv2D , MaxPool2D , Dense
from keras.layers import Reshape , Flatten , Activation
from keras.optimizers import SGD , Adam
from keras import backend as K

(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()
train_images = train_images.reshape((60000 , 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000 , 28*28))
test_images = test_images.astype('float32') / 255

train_labels = np_utils.to_categorical(train_labels , num_classes=10)
test_labels = np_utils.to_categorical(test_labels , num_classes=10)

# digit = train_images[4]
# plt.imshow(digit)
# plt.show()

# my_slice = train_images[: , 14: , 14:]
print(train_images)

# input_data = Input(shape=[784 ,] , name='Input')
# layer1 = Dense(units=512 , activation='relu')(input_data)
# pred = Dense(10 , activation='softmax')(layer1)
# model = Model(inputs=input_data , outputs=pred)
#
# adam = Adam(lr=0.01)
# model.compile(optimizer=adam ,
#               loss='categorical_crossentropy' ,
#               metrics=['accuracy'])
#
# model.fit(train_images , train_labels , 10 , 32 , validation_split=0.2)
#
# score = model.evaluate(test_images , test_labels)
# print('test accuracy :' , score[1])

