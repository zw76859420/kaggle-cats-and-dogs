#-*- coding:utf-8 -*-
#author:zhangwei

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import boston_housing
from keras.models import Sequential , Model
from keras.layers import  Input , Conv2D , MaxPool2D , Dense
from keras.layers import Reshape , Flatten , Activation , Dropout , BatchNormalization
from keras.optimizers import SGD , Adam , RMSprop
from keras import backend as K

(train_images , train_labels) , (test_images , test_labels) = boston_housing.load_data()
mean = train_images.mean(axis=0)
train_images -= mean
std = train_images.std(axis=0)
train_images /= std
test_images -= mean
test_images /= std

# print(type(train_images))
# print(train_labels.shape)

# def build_model():
#     model = Sequential()
#     model.add(Dense(units=64 ,
#                     activation='relu' ,
#                     use_bias=True ,
#                     input_shape=[train_images.shape[1] ,]))
#     model.add(Dense(units=64 ,
#                     activation='relu' ,
#                     use_bias=True))
#     model.add(Dense(units=1))
#     adam = Adam(lr=0.01 ,
#                 beta_1=0.9 ,
#                 beta_2=0.99)
#     model.compile(optimizer=adam ,
#                   loss='mse' ,
#                   metrics=['mae'])
#     return model
#
# k = 4
#
# num_val_examples = len(train_images) // k
# # print(num_val_examples)
# num_epochs = 100
# all_scores = []

# for i in range(k):
#     print("processing fold " , i)
#     val_data = train_images[i * num_val_examples : (i+1) * num_val_examples]
#     val_targets = train_labels[i * num_val_examples : (i+1) * num_val_examples]
#
#     partial_train_data = np.concatenate(
#         [train_images[: i * num_val_examples] , train_images[(i+1) * num_val_examples :]] ,
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_labels[: i * num_val_examples] , train_labels[(i+1) * num_val_examples :]] ,
#         axis=0)
#
#     # print(partial_train_data.shape)
#     # print(val_data.shape)
#
#     model = build_model()
#     model.fit(x=partial_train_data , y=partial_train_targets , epochs=num_epochs , batch_size=8)
#     val_mse , val_mae = model.evaluate(x=val_data , y=val_targets)
#     all_scores.append(val_mae)
#     pass
#
# print(all_scores)
# print(np.mean(all_scores))








# for sequence in train_images:
#     print(len(sequence))

# sequence = [max(sequence) for sequence in train_images]

# def vectorize_sequences(sequences , dimension=10000):
#     results = np.zeros([len(sequences) , dimension])
#     for i , sequence in enumerate(sequences):
#         results[i , sequence] = 1.
#     return results
#
# x_train = vectorize_sequences(train_images)
# x_test = vectorize_sequences(test_images)
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
#
# # print(train_labels.dtype)
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(train_labels).astype('float32')
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
# # print(x_val.shape)
#
# model = Sequential()
# model.add(Dense(16 , activation='relu' , input_shape=[10000 ,]))
# model.add(Dropout(0.2))
# model.add(Dense(units=16 , activation='relu'))
# model.add(Dense(units=1 , activation='sigmoid'))
# # model.summary()
# adam = RMSprop(lr=0.01)
# model.compile(optimizer=adam ,
#               loss='binary_crossentropy' ,
#               metrics=['accuracy'])
#
# history = model.fit(partial_x_train ,
#           partial_y_train ,
#           batch_size=128 ,
#           epochs=20 ,
#           validation_data=(x_val , y_val))
# score = model.evaluate(x_test , y_test)
# print("TestACC" , score[1])
# predicts = model.predict(x_test)
# print(predicts)
#
# history_dict = history.history
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# epochs = range(1 , len(acc) + 1)
# plt.plot(epochs , acc , 'bo' , label='Traing Accuracy')
# plt.plot(epochs , val_acc , 'b' , label='Validation Accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# x = [x for x in range(10)]
# print(type(x))

