import os
import keras
import pandas as pd
import numpy as np
import sklearn
import gc
import PIL
from PIL import Image
from keras import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Input
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler

TRAIN_DATA_PATH = "cifar-10-batches-py"
TRAIN_DATA_FILENAME = "data_batch_"
TRAIN_DATA_COUNT = 5
LR = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
BS = [8, 16, 32, 64, 128, 256]
EPOCH = 500
LENET_INPUT_SHAPE = (32, 32, 3)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


tmp_train_x = []
tmp_train_y = []

print("Importing train pictures...")
for id in range(1, TRAIN_DATA_COUNT + 1):
    print("Importing %d of %d" % (id, TRAIN_DATA_COUNT))
    dic = unpickle(os.path.join(
        os.getcwd(), TRAIN_DATA_PATH, TRAIN_DATA_FILENAME + str(id)))
    for item in dic[b'data']:
        img = np.asarray(item, dtype="float32")
        img.resize((32, 32, 3))
        img = img / 255
        tmp_train_x.append(img)
    for item in dic[b'labels']:
        y = []
        for i in range(0, 10):
            y.append(0)
        y[item] = 1
        tmp_train_y.append(y)

train_x = np.asarray(tmp_train_x)
del (tmp_train_x)
train_y = np.asarray(tmp_train_y)
del (tmp_train_y)
gc.collect


for lr in LR:
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C1', input_shape=LENET_INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='P1'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='P2'))
    model.add(Conv2D(filters=120, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C3'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='prediction'))

    model.summary()

    if os.path.exists('LeNet_origin_lr=%f.h5' % lr):
        model = load_model('LeNet_origin_lr=%f.h5' % lr)
    sgd = SGD(lr=lr, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=128,
              epochs=EPOCH, verbose=1, validation_split=0.05, callbacks=[TensorBoard(log_dir='./log/LeNet_origin_lr=%f.h5' % lr)])
    model.save('LeNet_origin_lr=%f.h5' % lr)
    del (model)
    gc.collect


for bs in BS:
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C1', input_shape=LENET_INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='P1'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='P2'))
    model.add(Conv2D(filters=120, kernel_size=(5, 5), activation='relu',
                     padding='valid', name='C3'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='prediction'))

    model.summary()

    if os.path.exists('LeNet_origin_bs=%f.h5' % bs):
        model = load_model('LeNet_origin_bs=%f.h5' % bs)
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=bs,
              epochs=EPOCH, verbose=1, validation_split=0.05, callbacks=[TensorBoard(log_dir='./log/LeNet_origin_bs=%f.h5' % bs)])
    model.save('LeNet_origin_bs=%f.h5' % bs)
    del (model)
    gc.collect
