#Importing Required packages
import keras
import os
from keras import optimizers
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
import keras
import os
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D

#Tensorflow & Tensorboard packages
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


num_classes = 5
img_rows,img_cols = 120,120
batch_size = 32
train_data_dir = 'Dataset/train'
validation_data_dir = 'Dataset/validation'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode="rgb",
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)


validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='rgb',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

#compiling model 1st : custom
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


from tensorflow.keras.optimizers import RMSprop,SGD,Adam


#compiling general
print(model.summary())
  

    ## Use one optimiser at a time and check which one is the best
    

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(filepath = 'Model/HybridModel.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=8,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=8,
                              verbose=1,
                              min_delta=0.0001)


NAME = "EmotionRecHist -{}".format(int(time.time()))
#callbacks 
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


my_callbacks = [earlystop,checkpoint,reduce_lr, tensorboard]



nb_train_samples = 1703
nb_validation_samples = 94


epochs=50

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=my_callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)
    

