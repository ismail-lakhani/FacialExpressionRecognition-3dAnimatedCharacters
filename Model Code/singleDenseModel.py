from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os
from keras import optimizers
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
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

# define convnet model.
model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


# Block-2 

model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_2'))



# Block-3

model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(9,9)))
          
          
# dropout 50% and flatten layer.
model.add(Dropout(0.5))
model.add(Flatten(name='flatten_1'))

          
# Full connection
#output
model.add(Dense(5, activation='softmax', name='output_layer'))


#compiling model 1st : custom
#o = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=o, metrics=['accuracy'])
#compiling general

model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics=['accuracy'])	

print(model.summary())

 ## Use one optimiser at a time and check which one is the best
from tensorflow.keras.optimizers import RMSprop, SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(filepath = 'Model/singleDenseModel.h5',
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
    



