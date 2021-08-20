import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
%matplotlib inline

train_path= 'Dataset/train/'
valid_path='Dataset/test/'

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224),batch_size=10)
validation_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path,target_size=(224,224),batch_size=10)

##importing model from Keras
mobile=tf.keras.applications.mobilenet.MobileNet()

mobile.summary()

##Model parameters
x=mobile.layers[-6].output
predictions = Dense(5,activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)

model.summary()

##Freezing the last layer of pre trainned model
for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(Adam(lr=.0001),loss ='categorical_crossentropy', metrics =['accuracy'])

##Model Training
model.fit_generator(train_batches,steps_per_epoch=4,
                    validation_data=validation_batches, validation_steps=2,epochs=60,verbose=2)

model.save('mobileNet.h5')