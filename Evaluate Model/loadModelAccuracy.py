from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os
from keras import optimizers
import numpy as np
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
from keras.models import load_model
from tensorflow.python.keras.preprocessing import image


##Testing Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = 'TT'

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(120, 120),
batch_size=10,
class_mode='categorical')

##Load Models
num_of_test_samples = 350
model = load_model('Model/Emotion_Potato1.1hist.h5')

##Evaluate model over test data
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

test_score = model.evaluate_generator(test_generator, batch_size)
print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100))
print("[INFO] Loss: ",test_score[0])