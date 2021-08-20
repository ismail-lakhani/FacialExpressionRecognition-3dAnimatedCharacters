from keras.models import Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D

from keras.applications import xception
from keras import backend as K
from keras.utils import np_utils
# from keras.callbacks import ModelCheckPoint
import glob
import numpy as np
import scipy as scp
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Dataset/train'
valid_path = 'Dataset/test'


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Potato Head Results/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory('Potato Head Results/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# useful for getting number of classes
folders = glob('Potato Head Results/train/*')


# Tensorflow requires channels in the last for input shape
baseModel = xception.Xception(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])
print (baseModel.summary())

# New Layers which will be trained on our data set and will be stacked over the Xception Model
x=baseModel.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
output=Dense(len(folders),activation='softmax')(x)


print('Stacking New Layers')

model=Model(baseModel.input,output)


# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "XCeption-EmotionRecHist -{}".format(int(time.time()))
#callbacks 
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


# fit the model
r = model.fit_generator(
  training_set,
  validation_data=validation_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(validation_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('Final/xceptionModel.h5')
