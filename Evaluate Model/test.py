


from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

model = load_model('Model/VGG16.h5')
img_rows,img_cols = 224	,224

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

def check(path):
    
    # prediction
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')/255
    pred = np.argmax(model.predict(x))
   
    print("It's a {}.".format(class_labels[pred])) 
 


check(Dataset/test/Angry/A_7.jpg)
