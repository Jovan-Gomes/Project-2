
"""
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = '/ChestXRay_Dataset/pneumonia/data/train'
valid_path = '/ChestXRay_Dataset/pneumonia/data/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob('/ChestXRay_Dataset/pneumonia/data/train/*')
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/ChestXRay_Dataset/pneumonia/data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('/ChestXRay_Dataset/pneumonia/data/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')

r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import tensorflow as tf
from keras.models import load_model

model.save('Pneumonia.h5')
"""
from keras.models import load_model

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import numpy as np

model=load_model('Pneumonia.h5')

img=image.load_img('/ChestXRay_Dataset/pneumonia/data/test/PNEUMONIA/person1_virus_7.jpeg',target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x, axis=0)

img_data=preprocess_input(x)

classes=model.predict(img_data)

result=int(classes[0][0])

if result==0:
    print("Person is Affected By PNEUMONIA, confidence: ", classes[0][0])
else:
    print("Result is Normal, confidence: ", classes[0][0])

