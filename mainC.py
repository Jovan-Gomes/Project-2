

"""
#Essential libraries for data visualization and Deep learning
import os
import cv2
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#%matplotlib inline

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Model, layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2

DIR = 'Images/data'
SIZE = 64
TARGET_SIZE = (SIZE, SIZE)


train_datagen = ImageDataGenerator( 
    zoom_range=0.2,
    rotation_range=10,
    horizontal_flip=False,
    rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    DIR + '/train',
    batch_size=100,
    class_mode='categorical',
    target_size=TARGET_SIZE)
 
val_datagen = ImageDataGenerator(rescale=1.0/255)
 
val_generator = val_datagen.flow_from_directory(
    DIR + '/val',
    batch_size=100,
    class_mode='categorical',
    target_size=TARGET_SIZE)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    DIR + '/test',
    batch_size=100,
    class_mode='categorical',
    target_size=TARGET_SIZE,shuffle=False)

print(train_generator.class_indices)

# code to print the images and see if there are any specific visual analysis. 
input_path = 'Images/data/'

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()

# Iterate over train, val, and test sets
for i, dataset in enumerate(['train', 'val', 'test']):
    set_path = os.path.join(input_path, dataset)
    
    # Display Normal condition image
    normal_img_path = os.path.join(set_path, 'normal', os.listdir(os.path.join(set_path, 'normal'))[0])
    ax[i].imshow(plt.imread(normal_img_path), cmap='gray')
    ax[i].set_title(f'Set: {dataset}, Condition: Normal')
    
    # Display Cardiomegaly condition image
    cardiomegaly_img_path = os.path.join(set_path, 'pcardiomegaly', os.listdir(os.path.join(set_path, 'pcardiomegaly'))[0])
    ax[i+3].imshow(plt.imread(cardiomegaly_img_path), cmap='gray')
    ax[i+3].set_title(f'Set: {dataset}, Condition: Cardiomegaly')

plt.show()

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(SIZE, SIZE, 3), name="conv1"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', name="conv3"))
model.add(Conv2D(32, kernel_size=(3, 3), padding="same",activation='relu', name="conv4"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, kernel_size=(5, 5),activation='relu', name="conv5"))
model.add(Conv2D(16, kernel_size=(5, 5), padding="same",activation='relu', name="conv6"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(8, kernel_size=(5, 5),activation='relu', name="conv7"))
model.add(Conv2D(8, kernel_size=(5, 5), padding="same",activation='relu', name="conv8"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(4, kernel_size=(7, 7),activation='relu', name="conv9"))
model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.000035),
              metrics=['accuracy'])

class_weight = {0: 1, 1: 9}

history = model.fit(
    train_generator,
    steps_per_epoch=53,
    epochs=70,
    validation_data=val_generator,
    validation_steps=6,
    class_weight=class_weight
)
model.save('cardiomegaly.h5')

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('accurracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

score = model.evaluate(
    val_generator,
    steps=6,
    verbose=0
)

print('val loss:', score[0]*100)
print('val accuracy:', score[1]*100)



# serialize model to JSON
model_json = model.to_json()
with open("Cardiomegalycalssification_4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Cardiomegalycalssification_4.h5")
print("Saved model to disk")

true_labels = test_generator.classes
predictions = model.predict(test_generator,steps=len(test_generator), verbose=1)

y_pred=predictions.argmax(axis=1)

print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix


acc = accuracy_score(true_labels, np.round(y_pred))*100
cm = confusion_matrix(true_labels, np.round(y_pred))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('\nTRAIN METRIC ----------------------')
print('Train acc: {}'.format(np.round((history.history['accuracy'][-1])*100, 2)))

from sklearn import metrics
from sklearn.metrics import roc_curve

fpr_keras, tpr_keras, thresholds_keras = roc_curve(true_labels, y_pred)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
"""

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('cardiomegaly.h5')

import cv2
import numpy as np

def preprocess_image(image_path, target_size=(64, 64)):
    # Load the image using OpenCV
    img = cv2.imread(image_path)  
    img = cv2.resize(img, target_size)                 # Resize to model input size
    img = img / 255.0                                  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)                  # Add batch dimension
    return img.astype(np.float32)

def predict_image(model, image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(img)
    
    # Assuming a binary classification (e.g., Normal vs Cardiomegaly)
    class_label = 'Normal' if prediction[0][0] < 0.5 else 'Cardiomegaly'
    confidence = prediction[0][0]
    
    return class_label, confidence

# Path to a sample chest X-ray image
image_path = 'Images/data/test/pcardiomegaly/00000032_027.png'

label, confidence = predict_image(model, image_path)
print(f'Predicted Condition: {label} (Confidence: {confidence:.2f})')

