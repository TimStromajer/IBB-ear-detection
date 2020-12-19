from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

PATH_TO_TRAIN = "./images/train/"
PATH_TO_VALIDATION = "./images/validation/"
PATH_TO_TEST_FEMALE = "./images/test/female"
PATH_TO_TEST_MALE = "./images/test/male"

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(PATH_TO_TRAIN,
                                          target_size=(100, 100),  #transform all images
                                          batch_size=3,
                                          class_mode="binary"
                                          )

validation_dataset = train.flow_from_directory(PATH_TO_VALIDATION,
                                          target_size=(100, 100),  #transform all images
                                          batch_size=3,
                                          class_mode="binary"
                                          )

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16, (3,3), activation="relu",input_shape=(100, 100, 3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
                                     tf.keras.layers.MaxPool2D(2, 2),
                                     #
                                     tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
                                     tf.keras.layers.MaxPool2D(2, 2),
                                     ##
                                     tf.keras.layers.Flatten(),
                                     ##
                                     tf.keras.layers.Dense(512, activation="relu"),
                                     ##
                                     tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(lr=0.001),
              metrics=["accuracy"])
model_fit = model.fit(train_dataset,
                      steps_per_epoch=3,
                      epochs=600,                            # increase for longer training
                      validation_data=validation_dataset)

allF = 0
foundF = 0
for i in os.listdir(PATH_TO_TEST_FEMALE):
    allF += 1
    img = image.load_img(PATH_TO_TEST_FEMALE+"//"+i, target_size=(100, 100))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 1:
        print("F male ", i)
    else:
        foundF += 1
        print("T female ", i)

print("--------------------")

allM = 0
foundM = 0
for i in os.listdir(PATH_TO_TEST_MALE):
    allM += 1
    img = image.load_img(PATH_TO_TEST_MALE+"//"+i, target_size=(100,100))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 1:
        foundM += 1
        print("T male ", i)
    else:
        print("F female ", i)
        continue

print(foundF, " / ", allF)
print(foundM, " / ", allM)
