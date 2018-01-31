import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import *
from keras.callbacks import TensorBoard
from keras.preprocessing.image import img_to_array
import numpy as np
import math
import os
from time import time
import cv2
import glob
import random


class Network:

    def __init__(self):
        self.x_train = []
        self.y_train = []

    def build_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(1))


    def compile_model(self):

        self.model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])


    def train(self):
        #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.fit(self.x_train,self.y_train,epochs=3)#,callbacks=[tensorboard])
        model_json = self.model.to_json()
        with open("model_new_grey.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights('weights_new_grey.h5')


    def read_data(self):
        x_train = []
        y_train = []
        fold = ["Far","Middle","Left","Right"]
        for typ in fold:
            images = glob.glob('../../../test/keras/OwnCollection/non-vehicles/'+typ+'/*.png')
            print(len(images))
            for names in images:
                image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
                #image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
                #image = cv2.resize(image, (28, 28))
                image = img_to_array(image)
                x_train.append(list(image))
                y_train.append(0)
        #print(image)
        for typ in fold:
            images = glob.glob('../../../test/keras/OwnCollection/vehicles/'+typ+'/*.png')
            print(len(images))
            for names in images:
                image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
                #image = cv2.resize(image, (28, 28))
                image = img_to_array(image)
                x_train.append(list(image))
                y_train.append(1)
        for _ in range(len(x_train)):
            i = random.randint(0,len(x_train)-1)
            self.x_train.append(x_train.pop(i))
            self.y_train.append(y_train.pop(i))
        self.x_train = np.array(self.x_train)
        #print(self.x_train)



    def test(self,x_test,y_test):
        score = self.model.evaluate(x_test, y_test)

        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
