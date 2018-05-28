import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,BatchNormalization
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator

nrClasses = 0
current_ind = 0

def loadMedatada(path):
    labelsList = {}
    labels = {}
    nr = 0
    for _,_,files in os.walk(path):
        filesList = files
        for i in files:
            category = i.split("5",1)
            label = category[0]
            name = category[0]+"5"+category[1]
            if label not in labelsList:
                nr = nr + 1
                labelsList[label] = nr
            labels[name]=labelsList[label]
        break
    global nrClasses
    nrClasses = nr
    return labels,filesList


def createModel():
    model = Sequential()
    model.add(Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', input_shape=[64,64,3]))
    model.add(BatchNormalization())
    # model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.6))

    model.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.6))

    model.add(Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.6))

    # model.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(Flatten())

    # model.add(Dense(512, activation='relu'))

    model.add(Dense(nrClasses+1, activation='softmax'))

    return model

def normalize_img(img):
    img = img.astype(np.float32)
    img = img / 127.5 - 1
    return img


def get_batch(labels,fileList,batchSize,current_ind):
    x_batch = []
    y_batch = []
    start = current_ind
    end = min(start+batchSize,len(fileList))
    k = 0
    files_current = fileList[start:end]
    for x in files_current:
        try:
            k = k +1
            if (k%1000==0):
                print("merge "+ str(k))
            x_img = cv2.imread(x)
            x_img = cv2.resize(x_img,(64,64))
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
            x_img = normalize_img(x_img)
            x_batch.append(x_img)
            #one hot
            aux = np.zeros(nrClasses + 1)
            aux[labels[x]] = 1
            y_batch.append(aux)
        except:
            print(x)
    return x_batch,y_batch

if __name__ == '__main__':
    path ="C:/Users/Condr/Downloads/trashy trash strah/indoorCVPR_09/Images"
    labels,fileList = loadMedatada(path)
    np.random.shuffle(fileList)
    os.chdir(path)

    x_train,y_train = get_batch(labels,fileList, int(np.floor(len(fileList) * 8 / 10)),0)
    # x_val, y_val = get_batch(labels, fileList, int(np.floor(len(fileList) * 1 / 10)),int(np.ceil(len(fileList) * 7 / 10)))
    x_test, y_test = get_batch(labels, fileList, int(np.floor(len(fileList) * 2 / 10)), int(np.ceil(len(fileList)*8/10)))

    print("dpne loading")
    model1 = createModel()
    batchSize = 128
    epochs = 100

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    datagen = ImageDataGenerator(
        zoom_range=0.2, # randomly zoom into images
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    model1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    model1.summary()

    # history = model1.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, verbose=2, validation_split=0.15)
    history = model1.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize), epochs=epochs, verbose=2,
                                   validation_data=(x_test,y_test),workers=4)
    score = model1.evaluate(x_test, y_test)

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()

    print(score)
    cv2.waitKey(0)
