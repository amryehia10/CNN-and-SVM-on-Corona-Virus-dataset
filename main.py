import numpy as np
import cv2
import os.path
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense


path = r"C:\Dataset"
out = []

for i in os.listdir(path):
    subfolders = path + "\\" + i
    for j in os.listdir(subfolders):
        imagePath = subfolders + "\\" + j
        im_gray = cv2.imread(imagePath)
        thresh, images = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)  # Binarization
        save = cv2.imwrite(imagePath, images)
        out.append(images)

# Normalize data
x = np.array(out)
x = x / 255.0

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
# generate batches of data
label = test_datagen.flow_from_directory(path, x, batch_size=32, class_mode='binary', color_mode='grayscale')
label.class_indices
y = label.classes  # output [0,1]

# splitting data
xtrain, xtest, yTrain, yTest = train_test_split(x, y, test_size=0.20)
print(" ")
###############################################################################

# SVM

# reshaping data for svm
Rxtrain = xtrain.reshape(32, 3 * 100 * 100)
Rxtest = xtest.reshape(8, 3 * 100 * 100)
# kernel linear
modelLin = svm.SVC(kernel='linear')
modelLin.fit(Rxtrain, yTrain)
y_pred = modelLin.predict(Rxtest)
print("error rate for Linear Kernel = ", mean_squared_error(yTest, y_pred) * 100)
print("Train Score for Linear Kernel = ", modelLin.score(Rxtrain, yTrain) * 100)
print("Linear Accuracy:", metrics.accuracy_score(yTest, y_pred))
print(" ")

# different kernels

# kernel poly
modelPol = svm.SVC(kernel='poly')
modelPol.fit(Rxtrain, yTrain)
y_pred = modelPol.predict(Rxtest)
print("error rate for poly kernel = ", mean_squared_error(yTest, y_pred) * 100)
print("Train Score for poly kernel = ", modelLin.score(Rxtrain, yTrain) * 100)
print("SVM Accuracy for poly = ", metrics.accuracy_score(yTest, y_pred))
print(" ")

# Kernel rbf
modelPol = svm.SVC(kernel='rbf')
modelPol.fit(Rxtrain, yTrain)
y_pred = modelPol.predict(Rxtest)
print("error rate for RBF kernel = ", mean_squared_error(yTest, y_pred) * 100)
print("Train Score for RBF kernel = ", modelLin.score(Rxtrain, yTrain) * 100)
print("SVM Accuracy for RBF = ", metrics.accuracy_score(yTest, y_pred))
print(" ")

###########################################################################

# Neural Network , Sequential model with 4 layers (hidden layers)

model = Sequential()
# creat convolution Kernel
model.add(Conv2D(16, (3, 3), input_shape=(100, 100, 3)))  # 100 x 100 images
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten the multi dimensional input tensors into a single dimension
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',  # loss, saves time in memory
              optimizer='adam',
              metrics=['accuracy'])

validationSteps = len(xtest) // 32
epocheSteps = len(xtrain) // 32

history = model.fit(
    x=xtrain,
    y=yTrain,
    epochs=20,
    steps_per_epoch=epocheSteps,
    validation_data=(xtest, yTest),
    validation_steps=validationSteps)