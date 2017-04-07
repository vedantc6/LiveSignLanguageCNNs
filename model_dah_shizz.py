
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Dropout,Flatten
from keras.models import Model, Sequential
from keras.optimizers import SGD,Adadelta
import pickle

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.

model = Sequential()
model.add(Convolution2D(32,3, 3, activation='relu', input_shape=(28, 28, 2)))
model.add(Convolution2D(64,3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),metrics=['accuracy'])
model.save('ASLModel.h5')
