import os
from PIL import Image
from scipy import misc
import numpy as np
import pickle

from keras.models import load_model

dir_name = './final_dataset'
global x_train
global y_train
x_train = []
y_train = []
number_classes = 36
batch_size = 32
epoch_size = 100  # iterations for training
cnt = 0


def getImage(file):
    image = misc.imread(file)

    return image


classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}


# def load_data(name, file, filename):
#     # im = load_img(file)
#     # # print(im.size)
#     # im = img_to_array(im)
#     # x_train.append(im)
#     # y_train.append(name)
#     # # print(Y_train)
def load_data_set():
    global x_train
    global y_train
    i = 0
    j = 0
    model = neural_network()

    for path in dir_name:
        for root, directories, filenames in os.walk(dir_name):
            # print(root, filenames)
            for filename in filenames:
                if filename.endswith(".png"):

                    fullpath = os.path.join(root, filename)
                    img = getImage(fullpath)
                    x_train.append(img)
                    t = fullpath.rindex('/')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('/')
                    y_record = np.zeros(36)
                    y_record[classes[fullpath[n + 1:t]]] = 1.0
                    y_train.append(y_record)

    x_final_train = np.asarray(x_train).astype('float32')
    y_final_train = np.asarray(y_train).astype('float32')
    x_train = []
    y_train = []

    #train_model(model,x_final_train,y_final_train)
    pickle.dump(x_final_train,open('asl_x','wb'))
    pickle.dump(y_final_train,open('asl_y','wb'))



    print(i)




def neural_network():
    ASLModel = load_model('ASLModel.h5')
    return ASLModel
# def walk_dir():
#     for directory in os.walk(dir_name):
#         folder = directory[0]
#         # print(folder)
#         for filename in os.listdir(folder):
#             if len(filename) > 1:
#                 # print(folder[-1], folder + '/' + filename, filename)
#                 load_data(folder[-1], folder + '/' + filename, filename)


def train_model(model):
    # let's train the model using SGD + momentum (how original).
    X_train = pickle.load(open('asl_x','rb'))
    Y_train = pickle.load(open('asl_y','rb'))


    X_test = X_train[30000:]
    Y_test = Y_train[30000:]
    X_train = X_train[:30000]
    Y_train = Y_train[:30000]

    model.fit(X_train, Y_train,
              batch_size=128,
              nb_epoch=10,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


#load_data_set()
train_model(neural_network())
