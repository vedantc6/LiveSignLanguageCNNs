import os
import random
import new_model
from scipy import misc
import numpy as np

dir_name = './small_data'
class_size = 25
file_batch_size = 10000
epochs = 20

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


def getImage(file):
    return misc.imread(file)


def load_data():
    files = []
    for root, directories, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.endswith(".png"):
                fullpath = os.path.join(root, filename)
                category = root[-1:]
                record = [fullpath,category]
                files.append(record)

    return files


files = load_data()
random.shuffle(files)
length = len(files)
flag = 0
print(length)
for i in range(epochs):
    print('EPOCH ' + str(i))
    for j in range(0,length,file_batch_size):

        X = []
        Y = []
        for record in files[j:j+file_batch_size]:
            img = getImage(record[0])
            y = np.zeros(class_size)
            y[classes[str(record[1])]] = 1.0
            X.append(img)
            Y.append(y)

        X = np.asarray(X).astype('float32')
        Y = np.asarray(Y).astype('float32')
        new_model.train_model(X,Y,flag, i)
        flag = 1
