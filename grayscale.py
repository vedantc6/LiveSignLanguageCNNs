import os
from PIL import Image
from resizeimage import resizeimage

database_dir = './asl_dataset'
new_dir = './final_dataset/'


def getImage(name, file, filename):
    im = Image.open(file).convert('LA')
    cover = resizeimage.resize_width(im, 28)
    # print(cover.size)
    new_dir_name = new_dir + name
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
    cover.save(new_dir_name + '/' + filename[:-5] + '.png')


for old_dir_name in os.walk(database_dir):
    folder = old_dir_name[0]
    # print(folder)
    for filename in os.listdir(folder):
        if len(filename) > 1:
            print(filename)
            getImage(folder[-1], folder + '/' + filename, filename)
