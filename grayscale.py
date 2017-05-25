import os
from PIL import Image
from resizeimage import resizeimage

# database_dir = './apna_data'
# new_dir = './small_data/'
#
#
# def getImage(name, file, filename):
#     im = Image.open(file)
#     im = im.crop((0, 0, 288, 288))
#     cover = resizeimage.resize_width(im, 50)
#
#     # print(cover.size)
#     new_dir_name = new_dir + name
#     if not os.path.exists(new_dir_name):
#         os.makedirs(new_dir_name)
#     cover.save(new_dir_name + '/' + filename)
#
#
# for old_dir_name in os.walk(database_dir):
#     folder = old_dir_name[0]
#     # print(folder)
#     for filename in os.listdir(folder):
#         if len(filename) > 1:
#             print(filename)
#             getImage(folder[-1], folder + '/' + filename, filename)
def grey():
    im = Image.open("./test_image.png")
    im = im.crop((0, 0, 288, 288))
    cover = resizeimage.resize_width(im, 50)

    # print(cover.size)
    cover.save("test_image.png")
