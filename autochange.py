import os
dir_name = './finalz_dataset'
for path in dir_name:
    # print(path)
    for root, directories, filenames in os.walk(dir_name):
        for filename in filenames:
            # print(filename)
            if filename.endswith('.png.png'):
                fullpath = os.path.join(root, filename)
                os.rename(fullpath, root + directories + filename.replace(".png.png", ".png"))