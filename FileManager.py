from PIL import Image as Image
import numpy as np
import os
import time

path = 'data/train/train'
test = 'data/train/train/cat.4.jpg'


def read(file):
    return Image.open(file)


def save(file, destination, mode):
    upper_mode = mode.upper()
    try:
        file.save(destination, upper_mode)
    except KeyError:
        print("{} caused KeyError".format(mode))
        return False
    return True


def delete(filepath):
    try:
        if os.path.isfile(filepath):
            os.unlink(filepath)
    except Exception as e:
        print(e)


def close(file):
    file.close()
    return True


def dread(file):
    im = Image.open(file)
    #im.show()
    im_t = np.array(im)/255
    print("image: {}".format(im_t.shape))
    im.close()
    return True


def getMaxSizes():
    image_sizes = []
    start = time.time()
    for filename in os.listdir(path):
        if '.jpg' in filename:
            print("{}".format(filename))
            image = Image.open(path + '/' + filename)
            image_shape = np.array(image).shape
            image_sizes.append(image_shape)
            image.close()
    end = time.time()
    print("total time: {}".format(end - start))

    return max(image_sizes, key=lambda item:item[1])