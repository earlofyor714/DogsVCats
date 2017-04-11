import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import time

TRAIN_DIR = '../data/train'
TEST_DIR = '../data/test/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvcats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if '.jpg' in img:
            label = label_img(img)
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []


start = time.time()
create_train_data()
end = time.time()
print("total time: {}".format(end - start))