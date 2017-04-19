import os
from random import shuffle

import cv2

import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm

TRAIN_DIR = '../data/train'
TEST_DIR = '../data/test/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'models/dogsvcats-{}-{}.model'.format(LR, '2conv-basic')

class practice_nn:

    def __init__(self):
        self.a = 0
        self.MODEL_NAME = 'models/dogsvcats-{}-{}.model'.format(LR, '2conv-basic')

    def label_img(self, img):
        word_label = img.split('.')[0]
        if word_label == 'cat':
            return [1, 0]
        elif word_label == 'dog':
            return [0, 1]

    def create_train_data(self):
        training_data = []
        for img in tqdm(os.listdir(TRAIN_DIR)):
            if '.jpg' in img:
                label = self.label_img(img)
                path = os.path.join(TRAIN_DIR, img)
                # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_array = np.array(img[:, :, ::-1], dtype=np.float32)
                training_data.append([img_array, np.array(label)])
        shuffle(training_data)
        np.save('np_data/train_data.npy', training_data)
        return training_data

    def process_test_data(self):
        testing_data = []
        for img in tqdm(os.listdir(TEST_DIR)):
            path = os.path.join(TEST_DIR, img)
            img_num = img.split('.')[0]
            # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_array = np.array(img[:, :, ::-1], dtype=np.float32)
            testing_data.append([img_array, img_num])
        shuffle(testing_data)
        np.save('np_data/test_data.npy', testing_data)
        return testing_data

    def choose_model(self, IMG_SIZE):
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(self.MODEL_NAME)):
            model.load(self.MODEL_NAME)
            print('model loaded!')
            return model

        train_data = np.load('np_data/train_data.npy')
        train = train_data[:-500]
        test = train_data[-500:]

        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=self.MODEL_NAME)

        print('done')

        model.save(self.MODEL_NAME)

        return model

    def make_prediction(self, model):
        if model == None:
            print("no model")
            return

        test_data = self.process_test_data()
        with open('../submissions/submission_file.csv', 'w') as f:
            f.write('id,label\n')

        with open('../submissions/submission_file.csv', 'a') as f:
            for data in tqdm(test_data):
                img_num = data[1]
                img_data = data[0]
                orig = img_data
                data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
                model_out = model.predict([data])[0]
                f.write('{},{}\n'.format(img_num, model_out[1]))

pnn = practice_nn()
#train = pnn.create_train_data()
model = pnn.choose_model(IMG_SIZE)
pnn.make_prediction(model)
