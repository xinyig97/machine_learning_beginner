import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm



TRAIN_DIR = ' ' # path of train data 
TEST_DIR  = ' '# path of test data 
IMG_SIZE = 50 
LR = 1e-3 
MODEL_NAME = 'dogvscats-{}-{}.model'.format(LR,'2conv-basic')

# function to label the input images as [1,0]-> all cats  and [0,1]-> all dogs
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

# function to create train data
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data
