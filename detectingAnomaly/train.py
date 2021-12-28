import numpy as np
import glob
import os
import cv2

from keras.preprocessing.image import img_to_array,load_img

from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping
import imutils

print('\n'+15*'-'+' Rodando '+15*'-')
store_image = []
train_path = './train'
fps = 5
train_videos = os.listdir(train_path)
train_images_path = train_path+'/frames'
os.makedirs(train_images_path)

def store_inarray(image_path):
    image = load_img(image_path)
    image= img_to_array(image)
    image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
    gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
    store_image.append(gray)

for video in train_videos:
    os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(train_path,video,fps,train_path))
    images=os.listdir(train_images_path)
    for image in images:
        image_path=train_images_path + '/' + image
        store_inarray(image_path)
