# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
#from tensorflow.keras.models import model_from_json
import os


# image dims after resizing in preprocessing 2
im_width = 200
im_height = 50

# target dim
t_len = 3
t_letters = 24

letters = {'ALPHA': 0, 'BETA': 1, 'CHI': 2, 'DELTA': 3, 'EPSILON': 4, 'ETA': 5, 'GAMMA': 6,
           'IOTA': 7, 'KAPPA': 8, 'LAMDA': 9, 'MU': 10, 'NU': 11, 'OMEGA': 12, 'OMICRON': 13,
           'PHI': 14, 'PI': 15, 'PSI': 16, 'RHO': 17, 'SIGMA': 18, 'TAU': 19, 'THETA': 20,
           'UPSILON': 21, 'XI': 22, 'ZETA': 23}


def bgwhitener(img):
    height, width = img.shape[:2]
    bgcolor = img[0][0].copy()
    for i in range(height):
        for j in range(width):
            if(img[i][j][0] == bgcolor[0] and img[i][j][1] == bgcolor[1] and img[i][j][2] == bgcolor[2]):
                img[i][j] = [255, 255, 255].copy()


def preprocess1(images_filename):
    # put in the path to training images in here plz
    img = cv2.imread(str(images_filename))
    height, width = img.shape[:2]
    # print(height,width)

    bgwhitener(img)

    # dilation
    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=1)

    binary = dilate_img.copy()

    # background black, others white, binarization
    for i in range(height):
        for j in range(width):
            if(dilate_img[i][j][0] != 255 or dilate_img[i][j][1] != 255 or dilate_img[i][j][2] != 255):
                binary[i][j] = [255, 255, 255].copy()
            else:
                binary[i][j] = [0, 0, 0].copy()

    # contour detection
    # edged = cv2.Canny(binary, 50, 100)
    # contours,hier=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # contoursReq=[]
    # for i in contours:
    #     if(i.shape[0]>50):
    #         contoursReq.append(i)

    # # print(len(contoursReq))
    # # for i in range(3):
    # #     print(contoursReq[i].shape)

    # # drawing req contours
    # ima=cv2.drawContours(binary.copy(),contoursReq,-1,(0,255,0),2)

    # # masking
    # mask=np.zeros_like(img)
    # mask=cv2.drawContours(mask,contoursReq,-1,(255,255,255),2)

    # for i in range(height):
    #     for j in range(width):
    #         if(mask[i][j][0]==255 and mask[i][j][1]==255 and mask[i][j][2]==255):
    #             mask[i][j]=[0,0,0].copy()
    #         else:
    #             mask[i][j]=[255,255,255].copy()
    return binary[:, :, 0]


def preprocess2(images_filenames):
    # images_filenames is array of image file paths
    nsamples = len(images_filenames)
    # 2000*500*150*1 size array
    X = np.zeros((nsamples, im_height, im_width, 1))

    # put in the labels file location here plz

    for i in range(nsamples):
        # read image as grayscale to prevent noise
        img = preprocess1(str(images_filenames[i]))
        img = cv2.resize(img, (200, 50), cv2.INTER_LINEAR)
        img = img/255.0
        img = np.reshape(img, (im_height, im_width, 1))
        X[i] = img

    return X

def vector_to_labels(y):
    strings={0:'ALPHA',1:'BETA',2:'CHI',3:'DELTA',4:'EPSILON',5:'ETA',6:'GAMMA',
         7:'IOTA',8:'KAPPA',9:'LAMDA',10:'MU',11:'NU',12:'OMEGA',13:'OMICRON',
         14:'PHI',15:'PI',16:'PSI',17:'RHO',18:'SIGMA',19:'TAU',20:'THETA',
         21:'UPSILON',22:'XI',23:'ZETA'}
    ls=[]
    for i in range(len(y[0])):
        g=[]
        for j in range(t_len):
            maxval=0
            for k in range(24):
                if(y[j][i][k]>y[j][i][maxval]):
                    maxval=k
            g.append(strings[maxval])
        ans=''
        for s in g[:-1]:
            ans=ans+s+','
        ans+=g[-1]
        ls.append(ans)
    return ls


def decaptcha(filenames):
    nsamples = len(filenames)  # no.of samples
    X = preprocess2(filenames)
    labels = []
    # load model

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    yhat=loaded_model.predict(X)
    pred=vector_to_labels(yhat)
    labels=pred.copy()

    return labels
