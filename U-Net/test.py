import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from PIL import Image
import os
from keras.models import load_model
from unet_model_building import *
from dataset import read_image


def predict(x, model):
    # predictions:
    prediction = model.predict(x)

        
    return prediction



if __name__ == '__main__':
    model = load_model("/content/drive/My Drive/Building_Detection/models/last.ft.weights.012-0.767.hdf5") #weights2
    print(model.summary())
    test = read_image('/content/drive/My Drive/Building_Detection/data/resized_data/qgis_patch_27.tif')
    img = test.transpose([1,2,0]) # you can change index of the image here. It reads in sequence as in mydata.txt
    myimg = cv2.resize(cv2.imread('/content/drive/My Drive/Building_Detection/data/resized_data/qgis_patch_27.tif'),(400,400))
    print(img.shape)
    mymat = predict(np.expand_dims(img, axis=0), model)
    ret, bw_img = cv2.threshold(mymat[0],0.32,1,cv2.THRESH_BINARY) # you can change thrshold value to get visulaization, I have set this to what i find appropriate
    cv2.imwrite("temp.jpg", bw_img*255)
    pil_img = Image.open("temp.jpg")
    pil_img = pil_img.convert("RGBA")
    datas = pil_img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((0, 255, 0, 128))
        else:
            newData.append((255, 255, 255, 0))

    pil_img.putdata(newData)
    img = np.array(pil_img)
    rgb_img = cv2.cvtColor(myimg, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)
    h, w, c = rgb_img.shape
    print(rgb_img.shape)
    alpha = 0.5
    foreground = img.copy()
    background = rgb_img.copy()
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = (foreground[:, :, 3] / 255.0)
    print(alpha_foreground)
    # set adjusted colors
    for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    # display the image
    cv2.imwrite("result.jpg", background)
