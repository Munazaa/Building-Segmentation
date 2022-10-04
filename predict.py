import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
import cv2

import os
from keras.models import load_model
from unet_model_building import *
from dataset import LabeledImageDataset


def predict(x, model):
    # predictions:
    prediction = model.predict(x)

        
    return prediction



if __name__ == '__main__':
    model = load_model("/content/drive/My Drive/Building_Detection/models/last.ft.weights.012-0.767.hdf5") 
    print(model.summary())
    mean = np.load(os.path.join('/content/drive/My Drive/Building_Detection/data/dataSplit', "mean.npy"))
    test = LabeledImageDataset(os.path.join('/content/drive/My Drive/Building_Detection/data/dataSplit', "mydata.txt"), '/content/drive/My Drive/Building_Detection/data/resized_data', '/content/drive/My Drive/Building_Detection/data/newd',  mean=mean, crop_size=400, test=True, distort=False)
    for i in range(464):
      img = test.get_example(i)[0].transpose([1,2,0]) # you can change index of the image here. It reads in sequence as in mydata.txt
      fi, fn = test._pairs[i]
      print(test._pairs[i])
      myimg = cv2.resize(cv2.imread(os.path.join('/content/drive/My Drive/Building_Detection/data/resized_data' , fi )),(400,400))
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
      cv2.imwrite("in.jpg", rgb_img)
      rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)
      h, w, c = rgb_img.shape
      print(rgb_img.shape)
      alpha = 0.5
      foreground = img.copy()
      background = rgb_img.copy()
      # normalize alpha channels from 0-255 to 0-1
      alpha_background = background[:, :, 3] / 255.0
      alpha_foreground = (foreground[:, :, 3] / 255.0)
      # set adjusted colors
      for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_foreground)

      # set adjusted alpha and denormalize back to 0-255
      background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
      cv2.imwrite("output.jpg", background)
      

      # display the image
      cv2.imwrite("/content/drive/My Drive/Building_Detection/results/" + fn, background)


    
