#!/usr/bin/env python

import os
import numpy as np
import random
import cv2
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six

from chainer.dataset import dataset_mixin

from transforms import random_color_distort


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))


def _read_image_as_array(path, dtype):
    f = Image.open(path)
   # print(f.shape)
    try:
        image = np.asarray(f, dtype=dtype)
        #print(image.shape)

    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image.transpose(2, 0, 1)

class LabeledImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataset, root, label_root, dtype=np.float32,
                 label_dtype=np.int32, mean=0, crop_size=256, test=False,
                 distort=False):
        _check_pillow_availability()
        if isinstance(dataset, six.string_types):
            dataset_path = dataset
            with open(dataset_path) as f:
                pairs = []
                k = 0 
                for i, line in enumerate(f):
                    line = line.rstrip('\n')
                    image_filename = line
                    label_filename = line
                    pairs.append((image_filename, label_filename))
        self._pairs = pairs
        self._root = root
        self._label_root = label_root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._mean = mean[np.newaxis, np.newaxis, :]
        self._crop_size = crop_size
        self._test = test
        self._distort = distort

    def __len__(self):
        return len(self._pairs)
    def get_myexample(self, i):
        image_filename, label_filename = self._pairs[i]
        print(image_filename + "  " + label_filename)
        image_path = os.path.join(self._root, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)
        image = image / 255.0

        label_path = os.path.join(self._label_root, label_filename)
        label_image = cv2.imread(label_path, 0)
        h, w, _ = image.shape
        label_image = cv2.resize(label_image,(self._crop_size,self._crop_size))
        print(label_image.shape)
        labell = np.zeros(shape=[self._crop_size, self._crop_size], dtype=np.int32) # 0: background
        labell[label_image > 0] = 1 # 1: "building"
        # Padding
        if (h < self._crop_size) or (w < self._crop_size):
            H, W = max(h, self._crop_size), max(w, self._crop_size)
            
            pad_y1, pad_x1 = (H - h) // 2, (W - w) // 2
            pad_y2, pad_x2 = (H - h - pad_y1), (W - w - pad_x1)
            image = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')

            if self._test:
                # Pad with ignore_value for test set
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant', constant_values=255)
            else:
                # Pad with original label for train set  
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'symmetric')
            
            h, w = H, W
        

        image = cv2.resize(image,(400,400))
        #label = cv2.resize(labell,(400,400))
        #cv2.imwrite("datainend.jpg",image) 
        print(labell.shape)
        print(image.shape)
        return image.transpose(2, 0, 1), labell

    def get_example(self, i):
        image_filename, label_filename = self._pairs[i]
        
        print(image_filename + "  " + label_filename)
        image_path = os.path.join(self._root, image_filename)
        #image = _read_image_as_array(image_path, self._dtype)
        image = cv2.imread(image_path)
        #print(image.shape)
        
        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)

        image = (image - self._mean) / 255.0
        #cv2.imwrite("datain.jpg",image)
        label_filename_jpg = label_filename.split('.')[0] + '.jpg'
        label_path = os.path.join(self._label_root, label_filename_jpg)
        print(label_path)
        label_image = _read_image_as_array(label_path, self._label_dtype)
        h, w, _ = image.shape
        h, w, _ = image.shape
        print(i)
        lbl_image = Image.fromarray(label_image)
        label_image = lbl_image.resize((400 ,400))
        label_image = np.array(label_image)

        label = np.zeros(shape=[self._crop_size,self._crop_size], dtype=np.int32) # 0: background
        label[label_image > 0] = 1 # 1: "building"
        cv2.imwrite("lblin.jpg",label)
        # Padding
        if (h < self._crop_size) or (w < self._crop_size):
            H, W = max(h, self._crop_size), max(w, self._crop_size)
            
            pad_y1, pad_x1 = (H - h) // 2, (W - w) // 2
            pad_y2, pad_x2 = (H - h - pad_y1), (W - w - pad_x1)
            image = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')

            if self._test:
                # Pad with ignore_value for test set
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant', constant_values=255)
            else:
                # Pad with original label for train set  
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'symmetric')
            
            h, w = H, W
        
        # Randomly flip and crop the image/label for train-set
        if not self._test:

            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]
                label = label[:, ::-1]

            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
                label = label[::-1, :]                
            
            # Random crop
            top  = random.randint(0, h - self._crop_size)
            left = random.randint(0, w - self._crop_size)
        
        # Crop the center for test-set
        else:
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2
        
        bottom = top + self._crop_size
        right = left + self._crop_size
        
        image = image[top:bottom, left:right]
        label = label[top:bottom, left:right]
        cv2.imwrite("datainend.jpg",image)
        return image.transpose(2, 0, 1), label

    def get_example_pred(self, image_filename):
        
        image_path = os.path.join(self._root, image_filename)
        image = _read_image_as_array(image_filename, self._dtype)
        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)

        image = (image - self._mean) / 255.0
        
        
        h, w, _ = image.shape
        
        
        # Padding
        if (h < self._crop_size) or (w < self._crop_size):
            H, W = max(h, self._crop_size), max(w, self._crop_size)
            
            pad_y1, pad_x1 = (H - h) // 2, (W - w) // 2
            pad_y2, pad_x2 = (H - h - pad_y1), (W - w - pad_x1)
            image = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')

            
            h, w = H, W
        
        # Randomly flip and crop the image/label for train-set
        if not self._test:

            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]

            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
            
            # Random crop
            top  = random.randint(0, h - self._crop_size)
            left = random.randint(0, w - self._crop_size)
        
        # Crop the center for test-set
        else:
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2
        
        bottom = top + self._crop_size
        right = left + self._crop_size
        
        image = image[top:bottom, left:right]
            
        return image
