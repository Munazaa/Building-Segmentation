#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.models import load_model
from unet_model_building import *
from dataset import LabeledImageDataset



import os


def _batch_generator(batch_size, dset):
    while True:
      x = []
      y = []
      for i in range(dset.__len__()):
        print(dset._pairs[i][0])
        xy = dset.get_example(i)
        x.append(xy[0].transpose([1, 2, 0]))
        y.append(np.expand_dims(xy[1], axis=0).transpose([1, 2, 0]))
        if len(y) >= int(args.batchsize):
           yield np.array(x),np.array(y)
           x = []
           y = []

def train_model(args):
    

    assert (args.tcrop % 16 == 0) and (args.vcrop % 16 == 0), "tcrop and vcrop must be divisible by 16."

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# Crop-size: {}'.format(args.tcrop))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(this_dir, "../../models"))
    log_dir = os.path.join(models_dir, args.out)
    
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = unet_model(im_sz=400)
    model = load_model(models_dir + '/last.ft.weights.045-0.776.hdf5')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_checkpoint = ModelCheckpoint(models_dir + '/last.ft.weights.{epoch:03d}-{val_loss:.3f}.hdf5', monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('/content/drive/My Drive/Building_Detection/log_unet2.csv', append=True, separator=';')
    tensorboard = TensorBoard(log_dir='/content/drive/My Drive/Building_Detection/tensorboard/', write_graph=True, write_images=True)
    model.summary()


    # Load mean image
    print(args.dataset)
    mean = np.load(os.path.join(args.dataset, "mean.npy"))
    
    # Load the dataset # Trained and tested on same dataset
    train = LabeledImageDataset(os.path.join(args.dataset, "mydata.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.tcrop, test=False, distort=False)
    test = LabeledImageDataset (os.path.join(args.dataset, "mydata.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.vcrop, test=True, distort=False)

    print("Training")
    history = model.fit_generator(generator = _batch_generator(args.batchsize, train), steps_per_epoch = train.__len__()/int(args.batchsize), epochs = args.epoch, validation_data=_batch_generator(args.batchsize,test), validation_steps=test.__len__()/int(args.batchsize) ,callbacks=[csv_logger,model_checkpoint, tensorboard])
    model.save("weights_last.hdf5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='Path to directory containing train.txt, val.txt, and mean.npy')
    parser.add_argument('--images',  help='Root directory of input images')
    parser.add_argument('--labels',  help='Root directory of label images')
    
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--test-batchsize', '-B', type=int, default=4,
                        help='Number of images in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result under "models" directory')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')

    parser.add_argument('--tcrop', type=int, default=400,
                        help='Crop size for train-set images')
    parser.add_argument('--vcrop', type=int, default=400,
                        help='Crop size for validation-set images')

    args = parser.parse_args()
    train_model(args)
