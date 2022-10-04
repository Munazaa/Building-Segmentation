# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K


def unet_model(n_classes=1, im_sz=160, n_channels=3, n_filters_start=32, growth_factor=2):
    n_filters = n_filters_start
    inputs = Input(shape=(im_sz, im_sz, n_channels))
    #inputs = BatchNormalization()(inputs)
    c1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    b1 = BatchNormalization()(c1)
    n_filters *= growth_factor
    c2 = Conv2D(n_filters, (4, 4), activation='relu', strides=(2,2), padding='same')(b1)
    b2 = BatchNormalization()(c2)
    c3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(b2)
    b3 = BatchNormalization()(c3)
    n_filters *= growth_factor
    c4 = Conv2D(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(b3)
    b4 = BatchNormalization()(c4)
    c5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(b4)
    b5 = BatchNormalization()(c5)
    n_filters *= growth_factor
    c6 = Conv2D(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(b5)
    b6 = BatchNormalization()(c6)
    c7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(b6)
    b7 = BatchNormalization()(c7)
    n_filters *= growth_factor
    c8 = Conv2D(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(b7)
    b8 = BatchNormalization()(c8)
    c9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(b8)
    b9 = BatchNormalization()(c9)

    m1 = concatenate([b8, b9])

    d1  = Conv2DTranspose(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(m1)
    bd1 = BatchNormalization()(d1)
    n_filters = int(n_filters / growth_factor)

    d2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bd1)
    bd2 = BatchNormalization()(d2)

    m2 = concatenate([b7, bd2])

    d3  = Conv2DTranspose(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(m2)
    bd3 = BatchNormalization()(d3)
    n_filters = int(n_filters / growth_factor)

    d4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bd3)
    bd4 = BatchNormalization()(d4)

    m3 = concatenate([b5, bd4])

    d5  = Conv2DTranspose(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(m3)
    bd5 = BatchNormalization()(d5)

    n_filters = int(n_filters / growth_factor)

    d6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bd5)
    bd6 = BatchNormalization()(d6)

    m4 = concatenate([b3, bd6])

    d7  = Conv2DTranspose(n_filters, (4, 4), activation='relu', strides=(2, 2), padding='same')(m4)
    bd7 = BatchNormalization()(d7)
    n_filters = int(n_filters / growth_factor)

    d8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bd7)
    bd8 = BatchNormalization()(d8)

    m5 = concatenate([b1, bd8])

    d9 = Conv2D(n_classes, (3, 3), activation='relu', padding='same')(bd8)

    model = Model(inputs=inputs, outputs=d9)

    return model
