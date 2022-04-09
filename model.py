import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, concatenate, Dropout, MaxPooling2D
from tensorflow.keras.models import Model

class U_net_model:

    def __init__(self):
        pass

    def create_model(self, size, img_channels, filters, activation, final_activation):
        inputs = Input((size, size, img_channels))
        conv1 = Conv2D(filters, (3,3), activation=activation, padding='same')(inputs)
        conv1 = Conv2D(filters, (3,3), activation=activation, padding='same')(conv1)
        pool1 = MaxPool2D((2,2))(conv1)

        conv2 = Conv2D(filters*2, (3,3), activation=activation, padding='same')(pool1)
        conv2 = Conv2D(filters*2, (3,3), activation=activation, padding='same')(conv2)
        pool2 = MaxPool2D((2,2))(conv2)

        conv3 = Conv2D(filters*4, (3, 3), activation=activation, padding='same')(pool2)
        conv3 = Conv2D(filters*4, (3, 3), activation=activation, padding='same')(conv3)
        pool3 = MaxPool2D((2, 2))(conv3)

        conv4 = Conv2D(filters*8, (3, 3), activation=activation, padding='same')(pool3)
        conv4 = Conv2D(filters*8, (3, 3), activation=activation, padding='same')(conv4)
        pool4 = MaxPool2D((2, 2))(conv4)

        conv5 = Conv2D(filters*16, (3, 3), activation=activation, padding='same')(pool4)
        conv5 = Conv2D(filters*16, (3, 3), activation=activation, padding='same')(conv5)

        up_conv6 = Conv2DTranspose(filters*8, (2,2), strides=(2,2), padding='same')(conv5)
        up_conv6 = concatenate([up_conv6, conv4])
        conv6 = Conv2D(filters*8, (3, 3), activation=activation, padding='same')(up_conv6)
        conv6 = Conv2D(filters*8, (3, 3), activation=activation, padding='same')(conv6)

        up_conv7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(conv6)
        up_conv7 = concatenate([up_conv7, conv3])
        conv7 = Conv2D(filters*4, (3, 3), activation=activation, padding='same')(up_conv7)
        conv7 = Conv2D(filters*4, (3, 3), activation=activation, padding='same')(conv7)

        up_conv8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(conv7)
        up_conv8 = concatenate([up_conv8, conv2])
        conv8 = Conv2D(filters*2, (3, 3), activation=activation, padding='same')(up_conv8)
        conv8 = Conv2D(filters*2, (3, 3), activation=activation, padding='same')(conv8)

        up_conv9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv8)
        up_conv9 = concatenate([up_conv9, conv1], axis=3)
        conv9 = Conv2D(filters, (3, 3), activation=activation, padding='same')(up_conv9)
        conv9 = Conv2D(filters, (3, 3), activation=activation, padding='same')(conv9)

        conv10 = Conv2D(1, (1,1), activation=final_activation)(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model

    def load_model(self, model_path, custom_objects):
        model = tf.keras.models.load_model(model_path,
                                           custom_objects=custom_objects, compile=False)

        return model


