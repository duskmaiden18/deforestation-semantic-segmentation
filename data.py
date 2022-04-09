from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow as tf


class Data:

    def __init__(self, size, img_channels, imgs_path, masks_path,
                 valid_split, test_split, random_state):
        self.size = size
        self.img_channels = img_channels
        self.imgs_path = imgs_path
        self.masks_path = masks_path
        self.valid_split = valid_split
        self.test_split = test_split
        self.random_state = random_state

    def create_sets(self, mode='all'):
        images = sorted(glob(self.imgs_path))
        masks = sorted(glob(self.masks_path))

        dataset_size = len(images)
        valid_size = int(self.valid_split * dataset_size)
        test_size = int(self.test_split * dataset_size)

        train_x, valid_x, train_y, valid_y = train_test_split(images, masks,
                            test_size=valid_size, random_state=self.random_state)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,
                            test_size=test_size, random_state=self.random_state)

        if mode == 'all':
            return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
        elif mode == 'test':
            return (test_x, test_y)

    def prepare_image(self, path):
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (self.size, self.size))
        x = x / 256.0
        return x

    def prepare_mask(self, path):
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (self.size, self.size))
        x = x / 256.0
        x = np.expand_dims(x, axis=-1)
        return x


    def tf_parse(self, x, y):

        def _parse(x, y):
            '''A transformation function to preprocess raw data
                    into trainable input. '''
            x = self.prepare_image(x)
            y = self.prepare_mask(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([self.size, self.size, self.img_channels])
        y.set_shape([self.size, self.size, 1])
        return x, y

    def flip(self, x, y):
        x = tf.image.flip_left_right(x)
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_left_right(y)
        y = tf.image.flip_up_down(y)
        return x,y

    def brightness(self, x, y):
        x = tf.image.stateless_random_brightness(
            x, max_delta=0.2, seed=(0,3))
        return x, y


    def train_tf_dataset(self, x, y, batch=8):
        '''Construct a data generator using tf.Dataset'''
        dataset1 = tf.data.Dataset.from_tensor_slices((x, y))
        dataset2 = tf.data.Dataset.from_tensor_slices((x, y))
        dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
        dataset1 = dataset1.map(self.tf_parse)
        dataset2 = dataset2.map(self.tf_parse)
        dataset3 = dataset3.map(self.tf_parse)
        dataset2 = dataset2.map(self.flip)
        dataset3 = dataset3.map(self.brightness)
        dataset = dataset1.concatenate(dataset2)
        dataset = dataset.concatenate(dataset3)
        dataset = dataset.batch(batch)
        return dataset

    def valid_test_tf_dataset(self, x, y, batch=8):
        '''Construct a data generator using tf.Dataset'''
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        return dataset


