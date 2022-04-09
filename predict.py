import cv2
import numpy as np
import tensorflow as tf


class Prediction:

    def __init__(self, model, model_path, image_path, save_res_path):

        self.model = model
        self.model_path = model_path
        self.image_path = image_path
        self.save_res_path = save_res_path

    def prepare_image(self,path,size):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (size, size))
        x = x / 256.0
        return x

    def prep_mask(self, mask):
        mask = np.squeeze(mask)
        mask = [mask, mask, mask]
        mask = np.transpose(mask, (1, 2, 0))
        return mask

    def dice_coef(self, y_true, y_pred, smooth = 1.):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        sum = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
        return (2. * intersection + smooth) / (sum + smooth)

    def predict(self):
        loaded_model = self.model.load_model(self.model_path, {'dice_coef': self.dice_coef})
        size = loaded_model.layers[0].input_shape[0][1]
        x = self.prepare_image(self.image_path, size)
        y_pred = loaded_model.predict(np.expand_dims(x, axis=0))[0] > 0.5

        mask = self.prep_mask(y_pred) * 256.0
        cv2.imwrite(self.save_res_path, mask)
