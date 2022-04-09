import tensorflow as tf
import os
from data import Data
from training import Training
import datetime
import time
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
from model import U_net_model

class Ad_training(Training):

    def __init__(self, model, model_path, data, new_data, loss, batch_size, epochs, optimizer, save_path):
        super().__init__(model, data, loss, batch_size, epochs, optimizer, save_path)
        self.new_data = new_data
        self.model_path = model_path

    def prepare_data(self):

        (test_x_old, test_y_old) = self.data.create_sets(mode = 'test')
        (train_x, train_y), (valid_x, valid_y), (test_x_new, test_y_new) = self.new_data.create_sets()

        train_dataset = self.data.train_tf_dataset(train_x, train_y, batch=self.batch_size)
        valid_dataset = self.data.valid_test_tf_dataset(valid_x, valid_y, batch=self.batch_size)
        test_dataset_old = self.data.valid_test_tf_dataset(test_x_old, test_y_old, batch=self.batch_size)
        test_dataset_new = self.data.valid_test_tf_dataset(test_x_new, test_y_new, batch=self.batch_size)
        test_dataset = test_dataset_old.concatenate(test_dataset_new)

        return (train_dataset, valid_dataset, test_dataset, test_dataset_old)

    def train(self):
        train_dataset, valid_dataset, test_dataset, test_dataset_old = self.prepare_data()

        train_steps = len(train_dataset)
        print(train_steps)
        valid_steps = len(valid_dataset)
        test_steps = len(test_dataset)
        test_steps_old = len(test_dataset_old)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), self.dice_coef]

        loaded_model = self.model.load_model(self.model_path, {'dice_coef': self.dice_coef})
        loaded_model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=metrics)
        results_old = loaded_model.evaluate(test_dataset_old, steps=test_steps_old)
        print('Old: ', results_old)

        start = time.time()
        loaded_model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=self.epochs,
                  steps_per_epoch=train_steps,
                  validation_steps=valid_steps,
                  callbacks=tensorboard_callback)
        end = time.time()

        train_time = end - start
        results_new = loaded_model.evaluate(test_dataset, steps=test_steps)
        print('New: ', results_new)

        loaded_model.save(self.save_path)

        return (results_old, results_new, train_time)


