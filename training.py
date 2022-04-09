import os
import tensorflow as tf
import time
import datetime

if len(os.listdir('images')) != len(os.listdir('masks')):
    print('Images num doesnt equals masks num')

class Training:

    def __init__(self, model, data, loss, batch_size, epochs, optimizer, save_path):
        self.model = model
        self.data = data
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.save_path = save_path

    def dice_coef(self, y_true, y_pred, smooth = 1.):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        sum = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
        return (2. * intersection + smooth) / (sum + smooth)

    def prepare_data(self):

        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.data.create_sets()

        train_dataset = self.data.train_tf_dataset(train_x, train_y, batch=self.batch_size)
        valid_dataset = self.data.valid_test_tf_dataset(valid_x, valid_y, batch=self.batch_size)
        test_dataset = self.data.valid_test_tf_dataset(test_x, test_y, batch=self.batch_size)

        return (train_dataset, valid_dataset, test_dataset)


    def train(self):
        train_dataset, valid_dataset, test_dataset = self.prepare_data()

        train_steps = len(train_dataset)
        valid_steps = len(valid_dataset)
        test_steps = len(test_dataset)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), self.dice_coef]

        self.model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=metrics)

        start = time.time()
        self.model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=self.epochs,
                  steps_per_epoch=train_steps,
                  validation_steps=valid_steps,
                  callbacks=tensorboard_callback)
        end = time.time()

        train_time = end - start
        results = self.model.evaluate(test_dataset, steps=test_steps)

        self.model.save(self.save_path)

        return (results, train_time)

