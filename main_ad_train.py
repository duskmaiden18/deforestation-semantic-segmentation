import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
from data import Data
from model import U_net_model
from ad_training import Ad_training

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Data parameters
size = 512
img_channels = 3
valid_split = 0.2
test_split = 0.2
random_state = 40
imgs_path = 'D:\studying\8\diplom_nn\images\*'
masks_path = 'D:\studying\8\diplom_nn\masks\*'
imgs_path2 = 'D:\studying\8\diplom_nn\clear_images\*'
masks_path2 = 'D:\studying\8\diplom_nn\clear_masks\*'

# Train parameters
loss = tf.keras.losses.binary_crossentropy
learning_rate = 0.0003
batch_size = 4
epochs = 20
optimizer = Adam(learning_rate)
save_path = 'D:\studying\8\diplom_nn\diplom\modeladam_new.h5'
model_path = 'D:\studying\8\diplom_nn\diplom\modeladam.h5'

data = Data(size, img_channels, imgs_path, masks_path,
                    valid_split, test_split, random_state)
new_data = Data(size, img_channels, imgs_path2, masks_path2,
                    valid_split, test_split, random_state)

model = U_net_model()
res_old, res_new, time = Ad_training(model, model_path, data, new_data, loss, batch_size, epochs, optimizer, save_path).train()
print('Old model results:', res_old)
print('New model results:', res_new)
print('Time took time (in seconds):', time)