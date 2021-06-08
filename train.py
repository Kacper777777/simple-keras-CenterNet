import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import random
import os
from data_preprocessing.prepare_data import DataLoader
from utils import DATA_REAL_PATH
from centernet_detector import CenterNetDetector


def lr_schedule(epoch):
    lr = 1
    if epoch <= 20:
        lr = 1e-3
    elif epoch <= 40:
        lr = 1e-4
    else:
        lr = 1e-5
    return lr


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    # for readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    model_path = os.path.join(DATA_REAL_PATH, 'model.h5')
    input_size = 256
    channels = 3
    classes_list = ['car', 'human']
    num_classes = len(classes_list)
    max_objects = 50

    detector = CenterNetDetector(model_name='small_convnet',
                                 input_shape=(input_size, input_size, channels),
                                 classes_list=classes_list,
                                 max_objects=max_objects,
                                 resize_and_pad=False,
                                 grayscale=False,
                                 scale_values=1)

    # detector.load_weights(model_path)

    # load the training data
    data_loader = DataLoader(input_size=input_size, downsample_factor=4,
                             num_classes=num_classes, max_objects=max_objects)

    dir_ = os.path.join(DATA_REAL_PATH, 'cars/*.png')

    _, images, hms, whs, regs, reg_masks, indices = data_loader.load_from_dir(dir_, True)

    # training configuration
    epochs = 100
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [early_stopping, lr_scheduler]
    optimizer = tf.keras.optimizers.Adam()

    detector.model.compile(optimizer=optimizer,
                           loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                           run_eagerly=True)

    detector.model.summary()

    print(f'The shape of the training data is: {images.shape}')

    detector.model.fit(
        x=[images, hms, whs, regs, reg_masks, indices],
        y=np.zeros(images.shape[0]),
        epochs=epochs,
        batch_size=16,
        shuffle=True,
        validation_split=0.2,
        callbacks=callbacks_list,
    )

    detector.model.save_weights(model_path)


if __name__ == '__main__':
    main()
