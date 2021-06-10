import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import numpy as np
import random
import os
import glob
from data_preprocessing.prepare_data import DataLoader
from utils import DATA_REAL_PATH
from digit_detector.centernet_digit_detector import DigitDetector


def lr_schedule(epoch):
    lr = 1
    if epoch <= 30:
        lr = 1e-3
    elif epoch <= 60:
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
    input_size = 64
    channels = 1
    classes_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes_list)
    max_objects = 10

    detector = DigitDetector(model_name='small_convnet',
                             input_shape=(input_size, input_size, channels),
                             classes_list=classes_list,
                             max_objects=max_objects,
                             resize_and_pad=False,
                             grayscale=False,
                             scale_values=1)

    # detector.load_weights(model_path)

    # load the training data
    data_loader = DataLoader(input_size=input_size, downsample_factor=4,
                             num_classes=num_classes, max_objects=max_objects, grayscale=True)

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'numbers/*.png'))
    image_names = pngs
    # image_names = image_names[:100]
    random.shuffle(image_names)

    _, images, hms, whs, regs, reg_masks, indices = data_loader.load_from_dir(image_names)

    # training configuration
    epochs = 150
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True)
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    callbacks_list = [early_stopping, reduce_lr_on_plateau]
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
        batch_size=128,
        shuffle=True,
        validation_split=0.2,
        callbacks=callbacks_list,
    )

    detector.model.save_weights(model_path)


if __name__ == '__main__':
    main()
