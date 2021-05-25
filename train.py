import tensorflow as tf
import numpy as np
import random
import os
from data_preprocessing.prepare_data import DataLoader
from models import googlenet, small_conv
from utils import DATA_REAL_PATH


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    WORKING_DIR = DATA_REAL_PATH
    model_path = os.path.join(WORKING_DIR, 'model.h5')
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes)
    input_shape = (64, 64, 1)
    output_shape = (16, 16)
    max_objects = 20

    # build the model
    model, prediction_model, debug_model = small_conv(input_shape=input_shape,
                                                      output_shape=output_shape,
                                                      num_classes=num_classes,
                                                      max_objects=max_objects)

    # load weights
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # load the data
    data_loader = DataLoader(num_classes, input_shape, output_shape, max_objects)

    dir_ = os.path.join(WORKING_DIR, 'numbers/*.png')

    batch_images, batch_hms, batch_whs, batch_regs, \
    batch_reg_masks, batch_indices = data_loader.load_from_dir(dir_)

    # Useful callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True)

    epochs = 50
    lr = 1e-3

    # create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                  run_eagerly=True)

    model.summary()

    print(batch_images.shape)

    model.fit(
        x=[batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices],
        y=np.zeros(batch_images.shape[0]),
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    model.save_weights(model_path)


if __name__ == '__main__':
    main()
