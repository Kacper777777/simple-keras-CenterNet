import tensorflow as tf
import numpy as np
import random
import os
import glob
from utils import DATA_REAL_PATH
from training.prepare_dataset import ObjectDetectionDataset
from models.squezenet import squeezenet_centernet
import argparse


def main():
    # Check if GPU  is enabled
    print(tf.config.list_physical_devices("GPU"))
    # For reproducibility
    np.random.seed(42)
    random.seed(42)
    # For readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Dataset and training configuration
    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.jpg'))
    image_names = pngs + jpgs
    shuffle_data = True
    batch_size = 16
    preprocessing_strategy = 'resize_with_pad'
    input_size = 256
    channels = 3
    grayscale = False if channels == 3 else True
    downsample_factor = 4
    classes_list = ['chessboard']
    num_classes = len(classes_list)
    max_objects = 1
    train_ratio = 0.8
    epochs = 50

    # create train and test sets
    ds_train, ds_test = ObjectDetectionDataset(image_names=image_names,
                                               shuffle_data=shuffle_data,
                                               batch_size=batch_size,
                                               preprocessing_strategy=preprocessing_strategy,
                                               input_size=input_size,
                                               grayscale=grayscale,
                                               downsample_factor=downsample_factor,
                                               num_classes=num_classes,
                                               max_objects=max_objects,
                                               ).create_datasets(train_ratio=train_ratio)

    print("train size: ", ds_train.cardinality().numpy())
    print("test size: ", ds_test.cardinality().numpy())

    # Model paths
    model_path = os.path.join(DATA_REAL_PATH, 'newest_model')
    checkpoint_path = os.path.join(model_path, 'checkpoint_dir', 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Model creation
    model = squeezenet_centernet(image_shape=(input_size, input_size, channels),
                                 num_classes=num_classes,
                                 max_objects=max_objects).get('train_model')

    # Load weights
    # model.load_weights(os.path.join(model_path, 'model_tf_format', 'model'))
    # model.load_weights(checkpoint_path)

    # Some useful callbacks

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', verbose=1, save_weights_only=True,
        save_freq='epoch', mode='auto', save_best_only=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)

    callbacks_list = [early_stopping, checkpoint_callback, reduce_lr_on_plateau]

    # Compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                  run_eagerly=True)
    model.summary()

    model.fit(
        x=ds_train,
        epochs=epochs,
        validation_data=ds_test,
        callbacks=callbacks_list,
    )

    # Saving weights after training (tf format)
    model.save_weights(os.path.join(model_path, 'model_tf_format', 'model'), save_format='tf')

    # Saving weights after training (h5 format)
    model.save_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'), save_format='h5')


if __name__ == '__main__':
    main()
