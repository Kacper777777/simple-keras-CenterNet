import tensorflow as tf
import numpy as np
import random
import os
import glob
from utils import DATA_REAL_PATH
from data_preprocessing.generator import CustomGenerator
from models import miniconvnet_64, googlenet


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    # for readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    model_path = os.path.join(DATA_REAL_PATH, 'NEWEST_MODEL')
    checkpoint_path = os.path.join(model_path, 'checkpoint_dir', 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    input_size = 512
    channels = 3
    grayscale = False if channels == 3 else True
    downsample_factor = 4
    classes_list = ['vehicle', 'human']
    num_classes = len(classes_list)
    max_objects = 100

    model = googlenet(image_shape=(input_size, input_size, channels),
                      num_classes=num_classes,
                      max_objects=max_objects)

    model.load_weights(os.path.join(model_path, 'model_tf_format', 'model'))

    # load image names
    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.jpg'))
    image_names = pngs + jpgs
    random.shuffle(image_names)

    # training configuration

    # create custom generators
    train_gen = CustomGenerator(shuffle=True,
                                preprocessing_strategy='resize_with_pad',
                                input_size=input_size,
                                grayscale=grayscale,
                                downsample_factor=downsample_factor,
                                num_classes=num_classes,
                                max_objects=max_objects,
                                image_names=image_names[:0.8 * len(image_names)],
                                batch_size=32)

    val_gen = CustomGenerator(shuffle=True,
                              preprocessing_strategy='resize_with_pad',
                              input_size=input_size,
                              grayscale=grayscale,
                              downsample_factor=downsample_factor,
                              num_classes=num_classes,
                              max_objects=max_objects,
                              image_names=image_names[0.8 * len(image_names):],
                              batch_size=32)

    epochs = 50

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', verbose=1, save_weights_only=True,
        save_freq='epoch', mode='auto', save_best_only=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

    callbacks_list = [early_stopping, checkpoint_callback, reduce_lr_on_plateau]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss={'centernet_loss': lambda y_true, y_pred: y_pred},
                  run_eagerly=True)
    model.summary()

    print(f'The shape of the training data is: {len(image_names)}')

    model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
    )

    # Saving weights after training (tf format)
    model.save_weights(os.path.join(model_path, 'model_tf_format', 'model'), save_format='tf')

    # Saving weights after training (h5 format)
    model.save_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'), save_format='h5')


if __name__ == '__main__':
    main()
