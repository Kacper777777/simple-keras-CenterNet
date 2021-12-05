import tensorflow as tf
import numpy as np
import random
import os
import glob
from utils import DATA_REAL_PATH
from pretraining.generator import AutoEncoderGenerator
from models.googlenet import googlenet_autoencoder
from models.squezenet import squeezenet_autoencoder


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
    input_size = 256
    channels = 3
    grayscale = False if channels == 3 else True

    model = googlenet_autoencoder(image_shape=(input_size, input_size, channels))

    # model.load_weights(os.path.join(model_path, 'model_tf_format', 'model'))
    # model.load_weights(checkpoint_path)

    # load image names
    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.jpg'))
    image_names = pngs + jpgs
    # random.shuffle(image_names)

    # training configuration

    # create custom generators
    train_gen = AutoEncoderGenerator(shuffle=True,
                                     preprocessing_strategy='resize_with_pad',
                                     input_size=input_size,
                                     grayscale=grayscale,
                                     image_names=image_names[:int(0.8 * len(image_names))],
                                     batch_size=16)

    val_gen = AutoEncoderGenerator(shuffle=True,
                                   preprocessing_strategy='resize_with_pad',
                                   input_size=input_size,
                                   grayscale=grayscale,
                                   image_names=image_names[int(0.8 * len(image_names)):],
                                   batch_size=16)

    epochs = 50

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', verbose=1, save_weights_only=True,
        save_freq='epoch', mode='auto', save_best_only=True)

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

    callbacks_list = [early_stopping, checkpoint_callback, reduce_lr_on_plateau]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  run_eagerly=True)
    model.summary()

    print(f'The shape of the training data is: {len(image_names)}')

    model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        # workers=2,
        # use_multiprocessing=True
    )

    # Saving weights after training (tf format)
    model.save_weights(os.path.join(model_path, 'model_tf_format', 'model'), save_format='tf')

    # Saving weights after training (h5 format)
    model.save_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'), save_format='h5')


if __name__ == '__main__':
    main()
