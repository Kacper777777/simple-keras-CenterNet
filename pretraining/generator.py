import tensorflow as tf
import numpy as np
import random
import os
import glob
from tensorflow_core.python.keras.utils.data_utils import Sequence


class AutoEncoderGenerator(Sequence):
    def __init__(self, shuffle, preprocessing_strategy, input_size, grayscale,
                 image_names, batch_size):
        self.__shuffle = shuffle
        self.__preprocessing_strategy = preprocessing_strategy
        self.__input_size = input_size
        self.__grayscale = grayscale
        self.__image_preprocessor = ImagePreprocessor(preprocessing_strategy=self.__preprocessing_strategy,
                                                      target_shape=self.__input_size,
                                                      grayscale=self.__grayscale)
        self.__image_names = image_names
        self.__batch_size = batch_size
        self.__batch_number = 0

    def __len__(self):
        """Denotes the number of batches per epoch
            :return: number of batches per epoch
        """
        return int(np.floor(len(self.__image_names) / self.__batch_size))

    def __getitem__(self, item):
        """Generate one batch of data"""
        selected_image_names_indices = range(self.__batch_number * self.__batch_size,
                                             (self.__batch_number + 1) * self.__batch_size)
        X = self._generate_X(selected_image_names_indices)
        y = self._generate_y(selected_image_names_indices)
        self.__batch_number += 1
        if self.__batch_number >= self.__len__():
            self.__batch_number = 0
        return X, y

    def on_epoch_end(self):
        if self._shuffle:
            random.shuffle(self.__image_names)

    def _generate_X(self, selected_image_names_indices):
        channels = 1 if self.__grayscale else 3
        images = np.zeros((len(selected_image_names_indices), self.__input_size, self.__input_size, channels),
                          dtype=np.float32)
        file_index = 0
        selected_image_names = [self.__image_names[index] for index in selected_image_names_indices]
        for file in selected_image_names:
            img = cv2.imread(file)
            img = self.__image_preprocessor.preprocess_image(img)
            images[file_index] = img
            file_index += 1
        return [selected_image_names, images]

    def _generate_y(self, selected_image_names_indices):
        return self._generate_X(selected_image_names_indices)
