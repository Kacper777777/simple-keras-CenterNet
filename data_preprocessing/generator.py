import cv2
import math
import numpy as np
import random
from data_preprocessing.gaussian import gaussian_radius, draw_gaussian
from data_preprocessing.image_preprocessor import ImagePreprocessor
# from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence


class CenterNetGenerator(Sequence):
    def __init__(self, shuffle, preprocessing_strategy, input_size, grayscale, downsample_factor,
                 num_classes, max_objects,
                 image_names, batch_size):
        self.__shuffle = shuffle
        self.__input_size = input_size
        self.__grayscale = grayscale
        self.__image_preprocessor = ImagePreprocessor(preprocessing_strategy=preprocessing_strategy,
                                                      target_shape=self.__input_size,
                                                      grayscale=grayscale)
        self.__downsample_factor = downsample_factor
        self.__output_size = input_size // downsample_factor
        self.__num_classes = num_classes
        self.__max_objects = max_objects
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
        selected_image_names_indices = list(range(self.__batch_number * self.__batch_size,
                                                  (self.__batch_number + 1) * self.__batch_size))
        X = self._generate_X(selected_image_names_indices)
        y = self._generate_y(selected_image_names_indices)
        self.__batch_number += 1
        if self.__batch_number >= self.__len__():
            self.__batch_number = 0
        return X, y

    def on_epoch_end(self):
        if self.__shuffle:
            random.shuffle(self.__image_names)

    def _generate_X(self, selected_image_names_indices):
        channels = 1 if self.__grayscale else 3
        images = np.zeros((len(selected_image_names_indices), self.__input_size, self.__input_size, channels),
                          dtype=np.float32)
        hms = np.zeros((len(selected_image_names_indices), self.__output_size, self.__output_size, self.__num_classes),
                       dtype=np.float32)
        whs = np.zeros((len(selected_image_names_indices), self.__max_objects, 2), dtype=np.float32)
        regs = np.zeros((len(selected_image_names_indices), self.__max_objects, 2), dtype=np.float32)
        reg_masks = np.zeros((len(selected_image_names_indices), self.__max_objects), dtype=np.float32)
        indices = np.zeros((len(selected_image_names_indices), self.__max_objects), dtype=np.float32)

        file_index = 0
        selected_image_names = [self.__image_names[index] for index in selected_image_names_indices]
        for file in selected_image_names:
            img = cv2.imread(file)
            scaled_image_dims = self.__image_preprocessor.scaled_image_dims(img)
            img = self.__image_preprocessor.preprocess_image(img)
            images[file_index] = img

            with open(f'{file[:-4]}.txt') as reader:
                lines = reader.readlines()
                bbox_index = 0
                for line in lines:
                    object_class, x_center, y_center, w, h = line.split(' ')
                    object_class, x_center, y_center, w, h = \
                        float(object_class), float(x_center), float(y_center), float(w), float(h)

                    x_center, y_center = x_center * scaled_image_dims[1], y_center * scaled_image_dims[0]
                    w, h = w * scaled_image_dims[1], h * scaled_image_dims[0]
                    x_center += (self.__input_size - scaled_image_dims[1]) / 2
                    y_center += (self.__input_size - scaled_image_dims[0]) / 2

                    x_center, y_center = x_center / self.__downsample_factor, y_center / self.__downsample_factor
                    w, h = w / self.__downsample_factor, h / self.__downsample_factor

                    ct = np.array([x_center, y_center], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    radius = gaussian_radius((max(math.ceil(h), 1), max(math.ceil(w), 1)), min_overlap=0.3)
                    draw_gaussian(hms[file_index, :, :, int(object_class)], ct_int, int(radius), radius / 2)
                    whs[file_index, bbox_index] = 1. * w, 1. * h
                    regs[file_index, bbox_index] = ct - ct_int
                    reg_masks[file_index, bbox_index] = 1
                    indices[file_index, bbox_index] = (ct_int[1] * self.__output_size + ct_int[0])
                    bbox_index += 1

            file_index += 1

        return np.array(images), np.array(hms), np.array(whs), np.array(regs), \
               np.array(reg_masks), np.array(indices)

    def _generate_y(self, selected_image_names_indices):
        return np.zeros(len(selected_image_names_indices))
