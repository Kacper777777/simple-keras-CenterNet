import tensorflow as tf
import cv2
import math
import numpy as np
import random
from data_preprocessing.gaussian import gaussian_radius, draw_gaussian
from data_preprocessing.image_preprocessor import ImagePreprocessor


class ObjectDetectionDataset:
    def __init__(self, image_names, shuffle_data, batch_size, preprocessing_strategy, input_size, grayscale,
                 downsample_factor, num_classes, max_objects):
        self.__image_names = image_names
        self.__shuffle_data = shuffle_data
        self.__batch_size = batch_size
        self.__input_size = input_size
        self.__grayscale = grayscale
        self.__image_preprocessor = ImagePreprocessor(preprocessing_strategy=preprocessing_strategy,
                                                      target_shape=input_size,
                                                      grayscale=grayscale)
        self.__downsample_factor = downsample_factor
        self.__output_size = input_size // downsample_factor
        self.__num_classes = num_classes
        self.__max_objects = max_objects
        self.__ds_size = len(self.__image_names)

    def create_datasets(self, train_ratio):
        filelist_ds = tf.data.Dataset.from_tensor_slices(self.__image_names)

        ds_train = filelist_ds.take(int(self.__ds_size * train_ratio))
        ds_test = filelist_ds.skip(int(self.__ds_size * train_ratio))

        ds_train = ds_train.shuffle(int(self.__ds_size * train_ratio))
        ds_train = ds_train.map(self._combine_images_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.batch(self.__batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.shuffle(int(self.__ds_size * (1 - train_ratio)))
        ds_test = ds_test.map(self._combine_images_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(self.__batch_size)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_test

    def _tf_helper_func(self, file_path: tf.Tensor):
        x = self._generate_X(file_path)
        y = self._generate_y()
        return x[0], x[1], x[2], x[3], x[4], x[5], y

    def _combine_images_labels(self, x):
        a, b, c, d, e, f, y = tf.py_function(self._tf_helper_func, [x],
                                             Tout=[tf.float32, tf.float32, tf.float32,
                                                   tf.float32, tf.float32, tf.float32, tf.float32])
        return (a, b, c, d, e, f), y

    def _generate_X(self, file):
        channels = 1 if self.__grayscale else 3
        image = np.zeros((self.__input_size, self.__input_size, channels),
                         dtype=np.float32)
        hms = np.zeros((self.__output_size, self.__output_size, self.__num_classes),
                       dtype=np.float32)
        whs = np.zeros((self.__max_objects, 2), dtype=np.float32)
        regs = np.zeros((self.__max_objects, 2), dtype=np.float32)
        reg_masks = np.zeros((self.__max_objects), dtype=np.float32)
        indices = np.zeros((self.__max_objects), dtype=np.float32)

        file = str(file.numpy())[2:-1]
        img = cv2.imread(file)
        scaled_image_dims = self.__image_preprocessor.scaled_image_dims(img)
        img = self.__image_preprocessor.preprocess_image(img)
        image = img

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
                draw_gaussian(hms[:, :, int(object_class)], ct_int, int(radius), radius / 2)
                whs[bbox_index] = 1. * w, 1. * h
                regs[bbox_index] = ct - ct_int
                reg_masks[bbox_index] = 1
                indices[bbox_index] = (ct_int[1] * self.__output_size + ct_int[0])
                bbox_index += 1

        return np.array(image), np.array(hms), np.array(whs), np.array(regs), \
               np.array(reg_masks), np.array(indices)

    def _generate_y(self):
        return np.zeros(1)
