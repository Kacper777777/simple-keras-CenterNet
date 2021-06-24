import numpy as np
import tensorflow as tf


class ImagePreprocessor:
    def __init__(self, preprocessing_strategy, target_shape, grayscale,
                 scale_factor=1/255):

        # choose preprocessing strategy
        self.__preprocessing_strategy = preprocessing_strategy
        self.preprocessing_functions = {'resize': self.__resize,
                                        'resize_with_pad': self.__resize_with_pad}
        if preprocessing_strategy not in self.preprocessing_functions:
            raise Exception('Incorrect preprocessing strategy (should be "resize" or "resize_with_pad")')
        else:
            self.preprocessing_func = self.preprocessing_functions.get(preprocessing_strategy)

        # set target shape
        if type(target_shape) is int:
            self.__target_height = self.__target_width = target_shape
        elif type(target_shape) is tuple:
            self.__target_height = target_shape[0]
            self.__target_width = target_shape[1]
        else:
            raise Exception('Incorrect target shape (should be int or tuple of 2 integers)')

        # color or grayscale images
        self.__rgb_weights = [0.2989, 0.5870, 0.1140] if grayscale else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.__channels = 1 if grayscale else 3

        # scale factor used to normalize the data
        self.__scale_factor = scale_factor

    @property
    def preprocessing_strategy(self):
        return self.__preprocessing_strategy

    @property
    def target_height(self):
        return self.__target_height

    @property
    def target_width(self):
        return self.__target_width

    @property
    def channels(self):
        return self.__channels

    @property
    def scale_factor(self):
        return self.__scale_factor

    def preprocess_image(self, image):
        image = np.dot(image[..., :3], self.__rgb_weights)
        image = image.reshape((image.shape[0], image.shape[1], self.__channels))
        return self.preprocessing_func(image) * self.__scale_factor

    def scaled_image_dims(self, image):
        height_ratio = image.shape[0] / self.__target_height
        width_ratio = image.shape[1] / self.__target_width
        scale_factor = max(height_ratio, width_ratio)
        return round(image.shape[0] / scale_factor), round(image.shape[1] / scale_factor)

    def __resize(self, image):
        return np.array(tf.image.resize(image, [self.__target_height, self.__target_width]))

    def __resize_with_pad(self, image):
        return np.array(tf.image.resize_with_pad(image, self.__target_height, self.__target_width))
