import os
import numpy as np
import random
import time
from data_preprocessing.prepare_data import DataLoader
from utils import DATA_REAL_PATH
from digit_detector.centernet_digit_detector import DigitDetector


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
    score_threshold = 0.5

    detector = DigitDetector(model_name='small_convnet',
                             input_shape=(input_size, input_size, channels),
                             classes_list=classes_list,
                             max_objects=max_objects,
                             resize_and_pad=False,
                             grayscale=True,
                             scale_values=1)

    detector.load_weights(model_path)

    # load the data
    data_loader = DataLoader(input_size=input_size, downsample_factor=4,
                             num_classes=num_classes, max_objects=max_objects)

    dir_ = os.path.join(DATA_REAL_PATH, 'numbers/*.png')

    batch_images, batch_hms, batch_whs, batch_regs, \
    batch_reg_masks, batch_indices = data_loader.load_from_dir(dir_, False)

    for i in range(batch_images.shape[0]):
        inputs = batch_images[i]
        inputs = np.expand_dims(inputs, axis=0)
        number = detector.recognize_number(inputs, score_threshold)
        if len(number):
            print(f"Recognized number: {number}.")
        else:
            print(f"There is no number on the image.")


if __name__ == '__main__':
    main()
