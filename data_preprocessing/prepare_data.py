import cv2
import math
import random
import numpy as np
import glob
from data_preprocessing.gaussian import gaussian_radius, draw_gaussian
from data_preprocessing.padding_and_cutting import resize_and_pad


class DataLoader:
    def __init__(self, input_size, downsample_factor, num_classes, max_objects, grayscale=False):
        self.input_size = input_size
        self.downsample_factor = downsample_factor
        self.output_size = input_size // downsample_factor
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.grayscale = grayscale

    def load_from_dir(self, dir_, shuffle):
        image_names = glob.glob(dir_)
        if shuffle:
            random.shuffle(image_names)
        channels = 1 if self.grayscale else 3
        images = np.zeros((len(image_names), self.input_size, self.input_size, channels),
                          dtype=np.float32)
        hms = np.zeros((len(image_names), self.output_size, self.output_size, self.num_classes),
                             dtype=np.float32)
        whs = np.zeros((len(image_names), self.max_objects, 2), dtype=np.float32)
        regs = np.zeros((len(image_names), self.max_objects, 2), dtype=np.float32)
        reg_masks = np.zeros((len(image_names), self.max_objects), dtype=np.float32)
        indices = np.zeros((len(image_names), self.max_objects), dtype=np.float32)

        file_index = 0
        for file in image_names:
            img = cv2.imread(file)
            img, scaled_image_dims = resize_and_pad(img, (self.input_size, self.input_size))
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
            img = img / 255
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
                    x_center += (self.input_size - scaled_image_dims[1]) / 2
                    y_center += (self.input_size - scaled_image_dims[0]) / 2

                    x_center, y_center = x_center/self.downsample_factor, y_center/self.downsample_factor
                    w, h = w/self.downsample_factor, h/self.downsample_factor

                    ct = np.array([x_center, y_center], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    radius = gaussian_radius((max(math.ceil(h), 1), max(math.ceil(w), 1)), min_overlap=0.3)
                    draw_gaussian(hms[file_index, :, :, int(object_class)], ct_int, int(radius), radius/2)
                    whs[file_index, bbox_index] = 1. * w, 1. * h
                    regs[file_index, bbox_index] = ct - ct_int
                    reg_masks[file_index, bbox_index] = 1
                    indices[file_index, bbox_index] = (ct_int[1] * self.output_size + ct_int[0])
                    bbox_index += 1

            file_index += 1

        return [image_names, images, hms, whs, regs, reg_masks, indices]
