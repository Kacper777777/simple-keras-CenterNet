import cv2
import math
import random
import numpy as np
import glob
from data_preprocessing.gaussian import gaussian_radius, draw_gaussian
from data_preprocessing.padding_and_cutting import resize_and_pad


class DataLoader:
    def __init__(self, num_classes, input_shape, output_shape, max_objects):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_output_ratio = (input_shape[0]/output_shape[0], input_shape[1]/output_shape[1])
        self.max_objects = max_objects

    def load_from_dir(self, dir_):
        image_files = glob.glob(dir_)
        batch_images = np.zeros((len(image_files), self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)
        batch_hms = np.zeros((len(image_files), self.output_shape[0], self.output_shape[1], self.num_classes),
                             dtype=np.float32)
        batch_whs = np.zeros((len(image_files), self.max_objects, 2), dtype=np.float32)
        batch_regs = np.zeros((len(image_files), self.max_objects, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_files), self.max_objects), dtype=np.float32)
        batch_indices = np.zeros((len(image_files), self.max_objects), dtype=np.float32)

        file_index = 0
        for file in image_files:
            img = cv2.imread(file)
            img, scaled_image_dims = resize_and_pad(img, (self.input_shape[1], self.input_shape[0]))
            if self.input_shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
            img = img / 255
            batch_images[file_index] = img

            with open(f'{file[:-4]}.txt') as reader:
                lines = reader.readlines()
                bbox_index = 0
                for line in lines:
                    object_class, x_center, y_center, w, h = line.split(' ')
                    object_class, x_center, y_center, w, h = \
                        float(object_class), float(x_center), float(y_center), float(w), float(h)

                    x_center, y_center = x_center * scaled_image_dims[1], y_center * scaled_image_dims[0]
                    w, h = w * scaled_image_dims[1], h * scaled_image_dims[0]
                    x_center += (self.input_shape[1] - scaled_image_dims[1]) / 2
                    y_center += (self.input_shape[0] - scaled_image_dims[0]) / 2

                    x_center, y_center = x_center/self.input_output_ratio[1], y_center/self.input_output_ratio[0]
                    w, h = w/self.input_output_ratio[1], h/self.input_output_ratio[0]

                    ct = np.array([x_center, y_center], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    radius = gaussian_radius((max(math.ceil(h), 1), max(math.ceil(w), 1)), min_overlap=0.3)
                    draw_gaussian(batch_hms[file_index, :, :, int(object_class)], ct_int, int(radius), radius/2)
                    batch_whs[file_index, bbox_index] = 1. * w, 1. * h
                    batch_regs[file_index, bbox_index] = ct - ct_int
                    batch_reg_masks[file_index, bbox_index] = 1
                    batch_indices[file_index, bbox_index] = (ct_int[1] * self.output_shape[1] + ct_int[0])
                    bbox_index += 1

            file_index += 1

        return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices]
