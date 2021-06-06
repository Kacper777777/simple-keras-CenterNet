from centernet_detector import CenterNetDetector
import numpy as np


class NumbersDetector(CenterNetDetector):
    def __init__(self, model_name, input_shape, classes_list, max_objects,
                 resize_and_pad, grayscale, scale_values):
        super().__init__(model_name, input_shape, classes_list, max_objects,
                         resize_and_pad, grayscale, scale_values)

    def recognize_number(self, inputs, score_threshold):
        detections = self.detect(inputs, score_threshold)
        detections = detections[np.argsort(detections[:, 0])]
        numbers = detections[:, 5]
        numbers = numbers.astype(np.int16)
        return numbers

    def evaluate_model_performance(self, images, annotations):
        pass
