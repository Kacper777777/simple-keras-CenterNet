from centernet_detector import CenterNetDetector
import numpy as np


class DigitDetector(CenterNetDetector):
    def __init__(self, model_name, input_shape, classes_list, max_objects,
                 image_preprocessor):
        super().__init__(model_name, input_shape, classes_list, max_objects,
                         image_preprocessor)

    def recognize_number(self, inputs, score_threshold):
        detections = self.detect(inputs, score_threshold)
        detections = detections[np.argsort(detections[:, 0])]
        number = detections[:, 5]
        number = number.astype(np.int16)
        return number
