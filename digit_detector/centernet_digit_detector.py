from centernet_detector import CenterNetDetector
import numpy as np


class DigitDetector(CenterNetDetector):
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

    def evaluate_model_performance(self, images, annotations, score_threshold):
        correct_predictions = 0
        for i in range(images.shape[0]):
            inputs = images[i]
            inputs = np.expand_dims(inputs, axis=0)
            number = self.recognize_number(inputs, score_threshold)
            if number == annotations[i]:
                correct_predictions += 1
        return correct_predictions / images.shape[0]
