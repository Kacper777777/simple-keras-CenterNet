import cv2
import numpy as np
from data_preprocessing.padding_and_cutting import resize_and_pad
from models_factory import ModelsFactory


class CenterNetDetector:
    def __init__(self, model_name, input_shape, classes_list, max_objects,
                 resize_and_pad, grayscale, scale_values):
        self.input_shape = input_shape
        self.classes_list = classes_list
        self.num_classes = len(classes_list)
        self.max_objects = max_objects
        self.models_factory = ModelsFactory()
        self.model, self.prediction_model, self.debug_model = self.set_models(model_name)

        # preprocessing options
        self.resize_and_pad = resize_and_pad
        self.rgb_weights = [0.2989, 0.5870, 0.1140] if grayscale else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.channels = 1 if grayscale else 3
        self.scale_values = scale_values

    def set_models(self, model_name):
        return self.models_factory.get_model(model_name, self.input_shape, self.num_classes, self.max_objects)

    def load_weights(self, model_path):
        self.prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        self.debug_model.load_weights(model_path, by_name=True, skip_mismatch=True)

    def preprocess_inputs(self, inputs):
        if self.resize_and_pad:
            for i in range(len(inputs)):
                inputs[i], _ = resize_and_pad(inputs[i], (self.input_shape[0], self.input_shape[1]))
        inputs = np.dot(inputs[..., :3], self.rgb_weights)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], inputs.shape[2], self.channels))
        return inputs * self.scale_values

    def detect(self, inputs, score_threshold):
        inputs = self.preprocess_inputs(inputs)
        detections = self.prediction_model.predict_on_batch(inputs)[0]
        detections = np.array(detections)
        detections[:, 0:4] *= 4
        scores = detections[:, 4]
        # find indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]
        # select those detections
        detections = detections[indices]
        detections = detections.astype(np.float64)
        return detections
