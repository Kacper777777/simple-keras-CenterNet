import cv2
import numpy as np
from data_preprocessing.padding_and_cutting import resize_and_pad
from models_factory import ModelsFactory


class CenterNetDetector:
    def __init__(self, model_name, input_shape, classes_list, max_objects):
        self.input_shape = input_shape
        self.classes_list = classes_list
        self.num_classes = len(classes_list)
        self.max_objects = max_objects
        self.models_factory = ModelsFactory()
        self.model, self.prediction_model, self.debug_model = self.set_models(model_name)

    def set_models(self, model_name):
        return self.models_factory.get_model(model_name, self.input_shape, self.num_classes, self.max_objects)

    def load_weights(self, model_path):
        self.prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        self.debug_model.load_weights(model_path, by_name=True, skip_mismatch=True)

    def preprocess_inputs(self, inputs):
        for i in range(len(inputs)):
            inputs[i], scaled_image_dims = resize_and_pad(inputs[i], (self.input_shape[0], self.input_shape[1]))
            if self.input_shape[2] == 1:
                inputs[i] = cv2.cvtColor(inputs[i], cv2.COLOR_BGR2GRAY)
                inputs[i] = np.expand_dims(inputs[i], axis=-1)
        inputs = inputs / 255
        return inputs

    def detect(self, inputs, score_threshold):
        inputs = self.preprocess_inputs(inputs)
        detections = self.prediction_model.predict_on_batch(inputs)
        detections = np.array(detections)
        detections[:, 0:4] *= 4
        scores = detections[:, 4]
        # find indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]
        # select those detections
        detections = detections[indices]
        detections = detections.astype(np.float64)
        return detections
