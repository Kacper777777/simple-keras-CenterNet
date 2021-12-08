import numpy as np


class CenterNetDetector:
    def __init__(self, prediction_model, downsample_factor,
                 input_shape, classes_list, max_objects, image_preprocessor):
        self.prediction_model = prediction_model
        self.downsample_factor = downsample_factor
        self.input_shape = input_shape
        self.classes_list = classes_list
        self.num_classes = len(classes_list)
        self.max_objects = max_objects
        self.image_preprocessor = image_preprocessor

    def load_weights(self, model_path):
        self.prediction_model.load_weights(model_path, by_name=False, skip_mismatch=False)

    def detect(self, image, score_threshold):
        resize_factor = self.image_preprocessor.resize_factor(image)
        scaled_image_dims = self.image_preprocessor.scaled_image_dims(image)
        image = self.image_preprocessor.preprocess_image(image)
        image = np.expand_dims(image, axis=0)
        detections = self.prediction_model.predict_on_batch(image)[0]
        detections = np.array(detections)
        detections[:, 0:4] *= self.downsample_factor

        height_diff = self.image_preprocessor.target_height - scaled_image_dims[0]
        width_diff = self.image_preprocessor.target_width - scaled_image_dims[1]
        if height_diff != 0:
            detections[:, 1] -= (height_diff / 2)
            detections[:, 3] -= (height_diff / 2)
        elif width_diff != 0:
            detections[:, 0] -= (width_diff / 2)
            detections[:, 2] -= (width_diff / 2)

        detections[:, 0:4] *= resize_factor

        scores = detections[:, 4]
        # find indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]
        # select those detections
        detections = detections[indices]
        detections = detections.astype(np.float64)
        return detections
