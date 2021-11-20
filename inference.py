import os
import random
import cv2
import numpy as np
import time
import glob
from models import googlenet
from utils import DATA_REAL_PATH
from data_preprocessing.image_preprocessor import ImagePreprocessor
from centernet_detector import CenterNetDetector


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    # for readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    model_path = os.path.join(DATA_REAL_PATH, 'NEWEST_MODEL')
    checkpoint_path = os.path.join(model_path, 'checkpoint_dir', 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    input_size = 512
    channels = 3
    grayscale = False if channels == 3 else True
    classes_list = ['vehicle', 'human']
    num_classes = len(classes_list)
    max_objects = 100
    score_threshold = 0.7

    autoencoder_model, model, prediction_model, debug_model = googlenet(image_shape=(input_size, input_size, channels),
                                                                        num_classes=num_classes,
                                                                        max_objects=max_objects)

    image_preprocessor = ImagePreprocessor(preprocessing_strategy='resize_with_pad',
                                           target_shape=input_size,
                                           grayscale=grayscale)

    detector = CenterNetDetector(model=model,
                                 prediction_model=prediction_model,
                                 debug_model=debug_model,
                                 downsample_factor=4,
                                 input_shape=(input_size, input_size, channels),
                                 classes_list=classes_list,
                                 max_objects=max_objects,
                                 image_preprocessor=image_preprocessor)

    detector.load_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'))

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.jpg'))
    image_names = pngs + jpgs
    image_names = image_names[:100]
    random.shuffle(image_names)

    # load images
    images = [cv2.imread(file) for file in image_names]

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    for i in range(len(images)):
        input_image = images[i]
        output_image = images[i].copy()
        detections = detector.detect(input_image, score_threshold)

        for detection in detections:
            xmin = int(round(detection[0]))
            ymin = int(round(detection[1]))
            xmax = int(round(detection[2]))
            ymax = int(round(detection[3]))
            score = '{:.2f}'.format(detection[4])
            class_id = int(detection[5])
            color = colors[class_id]
            class_name = classes_list[class_id]
            label = '-'.join([class_name, score])

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            cv2.rectangle(output_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(output_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}.jpg'), images[i])
        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}_predicted.jpg'), output_image)


if __name__ == '__main__':
    main()
