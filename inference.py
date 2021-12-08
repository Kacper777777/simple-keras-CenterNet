import os
import random
import cv2
import numpy as np
import time
import glob
from models.squezenet import squeezenet_centernet
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
    input_size = 256
    channels = 3
    grayscale = False if channels == 3 else True
    classes_list = ['chessboard']
    num_classes = len(classes_list)
    max_objects = 1
    score_threshold = 0.8

    models = squeezenet_centernet(image_shape=(input_size, input_size, channels),
                                  num_classes=num_classes,
                                  max_objects=max_objects)
    model = models.get('train_model')
    prediction_model = models.get('prediction_model')
    debug_model = models.get('debug_model')

    image_preprocessor = ImagePreprocessor(preprocessing_strategy='resize_with_pad',
                                           target_shape=input_size,
                                           grayscale=grayscale)

    detector = CenterNetDetector(prediction_model=prediction_model,
                                 downsample_factor=4,
                                 input_shape=(input_size, input_size, channels),
                                 classes_list=classes_list,
                                 max_objects=max_objects,
                                 image_preprocessor=image_preprocessor)

    detector.load_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'))

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'datasets', 'chessboards/*.jpg'))
    image_names = pngs + jpgs
    image_names = image_names[:100]
    # random.shuffle(image_names)

    # load images
    images = [cv2.imread(file) for file in image_names]
    #images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in image_names]

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    for i in range(len(images)):
        input_image = images[i].copy()
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

        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}.jpg'), input_image)
        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}_predicted.jpg'), output_image)


if __name__ == '__main__':
    main()
