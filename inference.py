import os
import random
import cv2
import numpy as np
import time
import glob
from data_preprocessing.prepare_data import DataLoader
from utils import DATA_REAL_PATH
from centernet_detector import CenterNetDetector


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    # for readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    model_path = os.path.join(DATA_REAL_PATH, 'model.h5')
    input_size = 256
    channels = 3
    classes_list = ['vehicle', 'human']
    num_classes = len(classes_list)
    max_objects = 50
    score_threshold = 0.5

    detector = CenterNetDetector(model_name='average_convnet',
                                 input_shape=(input_size, input_size, channels),
                                 classes_list=classes_list,
                                 max_objects=max_objects,
                                 resize_and_pad=False,
                                 grayscale=True,
                                 scale_values=1)

    detector.load_weights(model_path)

    # load the data
    data_loader = DataLoader(input_size=input_size, downsample_factor=4,
                             num_classes=num_classes, max_objects=max_objects, grayscale=False)

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'vehicles/*.jpg'))
    image_names = pngs + jpgs
    image_names = image_names[:50]
    random.shuffle(image_names)

    _, images, _, _, _, _, _ = data_loader.load_from_dir(image_names)

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    for i in range(images.shape[0]):
        inputs = images[i]
        inputs = np.expand_dims(inputs, axis=0)
        detections = detector.detect(inputs, score_threshold)
        output_image = images[i] * 255

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

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 3)
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            cv2.rectangle(output_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(output_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}.jpg'), images[i] * 255)
        cv2.imwrite(os.path.join(DATA_REAL_PATH, 'output', f'image_{i}_predicted.jpg'), output_image)


if __name__ == '__main__':
    main()
