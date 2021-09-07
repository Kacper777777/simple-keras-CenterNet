import os
import random
import cv2
import numpy as np
import time
import glob
from utils import DATA_REAL_PATH
from data_preprocessing.image_preprocessor import ImagePreprocessor
from digit_detector.centernet_digit_detector import DigitDetector


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
    input_size = 64
    channels = 1
    grayscale = True
    classes_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes_list)
    max_objects = 10
    score_threshold = 0.7

    image_preprocessor = ImagePreprocessor(preprocessing_strategy='resize_with_pad',
                                           target_shape=input_size,
                                           grayscale=grayscale)

    detector = DigitDetector(model_name='small_convnet',
                             input_shape=(input_size, input_size, channels),
                             classes_list=classes_list,
                             max_objects=max_objects,
                             image_preprocessor=image_preprocessor)

    detector.load_weights(os.path.join(model_path, 'model_h5_format', 'model.h5'))

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'numbers/*.png'))
    jpgs = glob.glob(os.path.join(DATA_REAL_PATH, 'numbers/*.jpg'))
    image_names = pngs + jpgs
    image_names = image_names[:100]
    random.shuffle(image_names)

    # load images
    images = [cv2.imread(file) for file in image_names]

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    # to check the correctness of recognized numbers
    annotations_file = os.path.join(DATA_REAL_PATH, 'datasets', 'numbers', 'annotations.txt')
    with open(annotations_file, 'r') as reader:
        annotations = reader.readlines()
    correctly_classified = 0

    for i in range(len(images)):
        input_image = images[i]
        output_image = images[i].copy()
        detections = detector.detect(input_image, score_threshold)

        for detection in detections:
            xmin = int(round(detection[0]))
            ymin = int(round(detection[1]))
            xmax = int(round(detection[2]))
            ymax = int(round(detection[3]))
            class_id = int(detection[5])
            color = colors[class_id]
            class_name = classes_list[class_id]
            label = class_name

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 1)
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            cv2.rectangle(output_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(output_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

        number = detector.recognize_number(input_image, score_threshold)
        name = image_names[i]
        index = int(name[name.rfind('\\') + 1:name.rfind('.')]) - 1
        actual = annotations[index][:-1]
        predicted = [str(elem) for elem in number]
        predicted = ''.join(predicted)
        if predicted == actual:
            correctly_classified += 1
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_good', f'image_{i}.jpg'), images[i])
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_good', f'image_{i}_predicted.jpg'), output_image)
        else:
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_bad', f'image_{i}.jpg'), images[i])
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_bad', f'image_{i}_actual_{actual}_predicted_{predicted}.jpg'), output_image)

    print(f"The accuracy of the model in recognizing whole numbers is: {correctly_classified/len(images)}")


if __name__ == '__main__':
    main()
