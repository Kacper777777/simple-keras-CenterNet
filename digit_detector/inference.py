import os
import random
import cv2
import numpy as np
import time
import glob
from data_preprocessing.prepare_data import DataLoader
from utils import DATA_REAL_PATH
from digit_detector.centernet_digit_detector import DigitDetector


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    # for readability
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    model_path = os.path.join(DATA_REAL_PATH, 'model.h5')
    input_size = 64
    channels = 1
    classes_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes_list)
    max_objects = 10
    score_threshold = 0.5

    detector = DigitDetector(model_name='small_convnet',
                             input_shape=(input_size, input_size, channels),
                             classes_list=classes_list,
                             max_objects=max_objects,
                             resize_and_pad=False,
                             grayscale=True,
                             scale_values=1)

    detector.load_weights(model_path)

    # load the data
    data_loader = DataLoader(input_size=input_size, downsample_factor=4,
                             num_classes=num_classes, max_objects=max_objects)

    pngs = glob.glob(os.path.join(DATA_REAL_PATH, 'numbers/*.png'))
    image_names = pngs

    image_names, images, _, _, _, _, _ = data_loader.load_from_dir(image_names)

    # to check the correctness of recognized numbers
    annotations_file = os.path.join(DATA_REAL_PATH, 'numbers', 'annotations.txt')
    with open(annotations_file, 'r') as reader:
        annotations = reader.readlines()

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    correctly_classified = 0

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
            class_id = int(detection[5])
            color = colors[class_id]
            class_name = classes_list[class_id]
            label = class_name

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 1)
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            cv2.rectangle(output_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(output_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

        number = detector.recognize_number(inputs, score_threshold)
        name = image_names[i]
        index = int(name[name.rfind('\\') + 1:name.rfind('.')]) - 1
        actual = annotations[index][:-1]
        predicted = [str(elem) for elem in number]
        predicted = ''.join(predicted)
        if predicted == actual:
            correctly_classified += 1
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_good', f'image_{i}.jpg'), images[i] * 255)
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_good', f'image_{i}_predicted.jpg'), output_image)
        else:
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_bad', f'image_{i}.jpg'), images[i] * 255)
            cv2.imwrite(os.path.join(DATA_REAL_PATH, 'digits_output_bad', f'image_{i}_actual_{actual}_predicted_{predicted}.jpg'), output_image)

    print(f"The accuracy of the model in recognizing whole numbers is: {correctly_classified/images.shape[0]}")


if __name__ == '__main__':
    main()
