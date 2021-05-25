import os
import random
import cv2
import numpy as np
import time
import glob
from data_preprocessing.prepare_data import DataLoader
from models import googlenet, small_conv
from data_preprocessing.padding_and_cutting import resize_and_pad
from utils import DATA_REAL_PATH


def main():
    # for reproducibility
    np.random.seed(42)
    random.seed(42)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # configuration
    WORKING_DIR = DATA_REAL_PATH
    model_path = os.path.join(WORKING_DIR, 'model.h5')
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes)
    score_threshold = 0.7
    input_shape = (64, 64, 1)
    output_shape = (16, 16)
    max_objects = 20

    # build the model
    model, prediction_model, debug_model = small_conv(input_shape=input_shape,
                                                      output_shape=output_shape,
                                                      num_classes=num_classes,
                                                      max_objects=max_objects)

    # load weights
    prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
    debug_model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # load the data
    data_loader = DataLoader(num_classes, input_shape, output_shape, max_objects)

    dir_ = os.path.join(WORKING_DIR, 'numbers/*.png')

    batch_images, batch_hms, batch_whs, batch_regs, \
    batch_reg_masks, batch_indices = data_loader.load_from_dir(dir_)

    image_files = glob.glob(dir_)
    image_files = image_files[:20]
    colour_images = [cv2.imread(file) for file in image_files]
    colour_images = [resize_and_pad(img, (input_shape[0], input_shape[1]))[0] for img in colour_images]

    print(batch_images.shape)

    # for visualization
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    for i in range(batch_images.shape[0]):
        inputs = batch_images[i]
        inputs = np.expand_dims(inputs, axis=0)
        src_image = colour_images[i]

        detect = True
        if detect:
            # run network
            start = time.time()
            detections = prediction_model.predict_on_batch(inputs)[0]
            detections = np.array(detections)
            print(time.time() - start)
            scores = detections[:, 4]
            # find indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            # select those detections
            detections = detections[indices]
            detections = detections.astype(np.float64)

            for detection in detections:
                multiplier = (input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])
                xmin = int((round(detection[0])) * multiplier[1])
                ymin = int((round(detection[1])) * multiplier[0])
                xmax = int((round(detection[2])) * multiplier[1])
                ymax = int((round(detection[3])) * multiplier[0])
                score = '{:.4f}'.format(detection[4])
                class_id = int(detection[5])
                color = colors[class_id]
                class_name = classes[class_id]
                label = '-'.join([class_name, score])

                cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
                ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
                cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
                cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

            cv2.imwrite(os.path.join(WORKING_DIR, 'output', f'image_original{i}.jpg'), src_image)
            cv2.imwrite(os.path.join(WORKING_DIR, 'output', f'image_with_boxes{i}.jpg'), batch_images[i] * 255)


if __name__ == '__main__':
    main()
