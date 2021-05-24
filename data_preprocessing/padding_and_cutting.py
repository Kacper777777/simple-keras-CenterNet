import cv2
import numpy as np


def resize_and_pad(image, target_size):
    image_ratio = image.shape[1] / image.shape[0]
    target_ratio = target_size[1] / target_size[0]

    input_width, input_height = image.shape[1], image.shape[0]
    target_width, target_height = target_size[1], target_size[0]

    width_ratio = input_width / target_width
    height_ratio = input_height / target_height

    scale_factor = max(width_ratio, height_ratio)

    image = cv2.resize(image, (round(image.shape[1] / scale_factor), round(image.shape[0] / scale_factor)))
    scaled_image_dimensions = (image.shape[0], image.shape[1])
    final_image = np.zeros((target_size[0], target_size[1], 3), np.uint8)

    if target_ratio > image_ratio:
        # pad horizontally (the black bars will appear on the left and right side)
        to_pad = int((target_size[1] - image.shape[1]) / 2)
        final_image[:, to_pad:to_pad + image.shape[1]] = image[:, :]
    else:
        # pad vertically (the black bars will appear on the top and bottom)
        to_pad = int((target_size[0] - image.shape[0]) / 2)
        final_image[to_pad:to_pad + image.shape[0], :] = image[:, :]

    return final_image, scaled_image_dimensions


def resize_and_cut(image, target_size):
    image_ratio = image.shape[1] / image.shape[0]
    target_ratio = target_size[1] / target_size[0]

    input_width, input_height = image.shape[1], image.shape[0]
    target_width, target_height = target_size[1], target_size[0]

    width_ratio = input_width / target_width
    height_ratio = input_height / target_height

    scale_factor = min(width_ratio, height_ratio)
    print("scale_factor", scale_factor)

    image = cv2.resize(image, (round(image.shape[1] / scale_factor), round(image.shape[0] / scale_factor)))
    scaled_image_dimensions = (image.shape[0], image.shape[1])
    final_image = np.zeros((target_size[0], target_size[1], 3), np.uint8)

    if target_ratio > image_ratio:
        # cut vertically (top and bottom)
        to_cut = int((image.shape[0] - target_size[0]) / 2)
        final_image[:, :] = image[to_cut:to_cut + target_size[0], :]
    else:
        # cut horizontally (left and right side)
        to_cut = int((image.shape[1] - target_size[1]) / 2)
        final_image[:, :] = image[:, to_cut:to_cut + target_size[1]]

    return final_image, scaled_image_dimensions


if __name__ == '__main__':
    pass
