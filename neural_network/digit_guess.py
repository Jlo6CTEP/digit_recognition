import argparse
import cv2
import math
import numpy

from neural_network.io_images import ImageIO
from neural_network.network import Network

BOX_SIZE = 20
IMG_SIZE = 28
PAD = 4

parser = argparse.ArgumentParser(description="Guess what digit is it")
parser.add_argument("img", type=str, help="image with digit")
img_path = parser.parse_args().img

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

mean_x_axis = numpy.argwhere(img.mean(axis=0) != 255.0).flatten()
mean_y_axis = numpy.argwhere(img.mean(axis=1) != 255.0).flatten()

top_right = (mean_x_axis[len(mean_x_axis) - 1], mean_y_axis[len(mean_y_axis) - 1])
bottom_left = (mean_x_axis[0], mean_y_axis[0])

digit = img[bottom_left[1]: top_right[1] + 1, bottom_left[0]: top_right[0] + 1]

scaling = BOX_SIZE / max(digit.shape)

digit_norm = cv2.resize(digit, tuple((numpy.array(digit.shape[::-1]) * scaling).astype(dtype=int)), cv2.INTER_AREA)

cm_x = int(numpy.argwhere(digit_norm.mean(axis=0) != 255.0).flatten().mean())
cm_y = int(numpy.argwhere(digit_norm.mean(axis=1) != 255.0).flatten().mean())

target = numpy.full([IMG_SIZE, IMG_SIZE], 255)

offset = numpy.array([IMG_SIZE, IMG_SIZE]) // 2 - numpy.array([cm_x, cm_y])

target[offset[1]: digit_norm.shape[0] + offset[1], offset[0]: digit_norm.shape[1] + offset[0]] = digit_norm
target = target.astype(dtype=numpy.uint8)

nn = Network(64, 64, 3, 50, 100, True)

test_images = ImageIO(0)

cm_x = int(numpy.argwhere(target.mean(axis=0) != 255.0).flatten().mean())
cm_y = int(numpy.argwhere(target.mean(axis=1) != 255.0).flatten().mean())

final_image = (255 - target) / 255.0

cv2.imshow('lol', final_image)


print(nn.evaluate_img(final_image.flatten()))
cv2.waitKey()
