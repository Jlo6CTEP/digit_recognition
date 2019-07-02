import cv2
import numpy

from neural_network.io_images import ImageIO
from neural_network.network import Network

BOX_SIZE = 20
IMG_SIZE = 28
PAD = 4

nn = Network(64, 64, 3, 50, 100, True)
i_io = ImageIO(2)


def guess_digit(img):
    mean_x_axis = numpy.argwhere(img.mean(axis=0) != 0.0).flatten()
    mean_y_axis = numpy.argwhere(img.mean(axis=1) != 0.0).flatten()

    top_right = (mean_x_axis[len(mean_x_axis) - 1], mean_y_axis[len(mean_y_axis) - 1])
    bottom_left = (mean_x_axis[0], mean_y_axis[0])

    digit = img[bottom_left[1]: top_right[1] + 1, bottom_left[0]: top_right[0] + 1]

    scaling = BOX_SIZE / max(digit.shape)

    digit_norm = cv2.resize(digit, tuple((numpy.array(digit.shape[::-1]) * scaling).astype(dtype=int)), cv2.INTER_AREA)

    bin_x = numpy.sum(digit_norm > 100, axis=0)
    bin_y = numpy.sum(digit_norm > 100, axis=1)

    cm_x = int(sum(bin_x * numpy.array(range(digit_norm.shape[1])) / sum(bin_x)))
    cm_y = int(sum(bin_y * numpy.array(range(digit_norm.shape[0])) / sum(bin_y)))

    target = numpy.full([IMG_SIZE, IMG_SIZE], 0)

    offset = numpy.array([IMG_SIZE, IMG_SIZE]) // 2 - numpy.array([cm_x, cm_y])

    target[offset[1]: digit_norm.shape[0] + offset[1], offset[0]: digit_norm.shape[1] + offset[0]] = digit_norm
    target = target.astype(dtype=numpy.uint8)

    final_image = target / 255.0

    i_io.append_image(target)

    return nn.evaluate_img(final_image.flatten())
