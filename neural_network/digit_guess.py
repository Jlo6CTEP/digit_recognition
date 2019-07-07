import cv2
import numpy

from neural_network.io_images import ImageIO
from neural_network.network import Network

# size of the box to fit digit in
BOX_SIZE = 20
# resulting image size
IMG_SIZE = 28

# reject images with less than 50 filled pixels
IMG_REJECT_THRESHOLD = 50

nn = Network(64, 64, 3, 50, 100, True)
i_io = ImageIO(2)


def guess_digit(img):
    if numpy.count_nonzero(img) <= IMG_REJECT_THRESHOLD:
        return

    # first form the bounding box the digit fits in
    mean_x_axis = numpy.argwhere(img.mean(axis=0) != 0.0).flatten()
    mean_y_axis = numpy.argwhere(img.mean(axis=1) != 0.0).flatten()
    top_right = (mean_x_axis[len(mean_x_axis) - 1], mean_y_axis[len(mean_y_axis) - 1])
    bottom_left = (mean_x_axis[0], mean_y_axis[0])

    # then rescale digit to have length or width equal to 20
    digit = img[bottom_left[1]: top_right[1] + 1, bottom_left[0]: top_right[0] + 1]
    scaling = BOX_SIZE / max(digit.shape)
    digit_norm = cv2.resize(digit, tuple((numpy.array(digit.shape[::-1]) * scaling).astype(dtype=int)), cv2.INTER_AREA)

    # compute the center of mass of the digit
    bin_x = numpy.sum(digit_norm > 100, axis=0)
    bin_y = numpy.sum(digit_norm > 100, axis=1)
    cm_x = int(sum(bin_x * numpy.array(range(digit_norm.shape[1])) / sum(bin_x)))
    cm_y = int(sum(bin_y * numpy.array(range(digit_norm.shape[0])) / sum(bin_y)))

    # position the digit such that it's center of mass is in the center of IMG_SIZE x IMG_SIZE image
    target = numpy.full([IMG_SIZE, IMG_SIZE], 0)
    offset = numpy.array([IMG_SIZE, IMG_SIZE]) // 2 - numpy.array([cm_x, cm_y])
    shaper = target[offset[1]: digit_norm.shape[0] + offset[1], offset[0]: digit_norm.shape[1] + offset[0]]
    target[offset[1]: digit_norm.shape[0] + offset[1], offset[0]: digit_norm.shape[1] + offset[0]] = \
        digit_norm[0:shaper.shape[0], 0:shaper.shape[1]]
    target = target.astype(dtype=numpy.uint8)

    # convert image to 0 to 1 floating point image
    final_image = target / 255.0

    # add image into my_dataset
    i_io.append_image(target)
    return nn.evaluate_img(final_image.flatten())
