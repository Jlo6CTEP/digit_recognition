import os

import cv2
import numpy
from numba import jit

LABEL_MN = 0x00000801
IMAGE_MN = 0x00000803

WEIGHTS_MN = 0x00000805

COUNT_1 = 784
COUNT_2 = 16
COUNT_3 = 16
COUNT_4 = 10

WEIGHTS_COUNT = COUNT_1 * COUNT_2 + COUNT_2 * COUNT_3 + COUNT_3 * COUNT_4
BIASES_COUNT = COUNT_2 + COUNT_3 + COUNT_4


def sigmoid(x):
    return 1 / (1 + numpy.e ** (-x))


def parse_images(fname_labels, fname_imgs):
    # reading labels
    """
    TEST SET LABEL
    FILE(t10k - labels - idx1 - ubyte):
    [offset]   [type]           [value]           [description]
    0000       32 bit integer   0x00000801(2049)  magic number(MSB first)
    0004       32 bit integer   10000             number of items
    0008       unsigned byte    ??                label
    0009       unsigned byte    ??                label
    ........
    xxxx       unsigned byte    ??               label

    The labels values are 0 to 9.

    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    """

    f_labels = open(os.path.join('./imgs', fname_labels), 'rb')
    assert (int.from_bytes(f_labels.read(4), byteorder='big') == LABEL_MN)
    item_number = int.from_bytes(f_labels.read(4), byteorder='big')

    lbl = numpy.fromfile(f_labels, numpy.uint8, -1, "")
    f_labels.close()

    # reading images
    f_images = open(os.path.join('./imgs', fname_imgs), 'rb')
    assert (int.from_bytes(f_images.read(4), byteorder='big') == IMAGE_MN)

    item_number = int.from_bytes(f_images.read(4), byteorder='big')
    x_axis = int.from_bytes(f_images.read(4), byteorder='big')
    y_axis = int.from_bytes(f_images.read(4), byteorder='big')

    img = numpy.empty([item_number, x_axis * y_axis], dtype=numpy.float64)

    for x in range(item_number):
        img[x] = numpy.fromfile(f_images, numpy.uint8, x_axis * y_axis, "") / 255.0

    return img, lbl


def random_weights(n_weights, n_biases):
    try:
        f = open(os.path.join('./random_coefficients', 'vector'), 'xb')
    except FileExistsError:
        f = open(os.path.join('./random_coefficients', 'vector'), 'wb')

    f.write(int.to_bytes(WEIGHTS_MN, 4, byteorder='big'))
    f.write(int.to_bytes(n_weights, 4, byteorder='big'))
    f.write(int.to_bytes(n_biases, 4, byteorder='big'))
    weights = numpy.random.uniform(-5, 5, n_weights)
    biases = numpy.random.uniform(-5, 5, n_biases)

    weights.tofile(f, "", "")
    biases.tofile(f, "", "")

    return weights, biases


def read_weights(path):
    f = open(path, 'rb')

    assert (int.from_bytes(f.read(4), byteorder='big') == WEIGHTS_MN)

    n_weights = int.from_bytes(f.read(4), byteorder='big')
    n_biases = int.from_bytes(f.read(4), byteorder='big')

    weights = numpy.fromfile(f, numpy.float64, n_weights)
    biases = numpy.fromfile(f, numpy.float64, n_biases)

    return weights, biases


def setup_pointers():
    counts = [COUNT_1, COUNT_2, COUNT_3, COUNT_4]
    nn = numpy.empty([9], dtype=numpy.ndarray)

    for x in zip(range(4), counts):
        nn[x[0]] = numpy.zeros([x[1]], dtype=numpy.float64)

    nn[4] = numpy.zeros([WEIGHTS_COUNT], dtype=numpy.float64)
    nn[5] = numpy.zeros([BIASES_COUNT], dtype=numpy.float64)

    weights_count = 0
    biases_count = 0

    for x in zip(range(6, 9), counts[1:]):
        nn[x[0]] = numpy.empty([x[1]], dtype=numpy.object)

    for count in range(6, 9):
        step = counts[count - 6]
        for x in range(len(nn[count])):
            nn[count][x] = numpy.empty([3], dtype=numpy.object)
            nn[count][x][0] = nn[count - 6]
            nn[count][x][1] = nn[4][weights_count: weights_count + step]
            nn[count][x][2] = nn[5][biases_count: biases_count + 1]

            weights_count += step
            biases_count += 1
    return nn


def load_nn(weights, biases, nn):
    numpy.copyto(nn[4], weights)
    numpy.copyto(nn[5], biases)


def update_nn(weights, biases, nn):
    nn[4] += weights
    nn[5] += biases


def evaluate_img(img_row, nn):
    numpy.copyto(nn[0], img_row)

    for count in range(6, 9):
        for x in range(len(nn[count])):
            nn[count - 5][x] = sigmoid(nn[count][x][0].dot(nn[count][x][1]) + nn[count][x][2][0])


def gradient_descend(nn, batch, labels):
    weights_grad = numpy.zeros([WEIGHTS_COUNT], dtype=numpy.float64)
    biases_grad = numpy.zeros([BIASES_COUNT], dtype=numpy.float64)
    count = 0
    for x in zip(batch, labels):
        print(str(count) + ' ', end='')
        count += 1
        tgt = numpy.full([10], 0.0, dtype=numpy.float64)
        tgt[x[1]] = 1.0
        evaluate_img(x[0], nn)

        weight_row = numpy.empty([WEIGHTS_COUNT], dtype=numpy.float64)
        bias_row = numpy.empty([BIASES_COUNT], dtype=numpy.float64)
        w_count = 0
        b_count = 0

        for j in enumerate(nn[1]):
            for l in enumerate(nn[0]):
                k_ith = 0
                for k in zip(nn[3], tgt, range(COUNT_4), nn[8]):
                    for i in zip(range(COUNT_3), nn[2], nn[7]):
                        k_ith += 2 * (k[0] - k[1]) * k[0] * (1 - k[0]) * k[3][1][i[0]] * \
                                 i[1] * (1 - i[1]) * j[1] * (1 - j[1]) * i[2][1][k[2]] * j[1] * (1 - j[1]) * l[1]
                weight_row[w_count] = k_ith
                w_count += 1

            k_ith = 0
            for k in zip(nn[3], tgt, range(COUNT_4), nn[8]):
                for i in zip(range(COUNT_3), nn[2], nn[7]):
                    k_ith += 2 * (k[0] - k[1]) * k[0] * (1 - k[0]) * k[3][1][i[0]] * \
                             i[1] * (1 - i[1]) * j[1] * (1 - j[1]) * i[2][1][k[2]] * j[1] * (1 - j[1])
            bias_row[b_count] = k_ith
            b_count += 1

        for i in enumerate(nn[2]):
            for j in enumerate(nn[1]):
                k_th = 0
                for k in zip(nn[3], tgt, range(COUNT_4), nn[8]):
                    k_th += 2 * (k[0] - k[1]) * k[0] * (1 - k[0]) * k[3][1][i[0]] * i[1] * (1 - i[1]) * j[1]
                weight_row[b_count] = k_th
                w_count += 1

            k_th = 0
            for k in zip(nn[3], tgt, range(COUNT_4), nn[8]):
                k_th += 2 * (k[0] - k[1]) * k[0] * (1 - k[0]) * k[3][1][i[0]] * i[1] * (1 - i[1])
            bias_row[b_count] = k_th
            b_count += 1

        for k in zip(nn[3], tgt):
            for i in nn[2]:
                weight_row[w_count] = 2 * (k[0] - k[1]) * k[0] * (1 - k[0]) * i
                w_count += 1

            bias_row[b_count] = 2 * (k[0] - k[1]) * k[0] * (1 - k[0])
            b_count += 1

        weights_grad += weight_row
        biases_grad += bias_row

    weights_grad /= len(batch)
    biases_grad /= len(batch)

    return - weights_grad, - biases_grad


def cost(nn, tgt):
    return sum((nn[3] - tgt) ** 2)


random_weights(WEIGHTS_COUNT, BIASES_COUNT)
weight, bias = read_weights('./random_coefficients/vector')

n_n = setup_pointers()
load_nn(weight, bias, n_n)

images, lbl = parse_images('t10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte')

success_count = 0.0

for f in range(len(images) // 100):
    base = numpy.full([10], 0.0, dtype=numpy.float64)
    base[lbl[f]] = 1.0
    n_n[4], n_n[5] = n_n[4], n_n[5] + gradient_descend(n_n, images[f * 100: (f + 1) * 100], lbl[f * 100: (f + 1) * 100])
    print(cost(n_n, base))
# print("Success ratio: {}%".format(success_count / len(images) * 100))
