import os
import random

import numpy

LABEL_MN = 0x00000801
IMAGE_MN = 0x00000803

WEIGHTS_MN = 0x00000805

B_SIZE = 100

EPOCH_COUNT = 32

L_1 = 784
L_2 = 64
L_3 = 64
L_4 = 10

WEIGHTS_COUNT = L_1 * L_2 + L_2 * L_3 + L_3 * L_4
BIASES_COUNT = L_2 + L_3 + L_4

MODE = 0

TRAINED

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


def write_out_weights(weights, biases, preffix):
    try:
        f = open(os.path.join(preffix, 'vector'), 'xb')
    except FileExistsError:
        f = open(os.path.join(preffix, 'vector'), 'wb')

    f.write(int.to_bytes(WEIGHTS_MN, 4, byteorder='big'))
    f.write(int.to_bytes(len(weights), 4, byteorder='big'))
    f.write(int.to_bytes(len(biases), 4, byteorder='big'))

    weights.tofile(f, "", "")
    biases.tofile(f, "", "")

    return weights, biases


def random_weights(n_weights, n_biases):
    weights = numpy.random.uniform(-1, 1, n_weights)
    biases = numpy.random.uniform(-1, 1, n_biases)

    return write_out_weights(weights, biases, './random_coefficients')


def read_weights(path):
    f = open(path, 'rb')

    assert (int.from_bytes(f.read(4), byteorder='big') == WEIGHTS_MN)

    n_weights = int.from_bytes(f.read(4), byteorder='big')
    n_biases = int.from_bytes(f.read(4), byteorder='big')

    weights = numpy.fromfile(f, numpy.float64, n_weights)
    biases = numpy.fromfile(f, numpy.float64, n_biases)

    return weights, biases


def load_nn(weights, biases):
    return [
        numpy.zeros(L_1),
        numpy.zeros(L_2),
        numpy.zeros(L_3),
        numpy.zeros(L_4),

        numpy.reshape(weights[0: L_1 * L_2], (-1, L_2)),
        numpy.reshape(weights[L_1 * L_2: L_1 * L_2 + L_2 * L_3], (-1, L_3)),
        numpy.reshape(weights[L_1 * L_2 + L_2 * L_3: WEIGHTS_COUNT], (-1, L_4)),

        biases[0: L_2],
        biases[L_2: L_2 + L_3],
        biases[L_2 + L_3: BIASES_COUNT]
    ]


def evaluate_img(img_row, nn):
    numpy.copyto(nn[0], img_row)

    for x in range(0, 3):
        numpy.copyto(nn[x + 1], sigmoid(nn[x].dot(nn[x + 4]) + nn[x + 7]))
    return numpy.argmax(nn[3])


def gradient_descend(batch, labels, nn):
    a_1, a_2, a_3, a_4, w_lj, w_ji, w_ik, b_2, b_3, b_4 = nn

    b_2t, b_3t, b_4t = numpy.zeros(L_2), numpy.zeros(L_3), numpy.zeros(L_4)
    w_lj_t, w_ji_t, w_ik_t = numpy.zeros([L_1, L_2]), numpy.zeros([L_2, L_3]), numpy.zeros([L_3, L_4]),

    count = 0
    for sample in range(len(batch)):
        count += 1
        tgt = numpy.zeros(10)
        tgt[labels[sample]] = 1.0
        evaluate_img(batch[sample], nn)

        a = 2 * (a_4 - tgt) * a_4 * (1 - a_4)
        b = (2 * (a_4 - tgt) * a_4 * (1 - a_4)).dot(w_ik.transpose()) * (a_3 * (1 - a_3))
        c = (2 * (a_4 - tgt) * a_4 * (1 - a_4)).dot(w_ik.transpose()) * \
            (a_3 * (1 - a_3)).dot(w_ji.transpose()) * (a_2 * (1 - a_2))

        b_2t += c
        b_3t += b
        b_4t += a

        w_lj_t += c.reshape((-1, 1)).dot([a_1]).transpose()
        w_ji_t += b.reshape((-1, 1)).dot([a_2]).transpose()
        w_ik_t += a.reshape((-1, 1)).dot([a_3]).transpose()

    for x in zip(range(4, 10), [w_lj_t, w_ji_t, w_ik_t, b_2t, b_3t, b_4t]):
        nn[x[0]] -= x[1] / len(batch) * 3


def cost(nn, tgt):
    return sum((nn[3] - tgt) ** 2)


random_weights(WEIGHTS_COUNT, BIASES_COUNT)
weight, bias = read_weights()

n_n = load_nn(weight, bias)

train_images = list(zip(*parse_images('train-labels-idx1-ubyte', 'train-images-idx3-ubyte')))
test_images = list(zip(*parse_images('t10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte')))

print()
for epoch in range(EPOCH_COUNT):
    random.shuffle(train_images)
    for f in range(len(train_images) // B_SIZE):
        gradient_descend(*zip(*train_images[f * B_SIZE: (f + 1) * B_SIZE]), n_n)
        if f % 120 == 0:
            print("Epoch {} finished {}% percents".format(epoch, f * 100 * B_SIZE / len(train_images)))
    print("Epoch {} just ended".format(epoch))

    success_count = 0.0
    for f in test_images:
        if evaluate_img(f[0], n_n) == f[1]:
            success_count += 1.0
    print("Success ratio: {}%".format(success_count / len(test_images) * 100))

write_out_weights(numpy.concatenate([x.flatten() for x in n_n[4:7]]), numpy.concatenate(n_n[7:10]),
                  './trained_coefficients')
