import random

import numpy

from neural_network.io_images import ImageIO
from neural_network.io_weights import WeightIO

L_1 = 784
L_4 = 10


def sigmoid(x):
    return 1 / (1 + numpy.e ** (-x))


class Network:
    nn = None
    l_2 = None
    l_3 = None
    eta = None
    epoch_count = None
    batch_size = None

    weights_count = None
    biases_count = None

    def __init__(self, l_2, l_3, eta, epoch_count, batch_size, is_trained=False):
        self.l_2, self.l_3, self.eta, self.epoch_count, self.batch_size = l_2, l_3, eta, epoch_count, batch_size
        self.weights_count = L_1 * l_2 + l_2 * l_3 + l_3 * L_4
        self.biases_count = l_2 + l_3 + L_4

        if is_trained:
            train_weights = WeightIO(1, self.weights_count, self.biases_count)
            self.__load_nn(train_weights.weights, train_weights.biases)

    def __load_nn(self, weights, biases):
        self.nn = [
            numpy.zeros(L_1),
            numpy.zeros(self.l_2),
            numpy.zeros(self.l_3),
            numpy.zeros(L_4),

            numpy.reshape(weights[0: L_1 * self.l_2], (-1, self.l_2)),
            numpy.reshape(weights[L_1 * self.l_2: L_1 * self.l_2 + self.l_2 * self.l_3], (-1, self.l_3)),
            numpy.reshape(weights[L_1 * self.l_2 + self.l_2 * self.l_3: self.weights_count], (-1, L_4)),

            biases[0: self.l_2],
            biases[self.l_2: self.l_2 + self.l_3],
            biases[self.l_2 + self.l_3: self.biases_count]
        ]

    def evaluate_img(self, image_row):
        numpy.copyto(self.nn[0], image_row)

        for x in range(0, 3):
            numpy.copyto(self.nn[x + 1], sigmoid(self.nn[x].dot(self.nn[x + 4]) + self.nn[x + 7]))
        return numpy.argmax(self.nn[3])

    def __gradient_descend(self, batch, labels):
        a_1, a_2, a_3, a_4, w_lj, w_ji, w_ik, b_2, b_3, b_4 = self.nn

        b_2t, b_3t, b_4t = numpy.zeros(self.l_2), numpy.zeros(self.l_3), numpy.zeros(L_4)
        w_lj_t, w_ji_t, w_ik_t = numpy.zeros([L_1, self.l_2]), numpy.zeros([self.l_2, self.l_3]), \
                                 numpy.zeros([self.l_3, L_4]),

        count = 0
        for sample in range(len(batch)):
            count += 1
            tgt = numpy.zeros(10)
            tgt[labels[sample]] = 1.0
            self.evaluate_img(batch[sample])

            a = 2 * (a_4 - tgt) * a_4 * (1 - a_4)
            b = (2 * (a_4 - tgt) * a_4 * (1 - a_4)).dot(w_ik.T) * (a_3 * (1 - a_3))
            c = (2 * (a_4 - tgt) * a_4 * (1 - a_4)).dot(w_ik.T) * (a_3 * (1 - a_3)).dot(w_ji.T) * (a_2 * (1 - a_2))

            b_2t += c
            b_3t += b
            b_4t += a

            w_lj_t += c.reshape((-1, 1)).dot([a_1]).T
            w_ji_t += b.reshape((-1, 1)).dot([a_2]).T
            w_ik_t += a.reshape((-1, 1)).dot([a_3]).T

        for x in zip(range(4, 10), [w_lj_t, w_ji_t, w_ik_t, b_2t, b_3t, b_4t]):
            self.nn[x[0]] -= x[1] / len(batch) * self.eta

    def train_network(self):
        train_weights = WeightIO(0, self.weights_count, self.biases_count)
        self.__load_nn(train_weights.weights, train_weights.biases)

        train_images = ImageIO(1)
        test_images = ImageIO(0)
        for epoch in range(self.epoch_count):
            random.shuffle(train_images.train_set)
            for f in range(len(train_images.train_set) // self.batch_size):
                self.__gradient_descend(*zip(*train_images.train_set[f * self.batch_size: (f + 1) * self.batch_size]))
                if f % 120 == 0:
                    print("Epoch {} finished {}% percents"
                          .format(epoch, f * 100 * self.batch_size / len(train_images.train_set)))
            print("Epoch {} just ended".format(epoch))

            success_count = 0.0
            for f in test_images.train_set:
                if self.evaluate_img(f[0]) == f[1]:
                    success_count += 1.0
            print("Success ratio: {}%".format(success_count / len(test_images.train_set) * 100))
        train_weights.weights, train_weights.biases = \
            numpy.concatenate([x.flatten() for x in self.nn[4:7]]), numpy.concatenate(self.nn[7:10])

        train_weights.write_out_weights()
