import os

import numpy

# modes
RANDOM = 0
TRAINED = 1

# structure
#   random weights and biases will be in RAND_DIR/WEIGHTS_FILE/VECTOR
#   trained weights and biases will be in TRAINED_DIR/WEIGHTS_FILE/VECTOR
#   images must be in IMAGE_DIR

TRAINED_DIR = '../trained_coefficients'
RAND_DIR = '../random_coefficients'
WEIGHTS_FILE = 'vector'

TRAIN_SET_LABELS = 'train-labels-idx1-ubyte'
TRAIN_SET_IMAGES = 'train-images-idx3-ubyte'
TEST_SET_LABELS = 't10k-labels-idx1-ubyte'
TEST_SET_IMAGES = 't10k-images-idx3-ubyte'

LABEL_MN = 0x00000801
IMAGE_MN = 0x00000803

WEIGHTS_MN = 0x00000805


class WeightIO:
    mode = None
    weights = None
    biases = None

    weight_count = None
    bias_count = None

    def __init__(self, mode, n_weights, n_biases):
        self.mode = mode
        self.weight_count = n_weights
        self.bias_count = n_biases

        if self.mode == TRAINED:
            self.read_weights()
        else:
            self.random_weights()

    def write_out_weights(self):
        """
                WEIGHT VECTOR
                FILE(t10k - labels - idx1 - ubyte):
                [offset]   [type]           [value]           [description]
                0000       32 bit integer   0x00000805        magic number(MSB first)
                0004       32 bit integer   ??                number of weights
                0008       32 bit integer   ??                number of biases
                0012       64 bit float     ??                item
                ........
                xxxx       64 bit float     ??                item
        """

        prefix = TRAINED_DIR if self.mode == TRAINED else RAND_DIR

        try:
            f = open(os.path.join(prefix, WEIGHTS_FILE), 'xb')
        except FileExistsError:
            f = open(os.path.join(prefix, WEIGHTS_FILE), 'wb')

        f.write(int.to_bytes(WEIGHTS_MN, 4, byteorder='big'))
        f.write(int.to_bytes(len(self.weights), 4, byteorder='big'))
        f.write(int.to_bytes(len(self.biases), 4, byteorder='big'))

        self.weights.tofile(f, "", "")
        self.biases.tofile(f, "", "")

    def read_weights(self):

        prefix = TRAINED_DIR if self.mode == TRAINED else RAND_DIR

        f = open(os.path.join(prefix, WEIGHTS_FILE), 'rb')

        assert (int.from_bytes(f.read(4), byteorder='big') == WEIGHTS_MN)

        self.weight_count = int.from_bytes(f.read(4), byteorder='big')
        self.bias_count = int.from_bytes(f.read(4), byteorder='big')

        self.weights = numpy.fromfile(f, numpy.float64, self.weight_count)
        self.biases = numpy.fromfile(f, numpy.float64, self.bias_count)

    def random_weights(self):
        self.weights = numpy.random.uniform(-1, 1, self.weight_count)
        self.biases = numpy.random.uniform(-1, 1, self.bias_count)

        return self.write_out_weights()
