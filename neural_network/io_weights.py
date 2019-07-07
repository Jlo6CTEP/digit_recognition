import os

import numpy

# modes
RANDOM = 0
TRAINED = 1

# structure
#   random weights and biases will be in RAND_DIR/WEIGHTS_FILE/VECTOR
#   trained weights and biases will be in TRAINED_DIR/WEIGHTS_FILE/VECTOR
#   images must be in IMAGE_DIR

# set up couple of paths and constant
path = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(path, '../trained_coefficients')
RAND_DIR = os.path.join(path, '../random_coefficients')
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
        """
        If mode is set to TRAINED it will load weights from TRAINED_DIR, otherwise generate them randomly
        :param mode:
        :param n_weights:
        :param n_biases:
        """
        self.mode = mode
        self.weight_count = n_weights
        self.bias_count = n_biases

        if self.mode == TRAINED:
            self.read_weights()
        else:
            self.random_weights()

    def write_out_weights(self):
        """
        Structure of the output file
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

        # open file in correct directory
        prefix = TRAINED_DIR if self.mode == TRAINED else RAND_DIR

        # if such a file a exists - truncate its content
        try:
            f = open(os.path.join(prefix, WEIGHTS_FILE), 'xb')
        except FileExistsError:
            f = open(os.path.join(prefix, WEIGHTS_FILE), 'wb')

        # create the header
        f.write(int.to_bytes(WEIGHTS_MN, 4, byteorder='big'))
        f.write(int.to_bytes(len(self.weights), 4, byteorder='big'))
        f.write(int.to_bytes(len(self.biases), 4, byteorder='big'))

        # write the main content
        self.weights.tofile(f, "", "")
        self.biases.tofile(f, "", "")

    def read_weights(self):
        """
        Reads weights/biases from file and updates the internal state
        :return:
        """

        # open file in the right place and check if it is corrupted
        prefix = TRAINED_DIR if self.mode == TRAINED else RAND_DIR
        f = open(os.path.join(prefix, WEIGHTS_FILE), 'rb')
        assert (int.from_bytes(f.read(4), byteorder='big') == WEIGHTS_MN)

        # first read the number of weights and biases
        self.weight_count = int.from_bytes(f.read(4), byteorder='big')
        self.bias_count = int.from_bytes(f.read(4), byteorder='big')

        # then read the content itself
        self.weights = numpy.fromfile(f, numpy.float64, self.weight_count)
        self.biases = numpy.fromfile(f, numpy.float64, self.bias_count)

    def random_weights(self):
        """
        Creates random weights and writes them into RAND_DIR
        :return:
        """
        self.weights = numpy.random.uniform(-1, 1, self.weight_count)
        self.biases = numpy.random.uniform(-1, 1, self.bias_count)

        return self.write_out_weights()
