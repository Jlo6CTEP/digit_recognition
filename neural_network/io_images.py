import os

import numpy

# modes
path = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(path, '../imgs')

TRAIN_SET_LABELS = 'train-labels-idx1-ubyte'
TRAIN_SET_IMAGES = 'train-images-idx3-ubyte'
TEST_SET_LABELS = 't10k-labels-idx1-ubyte'
TEST_SET_IMAGES = 't10k-images-idx3-ubyte'

LABEL_MN = 0x00000801
IMAGE_MN = 0x00000803

MY_DATASET = 'my_dataset'

modes = {
    0: (TRAIN_SET_LABELS, TRAIN_SET_IMAGES),
    1: (TEST_SET_LABELS, TEST_SET_IMAGES),
    2: (None, MY_DATASET)
}


class ImageIO:
    mode = None
    images = None
    labels = None
    train_set = None

    def __init__(self, mode):
        self.mode = mode
        self.parse_images()

    def parse_images(self):
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

        (f_name_labels, f_name_images) = modes[self.mode]

        if os.stat(os.path.join(IMAGE_DIR, f_name_images)).st_size == 0:
            return

        # reading images
        f_images = open(os.path.join(IMAGE_DIR, f_name_images), 'rb')
        assert (int.from_bytes(f_images.read(4), byteorder='big') == IMAGE_MN)

        item_number = int.from_bytes(f_images.read(4), byteorder='big')
        x_axis = int.from_bytes(f_images.read(4), byteorder='big')
        y_axis = int.from_bytes(f_images.read(4), byteorder='big')

        images = numpy.empty([item_number, x_axis * y_axis], dtype=numpy.float64)

        for x in range(item_number):
            images[x] = numpy.fromfile(f_images, numpy.uint8, x_axis * y_axis, "") / 255.0

        if f_name_labels is not None:
            f_labels = open(os.path.join(IMAGE_DIR, f_name_labels), 'rb')
            assert (int.from_bytes(f_labels.read(4), byteorder='big') == LABEL_MN)
            item_number = int.from_bytes(f_labels.read(4), byteorder='big')
            labels = numpy.fromfile(f_labels, numpy.uint8, -1, "")
            self.train_set = list(zip(images, labels))
            f_labels.close()
        else:
            labels = None

        self.images, self.labels = images, labels

    def append_image(self, img):
        assert (self.mode == 2)

        size = os.stat(os.path.join(IMAGE_DIR, MY_DATASET)).st_size
        f = open(os.path.join(IMAGE_DIR, MY_DATASET), 'r+b')

        if size != 0:
            f.seek(4)
            item_number = int.from_bytes(f.read(4), byteorder='big')
            f.seek(4)
            f.write(int.to_bytes(item_number + 1, 4, byteorder='big', signed=True))

            f.seek(size)
            f.write(img.flatten().tobytes())
        else:
            f.write(int.to_bytes(IMAGE_MN, 4, byteorder='big'))
            f.write(int.to_bytes(0, 4, byteorder='big'))
            f.write(int.to_bytes(img.shape[0], 4, byteorder='big'))
            f.write(int.to_bytes(img.shape[1], 4, byteorder='big'))
        f.close()
