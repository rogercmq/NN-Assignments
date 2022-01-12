# Source: https://blog.csdn.net/u010874976/article/details/78571788

import numpy as np
import gzip

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def extract_data(filename, num_images):
    """ Extract the images into a 4D tensor [image index, y, x, channels]. Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


def get_data():
    test_data = extract_data('raw_data/t10k-images-idx3-ubyte.gz', 10000)
    test_labels = extract_labels('raw_data/t10k-labels-idx1-ubyte.gz', 10000)
    train_data = extract_data('raw_data/train-images-idx3-ubyte.gz', 60000)
    train_labels = extract_labels('raw_data/train-labels-idx1-ubyte.gz', 60000)
    return list(zip(train_data, train_labels)), list(zip(test_data, test_labels))


if __name__ == '__main__':
    test_data = extract_data('raw_data/t10k-images-idx3-ubyte.gz', 10000)
    test_labels = extract_labels('raw_data/t10k-labels-idx1-ubyte.gz', 10000)
    train_data = extract_data('raw_data/train-images-idx3-ubyte.gz', 60000)
    train_labels = extract_labels('raw_data/train-labels-idx1-ubyte.gz', 60000)
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
