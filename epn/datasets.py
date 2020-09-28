from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np


def get_mnist(fashion=False):
    if fashion:
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize training data
    x_train_norm = x_train.astype("float32") / 255
    x_train_norm = np.reshape(x_train_norm, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test_norm = x_test.astype("float32") / 255
    x_test_norm = np.reshape(x_test_norm, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    # Labels -> One Hot encoding
    print("Shape before one-hot encoding: ", y_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    print("Shape after one-hot encoding: ", y_train.shape)
    return (x_train_norm, y_train), (x_test_norm, y_test)


def get_cifar():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Normalize training data
    x_train_norm = x_train.astype("float32") / 255
    x_test_norm = x_test.astype("float32") / 255
    # Labels -> One Hot encoding
    print("Shape before one-hot encoding: ", y_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    print("Shape after one-hot encoding: ", y_train.shape)
    return (x_train_norm, y_train), (x_test_norm, y_test)


def get_maze_memories(path, test_size=0.2, shuffle=False):
    data = np.load(path, allow_pickle=True)
    np.random.shuffle(data) if shuffle else None

    x = np.zeros((len(data), data[0][0].shape[1] + data[0][1].shape[0]), dtype=int)
    y = np.zeros((len(data), data[0][3].shape[1]), dtype=int)
    # There is probably a vectorized way to do it
    for idx, row in enumerate(data):
        x[idx] = np.hstack((data[idx][0].astype(int), data[idx][1].reshape(1, -1)))
        y[idx] = data[idx][3]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)
