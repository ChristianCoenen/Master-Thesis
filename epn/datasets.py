from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import randint


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
    x_train, x_test, y_train, y_test = train_test_split(data[:, :3], data[:, 3:], test_size=test_size, shuffle=shuffle)
    return (x_train, y_train), (x_test, y_test)


def get_car_racing_memories(path, test_size=0.2, shuffle=False):
    data = np.load(path, allow_pickle=True)
    np.random.shuffle(data) if shuffle else None
    x_train, x_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2:], test_size=test_size, shuffle=shuffle)

    # Reshape data so that it can be used with Keras
    ix = range(0, x_train.shape[0])
    x_train = [np.array([*x_train[ix, i]]) for i in range(2)]
    y_train = [np.array([*y_train[ix, i]]) for i in range(2)]
    ix = range(0, x_test.shape[0])
    x_test = [np.array([*x_test[ix, i]]) for i in range(2)]
    y_test = [np.array([*y_test[ix, i]]) for i in range(2)]
    y_train[0] = y_train[0].reshape(-1, 1)
    y_test[0] = y_test[0].reshape(-1, 1)
    # This is needed as long as encoder output is not reshaped in the network
    y_train[1] = y_train[1].reshape(y_train[1].shape[0], np.prod(np.array(y_train[1].shape[1:])))
    y_test[1] = y_test[1].reshape(y_test[1].shape[0], np.prod(np.array(y_test[1].shape[1:])))

    return (x_train, y_train), (x_test, y_test)
