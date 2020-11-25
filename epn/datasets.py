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
    x_train, x_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2:4], test_size=test_size, shuffle=shuffle)

    # Reshape data so that it can be used with Keras
    ix_train = range(0, x_train.shape[0])
    train = {
        "state": np.array([*x_train[ix_train, 0]]),
        "action": np.array([*x_train[ix_train, 1]]),
        "next_state": np.array([*y_train[ix_train, 0]]),
        "reward": np.array([*y_train[ix_train, 1]]).reshape(-1, 1),
    }
    ix_test = range(0, x_test.shape[0])
    test = {
        "state": np.array([*x_test[ix_test, 0]]),
        "action": np.array([*x_test[ix_test, 1]]),
        "next_state": np.array([*y_test[ix_test, 0]]),
        "reward": np.array([*y_test[ix_test, 1]]).reshape(-1, 1),
    }
    ix_data = range(0, data.shape[0])
    data = {
        "state": np.array([*data[ix_data, 0]]),
        "action": np.array([*data[ix_data, 1]]),
        "next_state": np.array([*data[ix_data, 2]]),
        "reward": np.array([*data[ix_data, 3]]).reshape(-1, 1),
    }

    return train, test, data
