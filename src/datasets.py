from tensorflow import keras
import numpy as np


def get_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize training data
    x_train_norm = x_train.astype("float32") / 255
    x_train_norm = np.reshape(x_train_norm, (60000, 28, 28, 1))
    x_test_norm = x_test.astype("float32") / 255
    x_test_norm = np.reshape(x_test_norm, (10000, 28, 28, 1))
    # Labels -> One Hot encoding
    print("Shape before one-hot encoding: ", y_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    print("Shape after one-hot encoding: ", y_train.shape)
    return (x_train_norm, y_train), (x_test_norm, y_test)
