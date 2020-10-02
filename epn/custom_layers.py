import tensorflow as tf
from tensorflow import keras
from typing import List


# Custom layer to tie the weights of the encoder and decoder
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense_layers: List, activation=None, **kwargs):
        self.dense_layers = dense_layers
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        # Theoretically it can also be dense_layers[1] or [2] because they all have the same input shape
        self.biases = self.add_weight(name="bias", shape=[self.dense_layers[0].input_shape[-1]], initializer="zeros")
        super().build(batch_input_shape)

    def call(self, inputs):
        weights = tf.concat([layer.weights[0] for layer in self.dense_layers], axis=1)

        z = tf.matmul(inputs, weights, transpose_b=True)
        return self.activation(z + self.biases)
