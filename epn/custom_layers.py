import tensorflow as tf
from tensorflow import keras


# Custom layer to tie the weights of the encoder and decoder
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, dense2=None, activation=None, **kwargs):
        self.dense = dense
        self.dense2 = dense2
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros")
        super().build(batch_input_shape)

    def call(self, inputs):
        weights = self.dense.weights[0]
        if self.dense2:
            weights = tf.concat([self.dense.weights[0], self.dense2.weights[0]], 1)

        z = tf.matmul(inputs, weights, transpose_b=True)
        return self.activation(z + self.biases)
