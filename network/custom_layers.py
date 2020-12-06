import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from typing import List, Optional, Union


# Custom layer to tie the weights of the encoder and decoder
class DenseTranspose(Layer):
    """

    """
    def __init__(
        self,
        dense_layers: List[Layer],
        activation: Union[str, Layer, None] = None,
        custom_weights: Optional[tf.Variable] = None,
        **kwargs,
    ):
        """
        :param  dense_layers:
            1. One dense layer whose weight references are used as for the DenseTranspose layer
            2. Multiple dense layer whose weights are concatenated and used for the DenseTranspose layer
               (the DenseTranspose layer has the dimension of the concatenated matrix)
        :param activation:
            The activation function that is used for the DenseTranspose layer
        :param custom_weights:
            A tensorflow matrix variable that holds the weights whose references are used for the DenseTranspose layer
            (use case: if a subset of weights in an DenseLayer should be used for the DenseTranspose layer)
        """

        self.dense_layers = dense_layers
        self.custom_weights = custom_weights
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        """ Automatically called by TensorFlow during training. """
        if self.custom_weights is None:
            # Theoretically it can also be dense_layers[1] or [2] because they all have the same input shape
            self.biases = self.add_weight(
                name="bias", shape=[self.dense_layers[0].input_shape[-1]], initializer="zeros"
            )
        else:
            self.biases = self.add_weight(name="bias", shape=[self.custom_weights.shape[0]], initializer="zeros")

        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        """ Automatically called by TensorFlow during training. """
        if self.custom_weights is None:
            weights = tf.concat([layer.weights[0] for layer in self.dense_layers], axis=1)
        else:
            weights = self.custom_weights

        z = tf.matmul(inputs, weights, transpose_b=True)
        return self.activation(z + self.biases)
