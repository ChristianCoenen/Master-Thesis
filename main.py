import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

requests.packages.urllib3.disable_warnings()


def get_weights_no_bias(layer):
    # Get all weight matrices from a specific layer, ignore the bias arrays
    weights = []
    for i, weight_var in enumerate(layer.weights):
        weights.append(layer.get_weights()[i]) if 'kernel' in weight_var.name else None
    return weights


def verify_shared_weights(encoder_layer, decoder_layer, classification_layer=None):
    # Get all encoder & decoder weights (no biases, cause they can't be shared anyway)
    encoder_weights = get_weights_no_bias(encoder_layer)
    decoder_weights = get_weights_no_bias(decoder_layer)
    classification_weights = get_weights_no_bias(classification_layer) if classification_layer else None

    # Verify that the layers only have one weight matrix, otherwise shared weights can't be verified
    if classification_layer:
        assert len(encoder_weights) == 1 and len(classification_weights) == 1 and len(decoder_weights) == 2
    else:
        assert len(encoder_weights) == 1 and len(decoder_weights) == 1

    # Verify that weights are the same
    if classification_layer:
        if decoder_weights[0].shape == encoder_weights[0].shape:
            assert_array_equal(decoder_weights[0], encoder_weights[0])
            assert_array_equal(decoder_weights[1], classification_weights[0])
        else:
            assert_array_equal(decoder_weights[1], encoder_weights[0])
            assert_array_equal(decoder_weights[0], classification_weights[0])
    else:
        assert_array_equal(decoder_weights[0], encoder_weights[0])


# Config GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(physical_devices)}")
tf.config.experimental.set_memory_growth(physical_devices[0], True) if physical_devices else None

# Get data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_norm = x_train.astype("float32") / 255
x_train_norm = np.reshape(x_train_norm, (60000, 28, 28, 1))
x_test_norm = x_test.astype("float32") / 255
x_test_norm = np.reshape(x_test_norm, (10000, 28, 28, 1))

# Labels -> One Hot encoding
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", y_train.shape)


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def show_reconstructions(model, images=x_test_norm, n_images=10):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(3, n_images, 1 + image_index)
        plot_image(np.reshape(images[image_index], (28, 28)))
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plot_image(np.reshape(reconstructions[1][image_index], (28, 28)))
        x = plt.subplot(3, n_images, 1 + n_images + image_index)
        x.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, image_index))


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


input_shape = (28, 28, 1)
encoder_1_neurons = 100
latent_space_neurons = 40
classification_neurons = 10
decoder_1_neurons = encoder_1_neurons
# Model: Stacked Autoencoder with tied (shared) weights.

# Define encoder side
inputs = keras.Input(shape=input_shape)
flatten = layers.Flatten()
input_layer_to_encoder_1 = layers.Dense(encoder_1_neurons, activation="sigmoid", name='encoder')
encoder_1_to_classification = layers.Dense(classification_neurons, activation="softmax", name='classification')
encoder_1_to_latent_space = layers.Dense(latent_space_neurons, activation="sigmoid", name='latent_space')

# Define decoder side
latent_classification_to_decoder_1 = DenseTranspose(encoder_1_to_latent_space, encoder_1_to_classification,
                                                    activation="sigmoid", name="decoder")
decoder_1_to_output = DenseTranspose(input_layer_to_encoder_1, activation="sigmoid", name="outputs")

# Connect network
x = flatten(inputs)
x = input_layer_to_encoder_1(x)
classification = encoder_1_to_classification(x)
latent_space = encoder_1_to_latent_space(x)
# Merge classification neurons and latent space neurons into a single vector via concatenation
x = layers.concatenate([classification, latent_space])
x = latent_classification_to_decoder_1(x)
x = decoder_1_to_output(x)
outputs = layers.Reshape(input_shape, name='reconstructions')(x)

# Build & Show the model
tied_ae_model = keras.Model(inputs=inputs, outputs=[classification, outputs], name="mnist")
tied_ae_model.summary()
keras.utils.plot_model(tied_ae_model, "model_architecture.png", show_shapes=True)

# Compile & Train
tied_ae_model.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam", metrics=["accuracy"])
history = tied_ae_model.fit(x_train_norm, [y_train, x_train_norm], epochs=5,
                            validation_split=[(x_test_norm, y_test), (x_test_norm, x_test_norm)])

# verify that encoder & decoder weights are the same
verify_shared_weights(input_layer_to_encoder_1, decoder_1_to_output)
verify_shared_weights(encoder_1_to_latent_space, latent_classification_to_decoder_1, encoder_1_to_classification)

# Show 10 inputs and their outputs
show_reconstructions(tied_ae_model)
plt.show()
