import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
import matplotlib.pyplot as plt
import numpy as np
requests.packages.urllib3.disable_warnings()

# Get data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_norm = x_train.astype("float32") / 255
x_train_norm = np.reshape(x_train_norm, (60000,28,28,1))
x_test_norm = x_test.astype("float32") / 255
x_test_norm = np.reshape(x_test_norm, (10000,28,28,1))

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
        plot_image(np.reshape(images[image_index], (28,28)))
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plot_image(np.reshape(reconstructions[1][image_index], (28,28)))
        x = plt.subplot(3, n_images, 1 + n_images + image_index)
        x.annotate(str(np.argmax(reconstructions[0][image_index])),xy=(0,image_index))

# Custom layer to tie the weights of the encoder and decoder
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


latent_neurons = 940
classify_neurons = 10
input_shape = (28, 28, 1)
# Model: Stacked Autoencoder with tied (shared) weights.
# Encoder
inputs = keras.Input(shape=input_shape)
x = layers.Flatten()(inputs)
dense = layers.Dense(latent_neurons + classify_neurons, activation="sigmoid", name='encoder')
x = dense(x)

# Latent space: contains classification neurons and latent space neurons
classification_neurons = layers.Dense(classify_neurons, activation="softmax", name='classification')(x)
latent_space_neurons = layers.Dense(latent_neurons, activation="sigmoid", name='latent_space')(x)
# Merge classification neurons and latent space neurons into a single vector via concatenation
x = layers.concatenate([classification_neurons, latent_space_neurons])

# Decoder
x = DenseTranspose(dense, activation="sigmoid", name="decoder")(x)
outputs = layers.Reshape(input_shape, name='reconstructions')(x)

# Build & Show the model
tied_ae_model = keras.Model(inputs=inputs, outputs=[classification_neurons, outputs], name="mnist")
tied_ae_model.summary()
keras.utils.plot_model(tied_ae_model, "model_architecture.png", show_shapes=True)

# Compile & Train
tied_ae_model.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam", metrics=["accuracy"])
history = tied_ae_model.fit(x_train_norm, [y_train, x_train_norm], epochs=2, 
    validation_data=[(x_test_norm, y_test), (x_test_norm, x_test_norm)])

# Show 10 inputs and their outputs
show_reconstructions(tied_ae_model)
plt.show()