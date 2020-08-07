from tensorflow import keras
from tensorflow.keras import layers
from src.dense_transpose import DenseTranspose
from numpy.random import randint
from numpy import zeros
import matplotlib.pyplot as plt
import numpy as np
import config


class Generator:
    """
    Autoencoder with shared weights
    """
    def __init__(self):
        # Variables that will be initialized later
        self.classification = None
        self.latent_space = None
        self.outputs = None
        self.autoencoder = None
        self.generator = None

        # Define encoder side
        self.inputs = keras.Input(shape=config.INPUT_SHAPE)
        self.flatten = layers.Flatten()
        self.input_layer_to_encoder_1 = layers.Dense(config.ENCODER_1_NEURONS, activation="sigmoid", name='encoder')
        self.encoder_1_to_classification = layers.Dense(config.CLASSIFICATION_NEURONS, activation="softmax",
                                                        name='classification')
        self.encoder_1_to_latent_space = layers.Dense(config.LATENT_SPACE_NEURONS, activation="sigmoid",
                                                      name='latent_space')

        # Define decoder side
        self.generator_inputs = keras.Input(shape=config.LATENT_SPACE_NEURONS + config.CLASSIFICATION_NEURONS)
        self.latent_classification_to_decoder_1 = DenseTranspose(self.encoder_1_to_latent_space,
                                                                 self.encoder_1_to_classification,
                                                                 activation="sigmoid", name="decoder")
        self.decoder_1_to_output = DenseTranspose(self.input_layer_to_encoder_1, activation="sigmoid", name="outputs")

    def build_autoencoder(self):
        # Connect network
        x = self.flatten(self.inputs)
        x = self.input_layer_to_encoder_1(x)
        classification = self.encoder_1_to_classification(x)
        latent_space = self.encoder_1_to_latent_space(x)
        # Merge classification neurons and latent space neurons into a single vector via concatenation
        x = layers.concatenate([classification, latent_space])
        x = self.latent_classification_to_decoder_1(x)
        x = self.decoder_1_to_output(x)
        outputs = layers.Reshape(config.INPUT_SHAPE, name='reconstructions')(x)

        # Build the connected network
        self.autoencoder = keras.Model(inputs=self.inputs, outputs=[classification, outputs], name="autoencoder")
        return self.autoencoder

    def build_generator(self):
        x = self.generator_inputs
        x = self.latent_classification_to_decoder_1(x)
        x = self.decoder_1_to_output(x)
        outputs = layers.Reshape(config.INPUT_SHAPE, name='reconstructions')(x)

        # Build the connected network
        self.generator = keras.Model(inputs=self.generator_inputs, outputs=outputs, name="generator")
        return self.generator

    # generate points in latent space as input for the generator
    def generate_latent_and_classification_points(self, n_samples=10):
        # generate points in the latent space
        x_latent = np.random.random((n_samples, config.LATENT_SPACE_NEURONS))
        x_classification = randint(config.CLASSIFICATION_NEURONS, size=n_samples)
        x_classification = keras.utils.to_categorical(x_classification, num_classes=config.CLASSIFICATION_NEURONS)
        return np.concatenate((x_latent, x_classification), axis=1)

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples=10):
        # generate points in latent space
        x_input = self.generate_latent_and_classification_points(n_samples)
        # predict outputs
        X = self.generator.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return X, y

    def show_reconstructions(self, images, n_images=10):
        reconstructions = self.autoencoder.predict(images[:n_images])
        fig = plt.figure(figsize=(n_images * 1.5, 3))
        for image_index in range(n_images):
            plt.subplot(3, n_images, 1 + image_index)
            plot_image(np.reshape(images[image_index], (28, 28)))
            plt.subplot(3, n_images, 1 + n_images + image_index)
            plot_image(np.reshape(reconstructions[1][image_index], (28, 28)))
            x = plt.subplot(3, n_images, 1 + n_images + image_index)
            x.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, image_index))


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
