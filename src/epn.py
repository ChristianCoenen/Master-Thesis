from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Reshape, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical, plot_model
from numpy.random import randint, random
from numpy import concatenate, zeros, ones
from src.custom_layers import DenseTranspose
from src import datasets
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import config


class EntropyPropagationNetwork:
    def __init__(self, dataset='mnist', weight_sharing=True, encoder_dims=None, latent_dim=40, classification_dim=10):
        # default None because of side effects with mutable default values
        self.encoder_dims = [100] if encoder_dims is None else encoder_dims
        self.latent_dim = latent_dim
        self.classification_dim = classification_dim

        self.weight_sharing = weight_sharing
        if dataset == 'mnist':
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist()
        elif dataset == 'fashion_mnist':
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist(fashion=True)

        self.input_shape = self.x_train_norm.shape[1:]

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder()

        self.autoencoder.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam",
                                 metrics=["accuracy"])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Combined model
        # TODO: combined model

        self.plot_models() if config.GRAPHVIZ else None

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim,))
        validity = model(encoded_repr)
        return Model(encoded_repr, validity)

    def build_autoencoder(self):
        # Define shared layers
        encoder_layers = []
        for idx, encoder_layer in enumerate(self.encoder_dims):
            encoder_layers.append(Dense(self.encoder_dims[idx], activation="sigmoid", name=f'encoder_{idx}'))
        encoder_to_classification = Dense(self.classification_dim, activation="softmax",
                                          name=f'encoder_{len(self.encoder_dims) }_classification')
        encoder_to_latent_space = Dense(self.latent_dim, activation="sigmoid",
                                        name=f'encoder_{len(self.encoder_dims)}_latent_space')

        # Build encoder model
        encoder_inputs = Input(shape=self.input_shape, name="encoder_input")
        x = Flatten()(encoder_inputs)

        for idx, encoder_layer in enumerate(encoder_layers):
            x = encoder_layers[idx](x)
        classification = encoder_to_classification(x)
        latent_space = encoder_to_latent_space(x)
        # Merge classification neurons and latent space neurons into a single vector via concatenation
        encoded = layers.concatenate([classification, latent_space])
        encoder_model = Model(encoder_inputs, outputs=encoded, name="encoder")

        # Build decoder side
        decoder_inputs = Input(shape=self.latent_dim + self.classification_dim, name="decoder_input")
        if self.weight_sharing:
            x = DenseTranspose(encoder_to_latent_space, encoder_to_classification,
                               activation="sigmoid", name="decoder_0")(decoder_inputs)
            for idx, encoder_layer in enumerate(reversed(encoder_layers)):
                x = DenseTranspose(encoder_layer, activation="sigmoid", name=f"decoder_{1 + idx}")(x)
        else:
            for idx, encoder_layer in enumerate(reversed(encoder_layers)):
                if idx == 0:
                    x = Dense(encoder_layer.output_shape[-1], activation="sigmoid",
                              name=f"decoder_{idx}")(decoder_inputs)
                else:
                    x = Dense(encoder_layer.output_shape[-1], activation="sigmoid", name=f"decoder_{idx}")(x)
            x = Dense(np.prod(self.input_shape), activation="sigmoid", name=f"decoder_{len(self.encoder_dims)}")(x)

        outputs = Reshape(self.input_shape, name="reconstructions")(x)
        decoder_model = Model(decoder_inputs, outputs=outputs, name="decoder")

        # Build autoencoder
        encoded_repr = encoder_model(encoder_inputs)
        reconstructed_img = decoder_model(encoded_repr)
        autoencoder_model = Model(encoder_inputs, outputs=[classification, reconstructed_img], name="autoencoder")
        return encoder_model, decoder_model, autoencoder_model

    def generate_fake_samples(self, n_samples):
        # generate random points in the latent space
        x_latent = random((n_samples, self.latent_dim))
        label = randint(self.classification_dim, size=n_samples)
        x_classification = to_categorical(label, num_classes=self.classification_dim)
        x_input = concatenate((x_latent, x_classification), axis=1)
        # predict outputs
        x = self.decoder.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return x, y, label

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = randint(0, self.x_train_norm.shape[0], n_samples)
        # retrieve selected images
        x = self.x_train_norm[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return x, y

    def train(self, epochs=5):
        self.autoencoder.summary()
        self.autoencoder.fit(self.x_train_norm, [self.y_train, self.x_train_norm], epochs=epochs,
                             validation_data=(self.x_test_norm, (self.y_test, self.x_test_norm)))

    def plot_models(self, path="images"):
        Path(path).mkdir(parents=True, exist_ok=True)
        plot_model(self.discriminator, f"{path}/discriminator_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.autoencoder, f"{path}/autoencoder_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.encoder, f"{path}/encoder_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.decoder, f"{path}/decoder_architecture.png", show_shapes=True, expand_nested=True)

    def show_reconstructions(self, samples, n_samples=10):
        reconstructions = self.autoencoder.predict(samples[:n_samples])
        plt.figure(figsize=(n_samples * 1.5, 3))
        for image_index in range(n_samples):
            plt.subplot(3, n_samples, 1 + image_index)
            plot_image(np.reshape(samples[image_index], (28, 28)))
            plt.subplot(3, n_samples, 1 + n_samples + image_index)
            plot_image(np.reshape(reconstructions[1][image_index], (28, 28)))
            x = plt.subplot(3, n_samples, 1 + n_samples + image_index)
            x.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, image_index))
        plt.show()

    def show_fake_samples(self, n_samples=10):
        samples = self.generate_fake_samples(n_samples)
        plt.figure(figsize=(n_samples * 1.5, 3))
        for image_index in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + image_index)
            plot_image(np.reshape(samples[0][image_index], (28, 28)))
            x = plt.subplot(3, n_samples, 1 + n_samples + image_index)
            x.annotate(str(samples[2][image_index]), xy=(0, image_index))
        plt.show()


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
