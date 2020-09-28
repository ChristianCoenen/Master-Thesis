from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Reshape, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from numpy.random import randint
from numpy import concatenate, zeros, ones
from epn.custom_layers import DenseTranspose
from epn import datasets
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math


class EntropyPropagationNetwork:
    """
    A class implementing the Entropy Propagation Network architecture consisting of:

    1. Encoder and Decoder Model (Decoder will mirror the encoder layers) (and weights if weight sharing is activated)
    2. Autoencoder model which is constructed by combining the Encoder model with the Decoder model
    3. Discriminator model
    4. GAN model which is constructed by combining the Decoder with the Discriminator model

    Note that the Decoder can also be referenced as Generator when generating fake samples
    """

    def __init__(
        self,
        dataset="mnist",
        dataset_path=None,
        weight_sharing=True,
        encoder_dims=None,
        latent_dim=40,
        classification_dim=10,
        graphviz_installed=False,
    ):
        """
        :param dataset: str
            Selects the underlying dataset.
            Valid values: ['mnist', 'fashion_mnist']
        :param weight_sharing: bool
            If set to true, the decoder will used the weights created on the encoder side using DenseTranspose layers
        :param encoder_dims: [int]
            Each value (x) represents one layer with x neurons.
        :param latent_dim:
            Number of latent space neurons (bottleneck layer in the Autoencoder)
        :param classification_dim:
            Output neurons to classify the inputs based on their label
            Needs to be of same dim as the number of output labels.
            Is not automatically generated based on dataset, because it might be variable for Reinforcement Learning
        """

        self.weight_sharing = weight_sharing
        # default None because of side effects with mutable default values
        self.encoder_dims = [100] if encoder_dims is None else encoder_dims
        self.latent_dim = latent_dim
        self.classification_dim = classification_dim
        self.dataset = dataset

        if dataset == "mnist":
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist()
        elif dataset == "fashion_mnist":
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist(fashion=True)
        elif dataset == "cifar10":
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_cifar()
        elif dataset == "maze_memories":
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_maze_memories(
                dataset_path
            )
        else:
            raise ValueError("Unknown dataset!")

        self.input_shape = self.x_train_norm.shape[1:]

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=["binary_crossentropy", "mean_squared_error"], optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
        )

        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder()

        self.autoencoder.compile(
            loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam", metrics=["accuracy"]
        )

        # GAN model (decoder & discriminator) - For the GAN model we will only train the generator
        self.discriminator.trainable = False
        self.gan = self.build_gan()
        self.gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

        # only set graphviz_installed to true if you have it installed (see README)
        self.save_model_architecture_images() if graphviz_installed else None
        self.is_pretraining = True

    def build_discriminator(self):
        """Creates a discriminator model.

        Leaky ReLU is recommended for Discriminator networks.
        'Within the discriminator we found the leaky rectified activation to work well ...'
            — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015.

        :return: Discriminator model
        """
        inputs = Input(shape=self.input_shape, name="discriminator_inputs")
        x = Flatten()(inputs)
        x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(0.3)(x)
        real_or_fake = Dense(1, activation="sigmoid", name="real_or_fake")(x)
        classification = Dense(self.classification_dim, activation="softmax", name="classification")(x)
        return Model(inputs, outputs=[real_or_fake, classification], name="discriminator")

    def build_autoencoder(self):
        """Creates an encoder, decoder and autoencoder model.

        :return: Encoder model, Decoder mode, Autoencoder model
        """
        # Define shared layers
        encoder_layers = []
        for idx, encoder_layer in enumerate(self.encoder_dims):
            encoder_layers.append(Dense(self.encoder_dims[idx], activation=LeakyReLU(alpha=0.2), name=f"encoder_{idx}"))
        encoder_to_classification = Dense(
            self.classification_dim, activation="softmax", name=f"encoder_{len(self.encoder_dims) }_classification"
        )
        encoder_to_latent_space = Dense(
            self.latent_dim, activation=LeakyReLU(alpha=0.2), name=f"encoder_{len(self.encoder_dims)}_latent_space"
        )

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
            x = DenseTranspose(
                encoder_to_latent_space, encoder_to_classification, activation=LeakyReLU(alpha=0.2), name="decoder_0"
            )(decoder_inputs)
            for idx, encoder_layer in enumerate(reversed(encoder_layers)):
                if idx == len(encoder_layers) - 1:
                    x = DenseTranspose(encoder_layer, activation="sigmoid", name=f"decoder_{1 + idx}")(x)
                else:
                    x = DenseTranspose(encoder_layer, activation=LeakyReLU(alpha=0.2), name=f"decoder_{1 + idx}")(x)
        else:
            for idx, encoder_layer in enumerate(reversed(encoder_layers)):
                if idx == 0:
                    x = Dense(encoder_layer.output_shape[-1], activation="sigmoid", name=f"decoder_{idx}")(
                        decoder_inputs
                    )
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

    # define the combined generator and discriminator model, for updating the generator
    def build_gan(self):
        """Defines the combined decoder and discriminator model, for updating the decoder

        :return:
        """
        # connect them
        inputs = Input(shape=self.latent_dim + self.classification_dim, name="gan_inputs")
        decoded = self.decoder(inputs)
        discriminated = self.discriminator(decoded)
        return Model(inputs, discriminated)

    def generate_latent_and_classification_points(self, n_samples):
        # generate random points in the latent space
        x_latent = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        labels = randint(self.classification_dim, size=n_samples)
        labels = to_categorical(labels, num_classes=self.classification_dim)
        x_input = concatenate((x_latent, labels), axis=1)
        return x_input, labels

    def generate_fake_samples(self, n_samples):
        """Generates fake samples for

        :param n_samples: int
            Number of samples that needs to be generated.
        :return:
            x: Random samples generator by the decoder (generator)
            y: Array with length of n_samples containing zeros (to indicate that those are fake samples)
            z: Array with length of n_samples showing the objective for the decoder for each samples
               Helps debugging whether the generated samples match the classification input (e.g. generate a 6)
        """
        # generate random points in the latent space
        x_inputs, labels = self.generate_latent_and_classification_points(n_samples)
        # predict outputs
        x = self.decoder.predict(x_inputs)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return x, y, labels

    def generate_real_samples(self, n_samples):
        """This method samples from the training data set to show real samples to the discriminator.

        :param n_samples: int
            Number of samples that needs to be extracted.
        :return:
            x: Random samples generator by the decoder (generator)
            y: Array with length of n_samples containing zeros (to indicate that those are fake samples)
        """
        # choose random instances
        ix = randint(0, self.x_train_norm.shape[0], n_samples)
        # retrieve selected images
        x = self.x_train_norm[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        labels = self.y_train[ix]
        return x, y, labels

    def train_autoencoder(self, epochs=5, batch_size=32, validation_split=0.1):
        self.autoencoder.fit(
            self.x_train_norm,
            [self.y_train, self.x_train_norm],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
        )
        if self.dataset != "maze_memories":
            self.save_reconstruction_plot_images(self.x_train_norm[10:20])
            self.save_fake_sample_plot_images()

    def train(self, epochs=5, batch_size=32, pre_train_epochs=3, train_encoder=True):
        batch_per_epoch = int(60000 / batch_size)
        half_batch = int(batch_size / 2)

        if pre_train_epochs:
            self.train_autoencoder(pre_train_epochs)
        self.is_pretraining = False

        # manually enumerate epochs
        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(batch_per_epoch):
                """ Discriminator training """
                # create training set for the discriminator
                x_real, y_real, labels_real = self.generate_real_samples(n_samples=half_batch)
                x_fake, y_fake, labels_fake = self.generate_fake_samples(n_samples=half_batch)
                x_discriminator, y_discriminator = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
                labels = np.vstack((labels_real, labels_fake))
                # One-sided label smoothing
                y_discriminator[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _, _, _, _ = self.discriminator.train_on_batch(x_discriminator, [y_discriminator, labels])

                """ Generator training (discriminator weights deactivated!) """
                # prepare points in latent space as input for the generator
                x_gan, labels = self.generate_latent_and_classification_points(batch_size)
                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1 we have an error of 0 and vice versa
                y_gan = ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss, _, _ = self.gan.train_on_batch(x_gan, [y_gan, labels])

                """ Autoencoder training """
                # this might result in the discriminator outperforming the generator depending on architecture
                self.autoencoder.train_on_batch(x_real, [labels_real, x_real]) if train_encoder else None

                # summarize loss on this batch
                print(
                    f">{i + 1}, {j + 1:0{len(str(batch_per_epoch))}d}/{batch_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}"
                )

            # evaluate the model performance each epoch
            self.summarize_performance(i)
        self.save_reconstruction_plot_images(self.x_train_norm[10:20])

    def summarize_performance(self, epoch, n_samples=100):
        """Evaluate the discriminator, plot generated images, save generator model

        :param epoch: int
            Current training epoch.
        :param n_samples: int
            Number of samples that are plotted.
        :return:
            None
        """
        # prepare real samples
        x_real, y_real, _ = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, _, _, acc_real, _ = self.discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake, labels = self.generate_fake_samples(n_samples)
        # evaluate discriminator on fake examples
        _, _, _, acc_fake, _ = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print(f">Accuracy real: {acc_real * 100:.0f}%%, fake: {acc_fake * 100:.0f}%%")
        # save plot
        self.save_fake_sample_plot_images(x_fake=x_fake, labels=labels, epoch=epoch)

    def evaluate(self):
        # Evaluates the autoencoder based on the test data
        return self.autoencoder.evaluate(self.x_test_norm, [self.y_test, self.x_test_norm], verbose=0)

    def save_model_architecture_images(self, path="images/architecture"):
        """Saves all EPN model architectures as PNGs into a defined sub folder.

        :param path: str
            Relative path from the root directory
        :return:
            None
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        plot_model(self.discriminator, f"{path}/discriminator_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.autoencoder, f"{path}/autoencoder_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.encoder, f"{path}/encoder_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.decoder, f"{path}/decoder_architecture.png", show_shapes=True, expand_nested=True)
        plot_model(self.gan, f"{path}/gan_architecture.png", show_shapes=True, expand_nested=True)

    def save_reconstruction_plot_images(self, samples, path="images/plots"):
        """Pushes x samples through the autoencoder to generate & visualize reconstructions

        :param samples:
            Samples that matches the following shape [n_samples, autoencoder input shape]
        :param path: str
            Path to the directory where the plots are getting stored.
        :return:
            None
        """
        n_samples = samples.shape[0]
        reconstructions = self.autoencoder.predict(samples)
        plt.figure(figsize=(n_samples * 1.5, 3))
        for image_index in range(n_samples):
            # orig image
            _ = add_subplot(image=samples[image_index, :, :, 0], n_cols=3, n_rows=n_samples, index=1 + image_index)
            # reconstruction
            plot_obj = add_subplot(
                image=reconstructions[1][image_index, :, :, 0],
                n_cols=3,
                n_rows=n_samples,
                index=1 + n_samples + image_index,
            )
            # label
            plot_obj.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, 0))

        filename = "pre_reconstructed_plot.png" if self.is_pretraining else "post_reconstructed_plot.png"
        save_plot_as_image(path=path, filename=filename)

    def save_fake_sample_plot_images(self, x_fake=None, labels=None, epoch=-1, n_samples=100, path="images/plots"):
        """Create and save a plot of generated images (reversed grayscale)

            Useful to show if the generator is able to generate real looking images from random points.

        :param x_fake:
        :param labels:
        :param epoch:
        :param n_samples: int
            Number of samples that should be generated and plotted.
        :param path: str
            Path to the directory where the plots are getting stored.
        :return:
        """
        n_columns = math.ceil(math.sqrt(n_samples))
        n_rows = math.ceil(n_samples / n_columns)
        if x_fake is None or labels is None:
            x_fake, _, labels = self.generate_fake_samples(n_samples)

        labels_numerical = tf.argmax(labels, axis=1).numpy()
        plt.figure(figsize=(n_columns, n_rows))
        for i in range(n_samples):
            plot_obj = add_subplot(image=x_fake[i, :, :, 0], n_cols=n_columns, n_rows=n_rows, index=1 + i)
            plot_obj.annotate(str(labels_numerical[i]), xy=(0, 0))

        save_plot_as_image(path=path, filename=f"generated_plot_e{epoch + 1:03d}.png")


def add_subplot(image, n_cols, n_rows, index):
    plot_obj = plt.subplot(n_cols, n_rows, index)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    return plot_obj


def save_plot_as_image(path, filename):
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = f"{path}/{filename}"
    plt.savefig(full_path)
    plt.close()
