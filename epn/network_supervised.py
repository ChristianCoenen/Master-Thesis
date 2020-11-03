from typing import List, Tuple, Optional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from numpy.random import randint
from numpy import zeros, ones
from epn.helper import add_subplot, save_plot_as_image
from epn.network import EPNetwork
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math


class EPNetworkSupervised(EPNetwork):
    """
    A class implementing the Entropy Propagation Network architecture consisting of:

    1. Encoder and Decoder Model (Decoder will mirror the encoder layers) (and weights if weight sharing is activated)
    2. Autoencoder model which is constructed by combining the Encoder model with the Decoder model
    3. Discriminator model
    4. GAN model which is constructed by combining the Decoder with the Discriminator model

    Note that the Decoder can also be referenced as Generator when generating fake samples.
    """

    def __init__(
        self,
        data: List[Tuple[List[np.ndarray], Tuple[List[np.ndarray]]]],
        latent_dim: int,
        autoencoder_loss: List[str],
        weight_sharing: bool,
        encoder_dims: List[int],
        discriminator_dims: List[int],
        seed: int,
    ):
        """
        :param data:
            Data that is trained on. Has to be a 4-tuple consisting of numpy arrays with shape (n_samples x data_shape)
        :param latent_dim:
            Number of latent space neurons (bottleneck layer in the Autoencoder)
        :param autoencoder_loss:
            This parameter allows to define the classification and reconstruction loss functions for the autoencoder.
        """
        super().__init__(weight_sharing, encoder_dims, discriminator_dims, seed)
        (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = data
        self.latent_dim = latent_dim
        self.autoencoder_loss = autoencoder_loss
        self.classification_dim = len(self.y_train[1])

        # Build Autoencoder
        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder(
            encoder_input_tensors=[
                Input(shape=self.x_train_norm.shape[1:], name="encoder_input"),
            ],
            encoder_output_layers=[
                Dense(self.latent_dim, activation=LeakyReLU(alpha=0.2), name="latent_space"),
                Dense(self.classification_dim, activation="softmax", name="classifier"),
            ],
            ae_ignored_output_layer_names=["latent_space"],
        )
        self.autoencoder.compile(loss=self.autoencoder_loss, optimizer="adam", metrics=["accuracy"])

        # Build Discriminator
        self.discriminator = self.build_discriminator(
            input_tensors=[
                Input(shape=self.decoder.output_shape[1:], name="reconstructions"),
            ],
            output_layers=[
                Dense(1, activation="sigmoid", name="real_or_fake"),
            ],
        )
        self.discriminator.compile(
            loss=["binary_crossentropy"],
            optimizer=Adam(lr=0.0002, beta_1=0.5),
            metrics=["accuracy"],
        )

        # Build GAN model (decoder & discriminator) - For the GAN model we will only train the generator
        self.discriminator.trainable = False
        self.gan = self.build_gan(self.decoder, self.discriminator)
        self.gan.compile(loss=["binary_crossentropy"], optimizer=Adam(lr=0.0002, beta_1=0.5))

    def generate_latent_and_classification_points(self, n_samples):
        # generate random points in the latent space
        x_latent = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        random_samples = np.random.rand(n_samples, *self.encoder.input_shape[1:])
        x_latent = self.encoder.predict(random_samples)[0]
        x_classification = randint(self.classification_dim, size=n_samples)
        x_classification = to_categorical(x_classification, num_classes=self.classification_dim)
        return x_latent, x_classification

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
        x_latent, x_classification = self.generate_latent_and_classification_points(n_samples)
        # predict outputs
        reconstruction = self.decoder.predict([x_latent, x_classification])
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return reconstruction, y, x_classification

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

    def train_autoencoder(self, **kwargs):
        self.autoencoder.fit(self.x_train_norm, [self.y_train, self.x_train_norm], **kwargs)

    def train(self, epochs: int, batch_size: int, steps_per_epoch: int, train_encoder: bool):
        half_batch = int(batch_size / 2)

        # manually enumerate epochs
        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(steps_per_epoch):
                """ Discriminator training """
                # create training set for the discriminator
                # TODO: labels_real one hot size 10 and y_real is 0 / 1. I think naming is not optimal (got confused)
                x_real, y_real, labels_real = self.generate_real_samples(n_samples=half_batch)
                x_fake, y_fake, labels_fake = self.generate_fake_samples(n_samples=half_batch)
                x_discriminator, y_discriminator = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
                labels = np.vstack((labels_real, labels_fake))
                # One-sided label smoothing
                y_discriminator[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _ = self.discriminator.train_on_batch(x_discriminator, y_discriminator)

                """ Generator training (discriminator weights deactivated!) """
                # prepare points in latent space as input for the generator
                x_latent, x_classification = self.generate_latent_and_classification_points(batch_size)
                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1, we have an error of 0 and vice versa
                y_gan = ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch([x_latent, x_classification], y_gan)

                """ Autoencoder training """
                # this might result in the discriminator outperforming the generator depending on architecture
                self.autoencoder.train_on_batch(x_real, [labels_real, x_real]) if train_encoder else None

                # summarize loss on this batch
                print(
                    f">{i + 1}, {j + 1:0{len(str(steps_per_epoch))}d}/{steps_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}"
                )

            # evaluate the model performance each epoch
            self.summarize_performance(i)
        self.save_reconstruction_plot_images(self.x_test_norm[10:20], state="post_gan_training")

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
        x_real, y_real, labels = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake, labels = self.generate_fake_samples(n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print(f">Accuracy real: {acc_real * 100:.0f}%%, fake: {acc_fake * 100:.0f}%%")
        # save plot
        self.save_fake_sample_plot_images(x_fake=x_fake, labels=labels, epoch=epoch)

    def evaluate(self):
        # Evaluates the autoencoder based on the test data (only returns classification accuracy, cause decoder accuracy
        # is not valuable (interpreting result as visualized image is better)
        acc = self.autoencoder.evaluate(self.x_test_norm, [self.y_test, self.x_test_norm], verbose=0)[3]
        return acc * 100

    def save_model_architecture_images(
        self, models: Optional[List[Model]] = None, path: str = "images/epn_supervised/architecture", fmt: str = "png"
    ):
        models = models if models is not None else []
        models.extend(
            [
                self.encoder,
                self.decoder,
                self.autoencoder,
                self.discriminator,
                self.gan,
            ]
        )
        super().save_model_architecture_images(models, path, fmt)

    def visualize_autoencoder_predictions_to_file(self, state, acc=None):
        self.save_reconstruction_plot_images(self.x_test_norm[20:30], state, acc=acc)
        self.save_fake_sample_plot_images()

    def create_modified_classification_plot(self, sample_idx=None, path="images/epn_supervised/plots", random=False):
        """Creates reconstructions for one sample with all possible labels

        :param sample_idx:
            Index that is used to extract a sample from the test data
        :param path: str
            Path to the directory where the plots are getting stored.
        :return:
            None
        """
        n_rows = self.classification_dim
        n_cols = 2
        if sample_idx:
            sample = self.x_test_norm[sample_idx]
        elif random:
            sample = np.random.rand(*self.x_test_norm[0].shape)
        else:
            raise ValueError("Either provide a sample index or set random to true!")

        sample = sample[np.newaxis, ...]

        sample_latent_space = self.encoder.predict(sample)[0]

        decoder_input = [np.array([sample_latent_space.flatten()] * n_rows), np.eye(n_rows)]
        reconstructions = self.decoder.predict(decoder_input)
        plt.figure(figsize=(n_rows * 1.5, n_cols))
        for image_index in range(n_rows):
            # orig image
            _ = add_subplot(sample[0, :, :, 0], n_cols=n_cols, n_rows=n_rows, index=1 + image_index)
            # reconstruction
            plot_obj = add_subplot(
                image=reconstructions[image_index, :, :, 0],
                n_cols=n_cols,
                n_rows=n_rows,
                index=1 + n_rows + image_index,
            )
            # label
            plot_obj.annotate(str(image_index), xy=(0, 0))
            # test accuracy

        save_plot_as_image(path=path, filename=f"modified_classifications_{'random' if random else 'real'}")

    def save_reconstruction_plot_images(self, samples, state, path="images/epn_supervised/plots", acc=None):
        """Pushes x samples through the autoencoder to generate & visualize reconstructions

        :param samples:
            Samples that matches the following shape [n_samples, autoencoder input shape]
        :param state: str
            State of the training
        :param path: str
            Path to the directory where the plots are getting stored.
        :param acc: float
            Test accuracy of the classifier in the autoencoder
        :return:
            None
        """
        n_rows = samples.shape[0]
        n_cols = 3 if acc else 2
        reconstructions = self.autoencoder.predict(samples)
        plt.figure(figsize=(n_rows * 1.5, n_cols))
        for image_index in range(n_rows):
            # orig image
            _ = add_subplot(image=samples[image_index, :, :, 0], n_cols=n_cols, n_rows=n_rows, index=1 + image_index)
            # reconstruction
            plot_obj = add_subplot(
                image=reconstructions[1][image_index, :, :, 0],
                n_cols=n_cols,
                n_rows=n_rows,
                index=1 + n_rows + image_index,
            )
            # label
            plot_obj.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, 0))
            # test accuracy

        if acc:
            subplot = plt.subplot(n_cols, n_rows, 1 + 2 * n_rows)
            subplot.axis("off")
            subplot.text(0.2, 0.5, f"Test accuracy: {round(acc, 2)}%", fontsize=18)

        save_plot_as_image(path=path, filename=state)

    def save_fake_sample_plot_images(
        self, x_fake=None, labels=None, epoch=-1, n_samples=100, path="images/epn_supervised/plots"
    ):
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
        n_columns = math.ceil(math.sqrt(n_samples)) if n_samples > 10 else 1
        n_rows = math.ceil(n_samples / n_columns) if n_samples > 10 else n_samples
        if x_fake is None or labels is None:
            x_fake, _, labels = self.generate_fake_samples(n_samples)

        labels_numerical = tf.argmax(labels, axis=1).numpy()
        plt.figure(figsize=(n_rows, n_columns))
        for i in range(n_samples):
            plot_obj = add_subplot(image=x_fake[i, :, :, 0], n_cols=n_columns, n_rows=n_rows, index=1 + i)
            plot_obj.annotate(str(labels_numerical[i]), xy=(0, 0))

        save_plot_as_image(path=path, filename=f"generated_plot_e{epoch + 1:03d}.png")
