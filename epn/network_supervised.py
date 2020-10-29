from typing import List, Tuple, Optional
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from epn.helper import add_subplot, save_plot_as_image
from epn.network import EPNetwork
import matplotlib.pyplot as plt
import numpy as np


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
    ):
        """
        :param data:
            Data that is trained on. Has to be a 4-tuple consisting of numpy arrays with shape (n_samples x data_shape)
        :param latent_dim:
            Number of latent space neurons (bottleneck layer in the Autoencoder)
        :param autoencoder_loss:
            This parameter allows to define the classification and reconstruction loss functions for the autoencoder.
        """
        super().__init__(weight_sharing, encoder_dims, discriminator_dims)
        (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = data
        self.latent_dim = latent_dim
        self.autoencoder_loss = autoencoder_loss

        # Build Autoencoder
        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder(
            encoder_input_tensors=[
                Input(shape=self.x_train_norm.shape[1:], name="encoder_input"),
            ],
            encoder_output_layers=[
                Dense(self.latent_dim, activation=LeakyReLU(alpha=0.2), name="latent_space"),
            ],
            ae_ignored_output_layer_names=["latent_space"],
        )
        self.autoencoder.compile(loss=self.autoencoder_loss, optimizer="adam", metrics=["accuracy"])

    def train_autoencoder(self, **kwargs):
        self.autoencoder.fit(self.x_train_norm, self.x_train_norm, **kwargs)

    def train(self, epochs: int, batch_size: int, steps_per_epoch: int, train_encoder: bool):
        pass

    def save_model_architecture_images(
        self, models: Optional[List[Model]] = None, path: str = "images/epn_supervised/architecture", fmt: str = "png"
    ):
        models = models if models is not None else []
        models.extend(
            [
                self.encoder,
                self.decoder,
                self.autoencoder,
            ]
        )
        super().save_model_architecture_images(models, path)

    def visualize_autoencoder_predictions_to_file(self, state, acc=None):
        self.save_reconstruction_plot_images(self.x_test_norm[20:30], state, acc=acc)

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
            add_subplot(
                image=reconstructions[image_index, :, :, 0],
                n_cols=n_cols,
                n_rows=n_rows,
                index=1 + n_rows + image_index,
            )
        save_plot_as_image(path=path, filename=state)
