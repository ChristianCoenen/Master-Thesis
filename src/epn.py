from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Reshape, concatenate, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical, plot_model
from numpy.random import randint
from numpy import concatenate, zeros, ones
from src.custom_layers import DenseTranspose
from src import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import config
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

    def __init__(self, dataset='mnist', weight_sharing=True, encoder_dims=None, latent_dim=40, classification_dim=10):
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

        if dataset == 'mnist':
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist()
        elif dataset == 'fashion_mnist':
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist(fashion=True)
        else:
            raise ValueError("Unknown dataset!")

        self.input_shape = self.x_train_norm.shape[1:]

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder()

        self.autoencoder.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam",
                                 metrics=["accuracy"])

        # GAN model (decoder & discriminator) - For the GAN model we will only train the generator
        self.discriminator.trainable = False
        self.gan = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        # only set config.GRAPHVIZ to true if you have it installed (see README)
        self.save_model_architecture_images() if config.GRAPHVIZ else None

    def build_discriminator(self):
        """ Creates a discriminator model.

        Leaky ReLU is recommended for Discriminator networks.
        'Within the discriminator we found the leaky rectified activation to work well ...'
            — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015.

        :return: Discriminator model
        """
        model = Sequential(name="discriminator")
        model.add(Flatten())
        model.add(Dense(1024, input_dim=np.prod(self.input_shape), activation=LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def build_autoencoder(self):
        """ Creates an encoder, decoder and autoencoder model.

        :return: Encoder model, Decoder mode, Autoencoder model
        """
        # Define shared layers
        encoder_layers = []
        for idx, encoder_layer in enumerate(self.encoder_dims):
            encoder_layers.append(Dense(self.encoder_dims[idx], activation=LeakyReLU(alpha=0.2), name=f'encoder_{idx}'))
        encoder_to_classification = Dense(self.classification_dim, activation="softmax",
                                          name=f'encoder_{len(self.encoder_dims) }_classification')
        encoder_to_latent_space = Dense(self.latent_dim, activation=LeakyReLU(alpha=0.2),
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
                               activation=LeakyReLU(alpha=0.2), name="decoder_0")(decoder_inputs)
            for idx, encoder_layer in enumerate(reversed(encoder_layers)):
                if idx == len(encoder_layers) - 1:
                    x = DenseTranspose(encoder_layer, activation="sigmoid", name=f"decoder_{1 + idx}")(x)
                else:
                    x = DenseTranspose(encoder_layer, activation=LeakyReLU(alpha=0.2), name=f"decoder_{1 + idx}")(x)
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

    # define the combined generator and discriminator model, for updating the generator
    def build_gan(self):
        """ Defines the combined decoder and discriminator model, for updating the decoder

        :return:
        """
        # connect them
        gan_model = Sequential()
        gan_model.add(self.decoder)
        gan_model.add(self.discriminator)
        return gan_model

    def generate_latent_and_classification_points(self, n_samples):
        # generate random points in the latent space
        x_latent = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        label = randint(self.classification_dim, size=n_samples)
        x_classification = to_categorical(label, num_classes=self.classification_dim)
        x_input = concatenate((x_latent, x_classification), axis=1)
        return x_input, label

    def generate_fake_samples(self, n_samples):
        """ Generates fake samples for

        :param n_samples: int
            Number of samples that needs to be generated.
        :return:
            x: Random samples generator by the decoder (generator)
            y: Array with length of n_samples containing zeros (to indicate that those are fake samples)
            z: Array with length of n_samples showing the objective for the decoder for each samples
               Helps debugging whether the generated samples match the classification input (e.g. generate a 6)
        """
        # generate random points in the latent space
        x_input, label = self.generate_latent_and_classification_points(n_samples)
        # predict outputs
        x = self.decoder.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return x, y, label

    def generate_real_samples(self, n_samples):
        """ This method samples from the training data set to show real samples to the discriminator.

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
        return x, y

    def train_autoencoder(self, epochs=5):
        self.autoencoder.fit(self.x_train_norm, [self.y_train, self.x_train_norm], epochs=epochs, validation_split=0.1)
        self.save_reconstruction_plot_images(self.x_train_norm[10:20])
        self.save_fake_sample_plot_images()

    def train(self, epochs=5,  batch_size=1024, pre_train_epochs=3):
        batch_per_epoch = int(60000 / batch_size)
        half_batch = int(batch_size / 2)

        self.train_autoencoder(pre_train_epochs)

        # manually enumerate epochs
        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(batch_per_epoch):
                ''' Discriminator training '''
                # create training set for the discriminator
                x_real, y_real = self.generate_real_samples(n_samples=half_batch)
                x_fake, y_fake, labels = self.generate_fake_samples(n_samples=half_batch)
                x_discriminator, y_discriminator = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
                # One-sided label smoothing
                y_discriminator[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _ = self.discriminator.train_on_batch(x_discriminator, y_discriminator)

                ''' Generator training (discriminator weights deactivated!) '''
                # prepare points in latent space as input for the generator
                x_gan, _ = self.generate_latent_and_classification_points(batch_size)
                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1 we have an error of 0 and vice versa
                y_gan = ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch(x_gan, y_gan)

                # TODO:
                '''
                I guess it makes sense that the generated images are also used as inputs for the encoder to see
                if they are good enough so that the classifier can classify them correctly. This should be another layer
                of quality assurance to improve the generator even further
                '''

                # summarize loss on this batch
                leading_digits = len(str(batch_per_epoch))
                print(f'>{i + 1}, {j + 1:0{leading_digits}d}/{batch_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}')

            # evaluate the model performance each epoch
            self.summarize_performance(i)

    def summarize_performance(self, epoch, n_samples=100):
        """ Evaluate the discriminator, plot generated images, save generator model

        :param epoch: int
            Current training epoch.
        :param n_samples: int
            Number of samples that are plotted.
        :return:
            None
        """
        # prepare real samples
        x_real, y_real = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake, labels = self.generate_fake_samples(n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print(f'>Accuracy real: {acc_real * 100:0f}%%, fake: {acc_fake * 100:0f}%%')
        # save plot
        self.save_fake_sample_plot_images(x_fake=x_fake, labels=labels, epoch=epoch)

    def evaluate(self):
        # Evaluates the autoencoder based on the test data
        return self.autoencoder.evaluate(self.x_test_norm, [self.y_test, self.x_test_norm], verbose=0)

    def save_model_architecture_images(self, path="images/architecture"):
        """ Saves all EPN model architectures as PNGs into a defined sub folder.

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
        """ Pushes x samples through the autoencoder to generate & visualize reconstructions

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
            plot_obj = add_subplot(image=reconstructions[1][image_index, :, :, 0], n_cols=3, n_rows=n_samples,
                                   index=1 + n_samples + image_index)
            # label
            plot_obj.annotate(str(np.argmax(reconstructions[0][image_index])), xy=(0, 0))

        save_plot_as_image(path=path, filename='reconstructed_plot.png')

    def save_fake_sample_plot_images(self, x_fake=None, labels=None, epoch=-1, n_samples=100, path="images/plots"):
        """ Create and save a plot of generated images (reversed grayscale)

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

        plt.figure(figsize=(n_columns, n_rows))
        for i in range(n_samples):
            plot_obj = add_subplot(image=x_fake[i, :, :, 0], n_cols=n_columns, n_rows=n_rows, index=1 + i)
            plot_obj.annotate(str(labels[i]), xy=(0, 0))

        save_plot_as_image(path=path, filename=f'generated_plot_e{epoch + 1:03d}.png')


def add_subplot(image, n_cols, n_rows, index):
    plot_obj = plt.subplot(n_cols, n_rows, index)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    return plot_obj


def save_plot_as_image(path, filename):
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = f'{path}/{filename}'
    plt.savefig(full_path)
    plt.close()
