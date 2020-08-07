from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Reshape, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical, plot_model
from numpy.random import randint, random
from numpy import concatenate, zeros, ones
from src.custom_layers import DenseTranspose
from src import datasets
import config


class EntropyPropagationNetwork:
    def __init__(self, dataset='mnist'):
        self.input_shape = config.INPUT_SHAPE
        self.latent_dim = config.LATENT_DIM
        self.encoder_dim = config.ENCODER_DIM
        self.classification_dim = config.CLASSIFICATION_DIM

        if dataset == 'mnist':
            (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = datasets.get_mnist()

        self.discriminator = self.build_discriminator()
        plot_model(self.discriminator, "images/discriminator_architecture.png", show_shapes=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.autoencoder = self.build_autoencoder()
        plot_model(self.autoencoder, "images/autoencoder_architecture.png", show_shapes=True)
        self.autoencoder.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam",
                                 metrics=["accuracy"])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Combined model

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
        input_layer_to_encoder = Dense(self.encoder_dim, activation="sigmoid", name='encoder')
        encoder_to_classification = Dense(self.classification_dim, activation="softmax", name='classification')
        encoder_to_latent_space = Dense(self.latent_dim, activation="sigmoid", name='latent_space')

        inputs = Input(shape=self.input_shape)
        x = Flatten()(inputs)

        x = input_layer_to_encoder(x)
        classification = encoder_to_classification(x)
        latent_space = encoder_to_latent_space(x)
        # Merge classification neurons and latent space neurons into a single vector via concatenation
        x = layers.concatenate([classification, latent_space])
        x = DenseTranspose(encoder_to_latent_space, encoder_to_classification, activation="sigmoid", name="decoder")(x)
        x = DenseTranspose(input_layer_to_encoder, activation="sigmoid", name="outputs")(x)
        outputs = Reshape(self.input_shape, name="reconstructions")(x)
        return Model(inputs, outputs=[classification, outputs], name="autoencoder")

    def generate_fake_samples(self, n_samples):
        x_latent = random((n_samples, self.latent_dim))
        x_classification = randint(self.classification_dim, size=n_samples)
        x_classification = to_categorical(x_classification, num_classes=self.classification_dim)
        x_input = concatenate((x_latent, x_classification), axis=1)
        # predict outputs TODO: create generator!
        x = self.generator.predict(x_input)
        # create 'fake' class labels (0)
        y = zeros((n_samples, 1))
        return x, y

    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = randint(0, self.x_train_norm.shape[0], n_samples)
        # retrieve selected images
        x = self.x_train_norm[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return x, y

    def train(self):
        self.autoencoder.summary()
        self.autoencoder.fit(self.x_train_norm, [self.y_train, self.x_train_norm], epochs=5,
                             validation_data=(self.x_test_norm, (self.y_test, self.x_test_norm)))


epn = EntropyPropagationNetwork()
epn.train()
