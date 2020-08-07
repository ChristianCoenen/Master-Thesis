import tensorflow as tf
from matplotlib import pyplot
from tensorflow import keras
from src.generator import Generator
from src import generator
from src import discriminator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from numpy import ones
from numpy import vstack
import matplotlib.pyplot as plt
from src import datasets, test_functions
import config
import numpy as np

# Config GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(physical_devices)}")
tf.config.experimental.set_memory_growth(physical_devices[0], True) if physical_devices else None

# Get data
(x_train_norm, y_train), (x_test_norm, y_test) = datasets.get_mnist()

# Labels -> One Hot encoding
print("Shape before one-hot encoding: ", y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes=config.CLASSIFICATION_NEURONS)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
print("Shape after one-hot encoding: ", y_train.shape)

''' Autoencoder '''
g = Generator()
a_model = g.build_autoencoder()
# Show the model
a_model.summary()
keras.utils.plot_model(a_model, "../model_architecture.png", show_shapes=True)
# Compile
a_model.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam", metrics=["accuracy"])

''' Generator '''
g_model = g.build_generator()

''' Discriminator'''
d_model = discriminator.define_discriminator()

''' GAN '''


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# create the gan
gan_model = define_gan(g_model, d_model)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, n_samples=100):
    # prepare real samples
    X_real, y_real = d_model.generate_real_samples(n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = d_model.generate_fake_samples(g_model, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, n_epochs=2, n_batch=1024):
    bat_per_epo = int(60000 / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = discriminator.generate_real_samples(half_batch)
            # generate 'fake' examples
            X_fake, y_fake = g.generate_fake_samples(half_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = g.generate_latent_and_classification_points(n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model)


# train model
train(g_model, d_model, gan_model)

fake_samples = g.generate_fake_samples()
generator.plot_image(np.reshape(fake_samples[0][1], (28, 28)))

''' Tests '''
# verify generator weight sharing after training
test_functions.verify_shared_weights(g.input_layer_to_encoder_1, g.decoder_1_to_output)
test_functions.verify_shared_weights(g.encoder_1_to_latent_space, g.latent_classification_to_decoder_1,
                                     g.encoder_1_to_classification)

''' Visualization '''
# Show 10 inputs and their outputs
g.show_reconstructions(x_test_norm, n_images=15)
plt.show()
