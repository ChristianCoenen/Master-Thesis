import tensorflow as tf
from tensorflow import keras
from src.generator import Generator
from src import generator
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


''' Generator '''
g = Generator()
g_model = g.build_autoencoder()
g_generator = g.build_generator()
# Show the model
g_model.summary()
keras.utils.plot_model(g_model, "../model_architecture.png", show_shapes=True)
# Compile & Train
g_model.compile(loss=["mean_squared_error", "binary_crossentropy"], optimizer="adam", metrics=["accuracy"])
history = g_model.fit(x_train_norm, [y_train, x_train_norm], epochs=5,
                      validation_data=(x_test_norm, (y_test, x_test_norm)))

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
