from epn import datasets
from epn.network_supervised import EPNetworkSupervised
import tensorflow as tf
import os
import random
import numpy as np

# Set a seed value
seed_value = 30
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ["PYTHONHASHSEED"] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

data = datasets.get_mnist(fashion=False)
# Configure and train the Entropy Propagation Network
epn = EPNetworkSupervised(
    data=data,
    latent_dim=20,
    autoencoder_loss=["categorical_crossentropy", "binary_crossentropy"],
    weight_sharing=True,
    encoder_dims=[1024, 512, 256],
    discriminator_dims=[1024, 512, 256],
)
# Only run the following line if you have graphviz installed, otherwise make sure to remove it or comment it out
# epn.save_model_architecture_images()

epn.visualize_autoencoder_predictions_to_file(state="pre_autoencoder_training")
epn.train_autoencoder(epochs=3, batch_size=32, validation_split=0.1)
epn.train(epochs=5, batch_size=32, steps_per_epoch=500, train_encoder=True)
acc = epn.evaluate()
epn.visualize_autoencoder_predictions_to_file(state="post_autoencoder_training", acc=acc)
epn.create_modified_classification_plot(sample_idx=4)
epn.create_modified_classification_plot(random=True)
