import gym
import numpy as np
from epn import datasets
from rl2.network_rl import EPNetworkRL
import os
import random
import tensorflow as tf

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


env = gym.make("CarRacing-v0")
dataset_path = f"./data/car_racing_data.npy"
data = datasets.get_car_racing_memories(dataset_path)
epn = EPNetworkRL(
    env=env,
    data=data,
    latent_dim=50,
    encoder_dims=[2000, 1000],
    discriminator_dims=[200, 200],
    weight_sharing=False,
    autoencoder_loss=[
        "binary_crossentropy",
        "mean_squared_error",
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
        "binary_crossentropy",
    ],
    seed=seed_value,
)
# epn.save_model_architecture_images()
# epn.visualize_outputs_to_file(state="pre_autoencoder_training")
epn.train_autoencoder(epochs=100, batch_size=8)
epn.visualize_outputs_to_file(state="post_autoencoder_training")
epn.train(epochs=50, batch_size=8, steps_per_epoch=100, train_encoder=True)
epn.visualize_outputs_to_file(state="post_gan_training")
