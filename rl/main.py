import gym
import epn.helper as helper
from maze.predefined_maze import *
from maze import Maze
from epn import datasets
from rl.network_rl import EPNetworkRL

seed_value = 30
helper.set_seeds(seed_value)


env = gym.make("maze:Maze-v0", maze=Maze(x3))
dataset_path = f"./data/{env.maze.__repr__()}.npy"
data = datasets.get_maze_memories(dataset_path, shuffle=True)
epn = EPNetworkRL(
    env=env,
    data=data,
    latent_dim=5,
    encoder_dims=[200, 200],
    discriminator_dims=[200, 200],
    weight_sharing=True,
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
epn.save_model_architecture_images()
epn.visualize_outputs_to_file(state="pre_autoencoder_training")
epn.train_autoencoder(epochs=100, batch_size=8)
epn.visualize_outputs_to_file(state="post_autoencoder_training")
epn.train(epochs=50, batch_size=8, steps_per_epoch=100, train_encoder=True)
epn.visualize_outputs_to_file(state="post_gan_training")