import gym
import epn.helper as helper
from maze.predefined_maze import *
from maze import Maze
from epn import datasets
from rl.network_rl import EPNetworkRL

seed_value = 30
helper.set_seeds(seed_value)


env = gym.make("maze:Maze-v0", maze=Maze(x10))
dataset_path = f"./data/{env.maze.__repr__()}.npy"
data = datasets.get_maze_memories(dataset_path, shuffle=True)
epn = EPNetworkRL(
    env=env,
    data=data,
    encoder_dims=[200, 200],
    discriminator_dims=[50],
    generator_loss=[
        "binary_crossentropy",
        "mean_squared_error",
    ],
    seed=seed_value,
)
epn.save_model_architecture_images()
epn.visualize_outputs_to_file(state="pre_training")
epn.train(epochs=50, batch_size=2, steps_per_epoch=100, train_generator_supervised=False)
epn.visualize_outputs_to_file(state="post_training")
