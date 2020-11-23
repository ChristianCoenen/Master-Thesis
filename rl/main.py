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
    discriminator_dims=[10, 10],
    generator_loss=[
        "binary_crossentropy",
        "mean_squared_error",
    ],
    seed=seed_value,
)
epn.train_generator(epochs=400, batch_size=4)
epn.save_model_architecture_images()
epn.train(epochs=60, batch_size=4, steps_per_epoch=200, train_generator_supervised=True)
for i in range(5):
    epn.visualize_outputs_to_file(f"trajectory{i}", trajectories=True, test_or_train_data="test")
    epn.visualize_outputs_to_file(f"trajectory{i}", trajectories=True, test_or_train_data="train")
