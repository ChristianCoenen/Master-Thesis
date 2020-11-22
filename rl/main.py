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
epn.save_model_architecture_images()
# epn.train_generator(epochs=100, batch_size=8)
epn.visualize_outputs_to_file(state="post_supervised_gen", test_or_train_data="train")
epn.visualize_outputs_to_file(state="post_supervised_gen", test_or_train_data="test")
# epn.visualize_outputs_to_file(state="pre_training", test_or_train_data="test")
# epn.visualize_outputs_to_file(state="pre_training", test_or_train_data="train")
epn.train(epochs=60, batch_size=4, steps_per_epoch=200, train_generator_supervised=True)
