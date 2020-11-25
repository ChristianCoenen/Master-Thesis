import gym
import epn.helper as helper
from rl.trainer import Trainer
from maze.predefined_maze import *
from maze import Maze

seed_value = 30
helper.set_seeds(seed_value)

env = gym.make("maze:Maze-v0", maze=Maze(x3))
trainer = Trainer(env, seed=seed_value)
trainer.train(epochs=100, max_episode_length=50, is_human_mode=False, randomize_start=True, save_to_file=True)
