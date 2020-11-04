import gym
from rl.trainer import Trainer
from maze.predefined_maze import *
from maze import Maze

env = gym.make("maze:Maze-v0", maze=Maze(x3))
trainer = Trainer(env)
trainer.train(
    epochs=10, data_size=1, epsilon=0.1, max_episode_length=50, max_memory=1, is_human_mode=False, save_to_file=True
)
