import gym
from .color_style import ObjectColor
from .object import Object
from .motion import VonNeumannMotion, MichaudMotion
from .motion import MooreMotion
from .maze import BaseMaze
from .env import Maze, MazeEnv

gym.envs.register(id="Maze-v0", entry_point="maze:MazeEnv")
