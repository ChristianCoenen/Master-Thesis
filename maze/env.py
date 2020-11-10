from abc import ABC
from abc import abstractmethod
import numpy as np
import gym
import random
from gym.utils import seeding
from gym.spaces import MultiBinary
from gym.spaces import Discrete
from PIL import Image

from maze import BaseMaze
from maze import Object
from .color_style import ObjectColor as Color
from maze import VonNeumannMotion


class BaseEnv(gym.Env, ABC):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 3}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self):
        self.viewer = None
        self.seed()

    @abstractmethod
    def step(self, action):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    def render(self, mode="human", max_width=300):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width / img_width
        img = Image.fromarray(img).resize([int(ratio * img_width), int(ratio * img_height)], resample=Image.NEAREST)
        img = np.asarray(img)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control.rendering import SimpleImageViewer

            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class Maze(BaseMaze):
    def __init__(self, maze):
        self.x = maze
        super().__init__()

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object("free", 0, Color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object("obstacle", 1, Color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        # visited = Object('visited', 2, Color.box, False, [])
        goal = Object("goal", 3, Color.goal, False, [])
        agent = Object("agent", 4, Color.agent, False, [])
        return free, obstacle, goal, agent


class MazeEnv(BaseEnv):
    def __init__(self, maze):
        super().__init__()
        # Initialize Maze
        self.maze = maze
        self.width, self.height = self.maze.size
        self.start_pos = [0, 0]
        self.goal_pos = [self.width - 1, self.height - 1]
        # Action space
        self.motions = VonNeumannMotion()
        self.action_space = Discrete(len(self.motions))
        # Observation space
        self.observation_space = MultiBinary(self.maze.size[0] * self.maze.size[1])

        self.reset()

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        was_visited_before = self._was_visited_before(new_position) if hasattr(self.maze.objects, "visited") else None
        if valid:
            self.maze.objects.agent.positions = [new_position]
            if was_visited_before is not None and not was_visited_before:
                self.maze.objects.visited.positions += [new_position]
        else:
            new_position = current_position
        if self._is_goal(new_position):
            reward = +1.0
            done = True
        elif not valid:
            reward = -0.6
            done = False
        elif was_visited_before:
            reward = -0.2
            done = False
        else:
            reward = -0.04
            done = False
        return self.maze.to_obs(), reward, done, {}

    def reset(self, randomize_start=False):
        if hasattr(self.maze.objects, "visited"):
            self.maze.objects.visited.positions = [self.start_pos]
        if randomize_start:
            self.maze.objects.agent.positions = [list(random.choice(self.maze.objects.free.positions))]
        else:
            self.maze.objects.agent.positions = [self.start_pos]
        self.maze.objects.goal.positions = [self.goal_pos]
        return self.maze.to_obs()

    def _was_visited_before(self, position):
        if position in self.maze.objects.visited.positions:
            return True
        else:
            return False

    def _is_valid(self, position):
        non_negative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        if non_negative and within_edge:
            passable = not self.maze.to_impassable()[position[0]][position[1]]
            return passable
        else:
            return False

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
