import numpy as np
from gym import logger, wrappers
from pathlib import Path


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


def train(env, epochs, is_human_mode, save_to_file=False):
    memories = []

    logger.set_level(logger.INFO)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    for epoch in range(epochs):
        ob = env.reset()
        ob = np.dot(ob[..., :3], [0.299, 0.587, 0.114])
        while True:
            env.render("human") if is_human_mode else env.render("rgb_array")

            # Random action
            action = agent.act()

            next_ob, reward, done, _ = env.step(action)
            next_ob = np.dot(next_ob[..., :3], [0.299, 0.587, 0.114])
            memories.append([ob, action, reward, next_ob])
            ob = next_ob
            if done:
                break

    if save_to_file:
        print("Saving memory to file!")
        save_memories_to_file(memories)


def save_memories_to_file(memories, path="data"):
    Path(path).mkdir(parents=True, exist_ok=True)
    # Convert all actions and positions to one hot encoded vectors
    np.save(f"{path}/car_racing_data", memories)
