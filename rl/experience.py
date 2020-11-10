import numpy as np
import tensorflow as tf
from pathlib import Path
from copy import deepcopy


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.permanent_memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode, unique_data=True):
        # episode = [env_state, action, reward, env_state_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

        # For the datafile, we store it in a slightly different format to simplify working with the dataset
        perm_episode = deepcopy(episode)
        perm_episode[1] = [*episode[1]][0]
        perm_episode[5] = [*episode[5]][0]

        # this snippet won't win a style price I guess, but checks if the episode is already present in the dataset
        # & it's logically quite time consuming, alternative would be to filter before writing the file
        if unique_data and any(
            [
                np.array_equal(perm_episode[1], unique_arr[1]) and perm_episode[2] == unique_arr[2]
                for unique_arr in self.permanent_memory
            ]
        ):
            return

        self.permanent_memory.append(perm_episode)

    def save_memories_to_file(self, maze, filename, path="data"):
        Path(path).mkdir(parents=True, exist_ok=True)
        # Convert all actions and positions to one hot encoded vectors
        for idx, episode in enumerate(self.permanent_memory):
            # Action
            self.permanent_memory[idx][2] = tf.keras.utils.to_categorical(episode[2], num_classes=self.num_actions)
            # Position
            position_1 = tf.keras.utils.to_categorical(episode[0][0], num_classes=maze.size[0])
            position_2 = tf.keras.utils.to_categorical(episode[0][1], num_classes=maze.size[1])
            position = np.concatenate((position_1, position_2))
            self.permanent_memory[idx][0] = position
            # Next Position
            position_1 = tf.keras.utils.to_categorical(episode[4][0], num_classes=maze.size[0])
            position_2 = tf.keras.utils.to_categorical(episode[4][1], num_classes=maze.size[1])
            position = np.concatenate((position_1, position_2))
            self.permanent_memory[idx][4] = position

        np.save(f"{path}/{filename}", self.permanent_memory)

    def predict(self, env_state):
        return self.model.predict(env_state)[0]

    def get_data(self, data_size=10):
        # env_state 1d size (1st element of episode)
        env_size = self.memory[0][1].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            position, env_state, action, reward, position_next, env_state_next, game_over = self.memory[j]
            inputs[i] = env_state
            # There should be no target values for actions not taken.
            targets[i] = self.predict(env_state)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(env_state_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
