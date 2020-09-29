from numpy.random import randint
from epn.network import EntropyPropagationNetwork
from epn.helper import add_subplot, save_plot_as_image
from copy import deepcopy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class EntropyPropagationNetworkRL(EntropyPropagationNetwork):
    """
    A class implementing the Entropy Propagation Network architecture in an reinforcement learning setting:
    Differences between EntropyPropagationNetworkRL and EntropyPropagationNetwork:

    General:
    1. The environment on which the data was generated needs to be passed in order to visualize the reconstructions, ...

    Autoencoder:
    1.

    GAN:
    1. Instead of random latent spaces, random inputs (env_states) are generated
    2. Instead of reconstructions, latent spaces are fed into the discriminator
    """

    def __init__(self, env, **kwargs):
        """
        :param env: MazeEnv
            MazeEnv object initialized with the same maze & config that was used to generate the dataset.
        """
        super().__init__(
            dataset=kwargs.get("dataset", "maze_memories"),
            classification_dim=np.count_nonzero(env.maze.to_impassable() == 0),
            **kwargs,
        )
        self.env = env
        self.nr_tiles = self.env.maze.size[0] * self.env.maze.size[1]

    def generate_random_episode(self):
        """Generates a random episode consisting of (env_state, action, reward, next_env_state, done).

        :return env_state_obj: Maze
            Maze object containing the env_state in different formats (binary, values, rgb)
        :return action: int
            Randomly chosen action between 0 and the number of possible actions - 1.
        :return reward: int
            Received reward from the environment for taking 'action' in 'env_state'.
        :next_env_state: [bool]
            Maze object containing the next env_state in different formats (binary, values, rgb).
            The next_env_state is the state received by the environment after taking 'action' in 'env_state'
        :done: bool
            Information whether the goal was reached by taking 'action' in 'env_state' or not.
        """
        self.env.reset(randomize_start=True)
        env_state_obj = deepcopy(self.env.maze)
        action = np.random.randint(0, self.env.action_space.n)
        _, reward, done, _ = self.env.step(action)
        next_env_state_obj = deepcopy(self.env.maze)
        return env_state_obj, action, reward, next_env_state_obj, done

    def predict_next_env_state_and_latent_space(self, env_state_obj, action):
        """Given an env_state, the generator predicts the resulting env_state + latent space"""
        inputs = self.env_state_and_action_to_inputs(env_state_obj.to_valid_obs(), action)
        return self.encoder.predict(inputs)

    def env_state_and_action_to_inputs(self, env_state, action):
        action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.n).reshape(1, -1)
        inputs = np.concatenate((env_state, action_one_hot), axis=1)
        return inputs

    def train(self, epochs=5, batch_size=32, pre_train_epochs=3, train_encoder=True):
        # TODO: implement
        pass

    def visualize_trained_autoencoder_to_file(self, state, n_samples=2, path="images/plots"):
        width = n_samples
        height = 4
        plt.figure(figsize=(width, height))
        for idx in range(n_samples):
            env_state_obj, action, _, _, _ = self.generate_random_episode()
            inputs = self.env_state_and_action_to_inputs(env_state_obj.to_valid_obs(), action)
            [pred_next_env_state, reconstruction] = self.autoencoder.predict(inputs)

            # Sampled maze state + action
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx)
            plt.annotate(self.env.motions._fields[action], xy=(0.25, -0.5), fontsize="medium")

            # Sampled maze state + action with next state prediction values
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx + n_samples)
            annotate_maze(pred_next_env_state[0], env_state_obj)

            # Reconstructed maze state + action
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx + 2 * n_samples)
            annotate_maze(reconstruction[0][: -self.env.action_space.n], env_state_obj)
            actions_one_hot = reconstruction[0][-self.env.action_space.n :]
            annotate_action_values(
                n_cols=height,
                n_rows=width,
                index=1 + idx + 3 * n_samples,
                action_names=self.env.motions._fields,
                values=actions_one_hot,
            )

        save_plot_as_image(path=path, filename=f"{self.nr_tiles}{state}", dpi=300 + self.nr_tiles * 5)


def annotate_maze(values, env_state_obj):
    is_wall = env_state_obj.to_impassable()
    # The values are from row to row and not column to column that's why we start with y!
    for y in range(env_state_obj.size[0]):
        for x in range(env_state_obj.size[1]):
            # Values are only representing valid states, so just annotate if current state is not a wall
            if not is_wall[y, x]:
                # Show value & remove from array. That way we don't need a counter
                plt.annotate(
                    np.round(values[0], 1),
                    xy=(x - 0.35, y + 0.15),
                    fontsize=5 / ((env_state_obj.size[0] * env_state_obj.size[1]) / 15),
                )
                values = np.delete(values, 0)


def annotate_action_values(n_cols, n_rows, index, action_names, values):
    subplot = plt.subplot(n_cols, n_rows, index)
    subplot.axis("off")
    for idx in range(len(action_names)):
        subplot.text(0, 0.5 - idx / 4, f"{action_names[idx]}: {str(np.round(values[idx], 2))}", fontsize="x-small")
