from numpy.random import randint
from epn.network import EntropyPropagationNetwork
from epn.helper import add_subplot, save_plot_as_image
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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
        self.nr_valid_tiles = np.count_nonzero(env.maze.to_impassable() == 0)
        self.nr_tiles = env.maze.size[0] * env.maze.size[1]
        self.env = env
        super().__init__(
            dataset=kwargs.get("dataset", "maze_memories"),
            classification_dim=self.nr_valid_tiles,
            **kwargs,
        )

        # Create the special GAN network that works against the encoder instead of the decoder
        self.special_discriminator = self.build_discriminator(custom_input_shape=self.classification_dim)
        self.special_discriminator.compile(
            loss=["binary_crossentropy", "mean_squared_error"], optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
        )
        # Special GAN model that uses the encoder for the inputs instead of the decoder
        self.special_discriminator.trainable = False
        self.special_gan = self.build_special_gan(self.encoder, self.special_discriminator)
        self.special_gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))
        # TODO: I think this can be in a method or so, but overwriting super class method results in error
        path = "images/architecture"
        plot_model(
            self.special_discriminator,
            f"{path}/special_discriminator_architecture.png",
            show_shapes=True,
            expand_nested=True,
        )
        plot_model(self.special_gan, f"{path}/special_gan_architecture.png", show_shapes=True, expand_nested=True)

    def build_special_gan(self, encoder, discriminator):
        inputs = Input(shape=self.classification_dim + self.env.action_space.n, name="gan_inputs")
        encoded = encoder(inputs)
        # Only use the classification outputs from the encoder in the GAN setting
        discriminated = discriminator(encoded[:, : self.classification_dim])
        return Model(inputs, discriminated)

    def _generate_random_episode(self, get_obj):
        """Generates a random episode consisting of (env_state, action, reward, next_env_state, done).

        :param get_obj: bool
            Whether to return a copy of the maze object in a specific env state or the env state (without walls) itself.

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
        action = np.random.randint(0, self.env.action_space.n)
        self.env.reset(randomize_start=True)
        env_state = deepcopy(self.env.maze) if get_obj else self.env.maze.to_valid_obs()
        next_env_state, reward, done, _ = self.env.step(action)
        next_env_state = deepcopy(self.env.maze) if get_obj else self.env.maze.to_valid_obs()
        return env_state, action, reward, next_env_state, done

    def generate_random_episodes(self, get_obj, n):
        if get_obj:
            env_state = np.empty((n, 1), dtype=type(self.env))
            next_env_state = np.empty((n, 1), dtype=type(self.env))
        else:
            env_state = np.empty((n, self.nr_valid_tiles), dtype=int)
            next_env_state = np.empty((n, self.nr_valid_tiles), dtype=int)

        action = np.empty((n, 1), dtype=int)
        reward = np.empty((n, 1), dtype=float)
        done = np.empty((n, 1), dtype=bool)

        for i in range(n):
            env_state[i], action[i], reward[i], next_env_state[i], done[i] = self._generate_random_episode(get_obj)
        return env_state, action, reward, next_env_state, done

    def predict_next_env_state_and_latent_space(self, env_state, action):
        """Given an env_state, the generator predicts the resulting env_state + latent space"""
        inputs = self.env_state_and_action_to_inputs(env_state, action)
        return self.encoder.predict(inputs)

    def env_state_and_action_to_inputs(self, env_state, action):
        action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.n)
        inputs = np.concatenate((env_state, action_one_hot), axis=1)
        return inputs

    def train(self, epochs=5, batch_size=2, steps_per_epoch=100, train_encoder=True):
        half_batch = int(batch_size / 2)
        # manually enumerate epochs
        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(steps_per_epoch):
                """ Discriminator training """
                # create training set for the discriminator
                env_state, action, _, next_env_state, _ = self.generate_random_episodes(get_obj=False, n=half_batch)
                pred = self.predict_next_env_state_and_latent_space(env_state, action)
                next_env_state_pred = pred[:, : self.nr_valid_tiles]
                real_labels, fake_labels = (np.ones(shape=(half_batch, 1)), np.zeros(shape=(half_batch, 1)))
                # This is a bit weird but the inputs to the encoder are always valid states
                x_discriminator = np.vstack((env_state, env_state))
                y_discriminator = np.vstack((next_env_state, next_env_state_pred))
                labels = np.vstack((real_labels, fake_labels))
                # One-sided label smoothing (not sure if it makes sense in rl setting)
                # y_discriminator[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _, _, _, _ = self.special_discriminator.train_on_batch(
                    x_discriminator, [labels, y_discriminator]
                )

                """ Generator training (discriminator weights deactivated!) """
                # prepare points in latent space as input for the generator
                env_state, action, _, next_env_state, _ = self.generate_random_episodes(get_obj=False, n=batch_size)
                env_state = self.env_state_and_action_to_inputs(env_state, action)
                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1 we have an error of 0 and vice versa
                real_labels = np.ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss, _, _ = self.special_gan.train_on_batch(env_state, [real_labels, next_env_state])

                """ Autoencoder training """
                # this might result in the discriminator outperforming the generator depending on architecture
                self.autoencoder.train_on_batch(env_state, [next_env_state, env_state]) if train_encoder else None

                # summarize loss on this batch
                print(
                    f">{i + 1}, {j + 1:0{len(str(steps_per_epoch))}d}/{steps_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}"
                )

    def visualize_trained_autoencoder_to_file(self, state, n_samples=10, path="images/plots"):
        width = n_samples
        height = 4
        plt.figure(figsize=(width, height))
        for idx in range(n_samples):
            env_state_obj, action, _, _, _ = self._generate_random_episode(get_obj=True)
            inputs = self.env_state_and_action_to_inputs(env_state_obj.to_valid_obs(), np.array(action).reshape(1, -1))
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
